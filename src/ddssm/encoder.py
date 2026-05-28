"""This module implements encoders and initial priors.
The encoders produce approximate posterior distributions
q_ϕ(z_{1:T} | x_{1:T}, u_{1:T}), while the initial priors
p_η(z_{1:j} | ·) provide distributions for the first j latent states.

As these have very similar structure (initprior is like an encoder without future summary), they are defined in the same file.
"""

import abc
from functools import partial
from typing import Callable, Dict, Tuple, Optional

import torch
import torch.nn as nn

from .aggregators import ContextProducerAggregator
from .combiners import BaseEncoderCombiner, CompoundCombiner
from .dist_heads import BaseDistHead, GaussianDistHead
from .fusions import ConcatLinearFusion
from .futsum import FutureSummary, GRUFutureSummary
from .diffnets import ContextProducer
from .gaussians import (
    GaussianHead,
    GaussianStats,
    gaussian_entropy,
    gaussian_log_prob,
)
from .net_utils import hist_abs_time_tokens
from .torch_compile import maybe_compile


class BaseEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Common interface for encoders q_ϕ(z_{1:T} | x_{1:T}, u_{1:T})."""

    @property
    def is_gaussian_family(self) -> bool:
        """True if this encoder has tractable Gaussian marginals per z_t
        and supports closed-form KL/entropy w.r.t. Gaussian priors.
        """
        return False

    @abc.abstractmethod
    def sample_paths(
        self,
        observed_data: torch.Tensor,  # (B, D, T)
        time_embed: torch.Tensor,  # (B, T, E_t)
        S: int = 1,
        cond_mask: Optional[torch.Tensor] = None,
        covariates: Optional[torch.Tensor] = None,
        static_embed: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, GaussianStats]:
        """Returns:
        zs        : (B, S, d, T)
        logq_paths: (B, S, T)  (log q_ϕ(z_t^{(s)} | ·)), MC densities
        stats     : EncoderStats (may be empty for non-Gaussian encoders)
        """
        ...

    def mc_entropy_transition(
        self,
        logq_paths: torch.Tensor,  # (B, S, T)
        j: int,
    ) -> torch.Tensor:
        """Monte Carlo posterior entropy term over t=j+1..T:

            sum_t E_q[ -log q(z_t | ·) ], averaged over S and batch.

        This works for any encoder, Gaussian or not.
        """
        B, S, T = logq_paths.shape
        if j >= T:
            return torch.zeros((), device=logq_paths.device, dtype=logq_paths.dtype)
        # t=j..T-1 (0-based) -> j+1..T (1-based)
        logs = logq_paths[:, :, j:]  # (B, S, T-j)
        # average over S, then sum over time, then mean over batch
        # logs is log q. Entropy is -E[log q].
        neg_entropy = logs.mean(dim=1).sum(dim=1)  # (B,)
        return -neg_entropy.mean()

    def entropy_init(
        self,
        stats: GaussianStats,
        steps: int,
    ) -> torch.Tensor:
        """Optional closed-form entropy over the init window z_{1:steps}."""
        raise NotImplementedError(
            "Closed-form init entropy not available for this encoder."
        )

    def entropy_transition(
        self,
        stats: GaussianStats,
        j: int,
    ) -> torch.Tensor:
        """Optional closed-form entropy term for Gaussian-family encoders.

        Non-Gaussian encoders should use `mc_entropy_transition` instead.
        """
        raise NotImplementedError("Closed-form entropy not available for this encoder.")


class GaussianEncoder(BaseEncoder):
    """Encoder producing q_ϕ(z_t | z_{t-j:t-1}, h_t).

    The encoder owns the future-summary RNN and the per-step time/mask
    construction logic; the actual mixing of ``h_fut`` with the latent
    history happens inside the configurable ``combiner`` slot, and the
    distribution parameterization lives in the ``dist_head`` slot. With
    ``combiner=LegacyJointCombiner`` and ``dist_head=GaussianDistHead``
    this matches the pre-refactor encoder bit-for-bit.

    The name retains "Gaussian" for now even though the dist head is
    pluggable, to avoid mass-renaming downstream call sites.
    """

    def __init__(
        self,
        data_dim: int,  # D
        latent_dim: int,  # d
        j: int,  # latent history length
        emb_time_dim: int,  # E_t
        use_mask: bool,  # whether to use observation mask
        hidden_dim: int = 64,  # H
        fut_mask_emb_dim: int = 8,
        pad_mask_emb_dim: int = 8,
        covariate_dim: int = 0,
        static_covariate_dim: int = 0,
        combiner: Callable[..., BaseEncoderCombiner] | None = None,
        dist_head: Callable[..., BaseDistHead] | None = None,
        fut_summary: Callable[..., FutureSummary] | None = None,
    ) -> None:
        super().__init__()
        if combiner is None:
            combiner = partial(
                CompoundCombiner,
                aggregator=partial(
                    ContextProducerAggregator, channels=8, num_layers=2
                ),
                fusion=partial(ConcatLinearFusion),
            )
        if dist_head is None:
            dist_head = partial(GaussianDistHead, clamp_logvar_min=-10.0)
        if fut_summary is None:
            fut_summary = partial(GRUFutureSummary, summary_dim=64, num_layers=2)

        self.hidden_dim = hidden_dim  # H
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.covariate_dim = covariate_dim
        self.j = j
        self.emb_time_dim = emb_time_dim
        self.fut_mask_emb_dim = fut_mask_emb_dim
        self.pad_mask_emb_dim = pad_mask_emb_dim
        self.use_mask = use_mask
        self.eps = 1e-8

        # -- static categorical embeddings info --
        self.total_static_dim = static_covariate_dim

        if self.total_static_dim > 0:
            # Project D (data) -> hidden_dim (combiner spatial dim)
            self.static_proj_context = nn.Linear(data_dim, self.hidden_dim)
        else:
            self.static_proj_context = None

        # -- future summary module --
        self.fut_sum_module = fut_summary(
            data_dim=data_dim,
            emb_time_dim=emb_time_dim + self.covariate_dim,
            use_mask=use_mask,
            static_embed_dim=self.total_static_dim,
        )
        self.summary_dim = self.fut_sum_module.summary_dim

        # -- combiner: (h_fut, z_hist, time, masks) -> features --
        self.combiner = combiner(
            latent_dim=latent_dim,
            j=j,
            summary_dim=self.summary_dim,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim + self.covariate_dim,
            pad_mask_emb_dim=pad_mask_emb_dim,
            fut_mask_emb_dim=fut_mask_emb_dim,
            static_emb_dim=self.total_static_dim,
        )

        # -- dist head: features -> (z, logq, step_params) --
        self.dist_head = dist_head(
            in_features=self.combiner.out_features,
            latent_dim=self.latent_dim,
        )

        self.fut_sum_module = maybe_compile(self.fut_sum_module, dynamic=True)

    @property
    def is_gaussian_family(self) -> bool:
        return bool(self.dist_head.is_gaussian_family)

    # ---- main calls ----
    def _forward_with_stats(
        self,
        *,
        z_prev: torch.Tensor,  # (B, d, k) k <= j, encoder-sampled latent history z_{t-k:t-1}
        z_padding: Optional[torch.Tensor] = None,  # (B, d, j)
        h_fut: torch.Tensor,  # (B, C_summary) # fut summary at t
        time_embed: torch.Tensor,  # (B, T, E_t) time embeddings
        time_idx: torch.Tensor,  # (B,) current time index t
        cond_mask: Optional[torch.Tensor] = None,  # (B,D,T-t) optional conditioning mask
        covariates: Optional[torch.Tensor] = None,
        static_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """One-step inference: returns (z_t, logq_t, step_params).

        Args:
            z_prev: (B, d, k) latent history z_{t-k:t-1} with ``k <= j``
            z_padding: (B, d, j) padding latents for when ``k < j``
            h_fut: (B, C_summary) future summary at time t
            time_embed: (B, T, E_t) time embeddings
            time_idx: (B,) current time index t
            cond_mask: (B, D, T-t) optional conditioning mask for observed data
            covariates: optional (B, V, T) time-varying covariates
            static_context: optional (B, E_s, H) projected static context

        Returns:
            z_t         : (B, d) sample from q_ϕ(z_t | ·)
            logq_t      : (B,)   log q_ϕ(z_t | ·)
            step_params : dict — per-step distribution parameters from the head
        """
        device = z_prev.device

        # z_prev: (B, d, k)
        B, d, k = z_prev.shape
        assert d == self.latent_dim, f"z_prev latent dim {d} != {self.latent_dim}"

        # Left-pad history to length j and build the per-step pad mask.
        if k < self.j:
            assert z_padding is not None, "z_padding required when history length < j"
            num_pad = self.j - k
            pad_z = z_padding[:, :, -num_pad:]  # (B, d, num_pad)
            z_prev_full = torch.cat([pad_z, z_prev], dim=-1)  # (B, d, j)
            pad_mask_hist = torch.cat(
                [
                    torch.zeros(B, num_pad, device=device, dtype=z_prev.dtype),
                    torch.ones(B, k, device=device, dtype=z_prev.dtype),
                ],
                dim=1,
            )
        else:
            z_prev_full = z_prev
            pad_mask_hist = torch.ones(B, self.j, device=device, dtype=z_prev.dtype)

        assert z_prev_full.shape == (B, d, self.j)
        assert h_fut.shape == (B, self.summary_dim), (
            f"h_fut shape {h_fut.shape} != (B={B}, summary_dim={self.summary_dim})"
        )
        assert time_embed.shape[0] == B and time_embed.shape[2] == self.emb_time_dim
        assert time_idx.shape == (B,) and time_idx.dtype == torch.long

        # History time embeddings: [t-j, ..., t-1] (length j).
        hist_time_emb = hist_abs_time_tokens(
            time_embed=time_embed,
            t_idx=time_idx,
            j=self.j,
            prepend_fut=False,
            plus_one=False,
        )  # (B, j, E_t)
        if covariates is not None:
            covs = covariates.permute(0, 2, 1)  # (B, T, V)
            hist_covs = hist_abs_time_tokens(
                time_embed=covs,
                t_idx=time_idx,
                j=self.j,
                prepend_fut=False,
                plus_one=False,
            )  # (B, j, V)
            hist_time_emb = torch.cat([hist_time_emb, hist_covs], dim=-1)

        # Combiner: (h_fut, z_hist) -> features
        features = self.combiner(
            h_fut=h_fut,
            z_hist=z_prev_full,
            hist_time_emb=hist_time_emb,
            pad_mask_hist=pad_mask_hist,
            static_context=static_context,
        )

        # Distribution head: features -> (z, logq, step_params)
        z_t, logq_t, step_params = self.dist_head(features)
        return z_t, logq_t, step_params

    def sample_paths(
        self,
        observed_data: torch.Tensor,  # (B, D, T)
        time_embed: torch.Tensor,  # (B, T, E_t)
        S: int = 1,
        cond_mask: Optional[torch.Tensor] = None,
        covariates: Optional[torch.Tensor] = None,
        static_embed: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        device = observed_data.device
        B, D, T = observed_data.shape
        assert time_embed.shape == (B, T, self.emb_time_dim)
        if self.use_mask:
            assert cond_mask is not None

        # compute future summaries h_t
        h_time_embed = time_embed
        if covariates is not None:
            h_time_embed = torch.cat(
                [h_time_embed, covariates.permute(0, 2, 1)], dim=-1
            )

        h = self.fut_sum_module(  # __call__ so the in-place torch.compile fires
            observed_data=observed_data.permute(0, 2, 1),  # (B, T, D)
            observed_mask=cond_mask.permute(0, 2, 1) if cond_mask is not None else None,
            t_emb=h_time_embed,
            static_embed=static_embed,
        )  # (B, T, C_summary)

        # expand these to S paths
        BS = B * S
        h_expanded = h.unsqueeze(1).expand(-1, S, -1, -1).reshape(BS, T, -1)
        time_embed_expanded = (
            time_embed.unsqueeze(1).expand(-1, S, -1, -1).reshape(BS, T, -1)
        )
        if cond_mask is not None:
            cond_mask_expanded = (
                cond_mask.unsqueeze(1).expand(-1, S, -1, -1).reshape(BS, D, T)
            )
        else:
            cond_mask_expanded = None

        if covariates is not None:
            covariates_expanded = (
                covariates
                .unsqueeze(1)
                .expand(-1, S, -1, -1)
                .reshape(BS, covariates.size(1), T)
            )
        else:
            covariates_expanded = None

        # Prepare expanded static context for the combiner
        static_context_expanded = None
        if self.static_proj_context is not None:
            assert static_embed is not None, (
                "static_embed required when static_proj_context is defined"
            )
            se_perms = static_embed.permute(0, 2, 1)  # (B, E_static, D)
            static_context = self.static_proj_context(se_perms)  # (B, E_static, hidden_dim)
            E_s = static_context.size(1)
            static_context_expanded = (
                static_context
                .unsqueeze(1)
                .expand(-1, S, -1, -1)
                .reshape(BS, E_s, self.hidden_dim)
            )

        # initialize empty latent history paths + zero padding
        z_prev_paths = torch.zeros(
            BS, self.latent_dim, 0, device=device, dtype=observed_data.dtype
        )
        z_padding = torch.zeros(
            BS, self.latent_dim, self.j, device=device, dtype=observed_data.dtype
        )

        zs_list = []
        logqs_list = []
        step_params_list: list[dict] = []

        for t in range(T):
            t_idx = torch.full((BS,), t, dtype=torch.long, device=device)

            z_prev = z_prev_paths
            k = z_prev.shape[-1]
            if k > self.j:
                z_prev_input = z_prev[:, :, -self.j:]
            else:
                z_prev_input = z_prev  # (BS, d, k) k <= j

            h_t = h_expanded[:, t, :]  # (BS, C_summary)

            z_t_sample, logq_t, step_params = self._forward_with_stats(
                z_prev=z_prev_input,
                z_padding=z_padding,
                h_fut=h_t,
                time_embed=time_embed_expanded,
                time_idx=t_idx,
                cond_mask=cond_mask_expanded,
                covariates=covariates_expanded,
                static_context=static_context_expanded,
            )

            z_prev_paths = torch.cat([z_prev_paths, z_t_sample.unsqueeze(-1)], dim=-1)
            if z_prev_paths.shape[-1] > self.j:
                z_prev_paths = z_prev_paths[..., -self.j:]

            zs_list.append(z_t_sample)
            logqs_list.append(logq_t)
            step_params_list.append(step_params)

        zs_bs = torch.stack(zs_list, dim=-1)  # (BS, d, T)
        logqs_bs = torch.stack(logqs_list, dim=-1)  # (BS, T)

        zs = zs_bs.view(B, S, self.latent_dim, T)
        logqs = logqs_bs.view(B, S, T)

        # Head stacks per-step params along the new last (T) axis; reshape BS -> (B, S).
        stats_bs = self.dist_head.stack_stats(step_params_list)
        stats: dict = {
            k: v.view(B, S, *v.shape[1:]) for k, v in stats_bs.items()
        }
        return zs, logqs, stats

    def entropy_transition(self, stats: dict, j: int) -> torch.Tensor:
        return self.dist_head.entropy_transition(stats, j)

    def entropy_init(self, stats: dict, steps: int) -> torch.Tensor:
        return self.dist_head.entropy_init(stats, steps)


# ---- Initial Prior Interface ---- ####
