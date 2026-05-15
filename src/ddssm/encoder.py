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


class GaussLatentInit(nn.Module):
    """Initialization module producing left-pad latents when needed
    in contextProducer (for early-sequence)
    """

    def __init__(
        self,
        latent_dim: int,  # d
        j: int,  # latent history length
        emb_time_dim: int,  # E_t TODO is this missing the covariate dim?
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.j = j
        self.emb_time_dim = emb_time_dim

    def sample_pad_latents(
        self,
        time_embed: torch.Tensor,  # (B, T, E_t)
        time_idx: torch.Tensor,  # (B,)
        S: int,  # number of paths
    ) -> torch.Tensor:
        """Sample pad latents for missing history positions.

        Returns:
          pad_z: (B, S, d, j)
        """
        device = time_embed.device
        B = time_idx.shape[0]

        # time embedding at current t
        t_emb = time_embed[torch.arange(B, device=device), time_idx, :]  # (B, E_t)

        # get pad parameters from Gaussian head
        pad_mu = torch.zeros(B, self.latent_dim, device=device, dtype=time_embed.dtype)
        pad_sigma = torch.ones(
            B, self.latent_dim, device=device, dtype=time_embed.dtype
        )

        # sample S * num_slots pad latents
        eps = torch.randn(
            B, S, self.j, self.latent_dim, device=device
        )  # (B, S, num_slots, d)
        pad_mu_exp = pad_mu.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, d)
        pad_sigma_exp = pad_sigma.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, d)
        pad_z = pad_mu_exp + pad_sigma_exp * eps  # (B, S, num_slots, d)
        pad_z = pad_z.permute(0, 1, 3, 2)  # (B, S, d, num_slots)
        return pad_z


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

        h = self.fut_sum_module.forward(
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


class BaseInitPrior(nn.Module, metaclass=abc.ABCMeta):
    """Common interface for initial priors p_η(z_{1:j} | ·).

    The state-space model is responsible for assembling KL terms.
    """

    @property
    def is_gaussian_family(self) -> bool:
        """True if this prior has tractable Gaussian marginals per z_t
        and exposes Gaussian parameters via stats.
        """
        return False

    @abc.abstractmethod
    def sample_initials(
        self,
        *,
        time_embed: torch.Tensor,  # (B, T, E_t)
        start_idx: torch.Tensor,  # (B,) index of the "last" init step (e.g. 0)
        S: int = 1,
        covariates: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, GaussianStats]:
        """Autoregressively sample initial j latents:

            z_1, ..., z_j  ~  p_η

        Returns:
            zs_init   : (B, S, d, j)
            logp_init : (B, S, j)   log p_η(z_t^{(s)} | ·)
            stats     : GaussianStats (e.g. mus/logvars per step), may be empty.
        """
        ...

    @abc.abstractmethod
    def log_prob(
        self,
        z: torch.Tensor,  # (B, d) or (B*S, d)
        z_padding: Optional[torch.Tensor] = None,  # (B, d, j)
        *,
        time_embed: torch.Tensor,  # (B, T, E_t) or (B*S, T, E_t)
        time_idx: torch.Tensor,  # (B,) or (B*S,)
        z_hist: torch.Tensor,  # (B, d, j) or (B*S, d, j)
        pad_mask: Optional[torch.Tensor] = None,  # (B, j) or (B*S, j)
        covariates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Log p_η(z | z_hist, ·) evaluated at given z.

        Returns:
            logp: (B,) or (B*S,) matching the leading batch of z.
        """
        ...

    def log_prob_initials(
        self,
        z_init: torch.Tensor,  # (B, S, d, T_init)
        time_embed: torch.Tensor,  # (B, T, E_t)
    ) -> torch.Tensor:
        """Computes log p_η(z_{1:T_init}) for a given sequence of latents.

        This method handles the autoregressive nature of the prior internally.

        Returns:
            logp_init: (B, S, T_init)
        """
        raise NotImplementedError(
            "Initial sequence log_prob not implemented for this prior."
        )

    def params(
        self,
        *,
        time_embed: torch.Tensor,  # (B, T, E_t)
        time_idx: torch.Tensor,  # (B,)
        z_hist: torch.Tensor,  # (B, d, j)
        z_padding: Optional[torch.Tensor] = None,
        covariates: Optional[torch.Tensor] = None,
    ) -> GaussianStats:
        """For Gaussian-family priors, return distribution parameters
        at a single step t, conditioned on history.

        Default: not implemented.
        """
        raise NotImplementedError("Params not available for this prior.")

    def compute_init_loss(
        self,
        *,
        zs_init: torch.Tensor,  # (B, S, d, steps)
        logq_init: torch.Tensor | None,  # (B, S, steps) or None
        enc_stats: GaussianStats,  # may be empty for non-Gaussian encoders
        time_embed: torch.Tensor,  # (B, T, E_t)
        start_idx: torch.Tensor,  # (B,)
        covariates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute initialization loss terms over the first `steps` latents.

        Returns a dict with keys:
          - 'loss': scalar tensor
          - 'entropy': scalar tensor (negative entropy contribution)
          - 'vhp': scalar tensor (0 if not applicable)
        """
        raise NotImplementedError(
            "Init loss computation not implemented for this prior."
        )


class GaussianInitPrior(BaseInitPrior):
    """Gaussian initialization prior for the first j latent states.

    Same structural components as GaussianEncoder, but:
      - no future summary h_fut
      - no fut mask (only pad mask)
      - operates on a sequence of length j: [t-j, ..., t-1]
      - uses Variational hiererchical prior with latent_init module to pad missing history
    """

    def __init__(
        self,
        latent_dim: int,  # d
        j: int,  # latent history length
        emb_time_dim: int,  # E_t
        covariate_dim: int = 0,
        hidden_dim: int = 64,
        pad_mask_emb_dim: int = 8,
        context: Callable[..., ContextProducer] | None = None,
        aux_context: Callable[..., ContextProducer] | None = None,
        gaussian_head: Callable[..., GaussianHead] | None = None,
        aux_posterior_head: Callable[..., GaussianHead] | None = None,
    ) -> None:

        super().__init__()
        if context is None:
            context = partial(ContextProducer, channels=8, num_layers=2)
        if aux_context is None:
            aux_context = partial(ContextProducer, channels=8, num_layers=2)
        if gaussian_head is None:
            gaussian_head = partial(GaussianHead, clamp_logvar_min=-10.0)
        if aux_posterior_head is None:
            aux_posterior_head = partial(GaussianHead, clamp_logvar_min=-10.0)

        self.hidden_dim = hidden_dim  # H
        self.latent_dim = latent_dim
        self.j = j
        self.emb_time_dim = emb_time_dim
        self.covariate_dim = covariate_dim
        self.pad_mask_emb_dim = pad_mask_emb_dim
        self.mask_emb_dim = self.pad_mask_emb_dim  # only pad mask here

        combined_dim = self.hidden_dim
        self.context_producer_init = context(
            combined_dim=combined_dim,
            mask_tot_dim=self.mask_emb_dim,
            emb_time_dim=self.emb_time_dim + self.covariate_dim,
            combined_len=self.j,  # length j
        )

        self.context_producer_aux = aux_context(
            combined_dim=combined_dim,
            mask_tot_dim=0,
            emb_time_dim=self.emb_time_dim + self.covariate_dim,
            combined_len=self.j,  # length j
            skip_mask=True,
        )

        self.latent_init = GaussLatentInit(
            latent_dim=latent_dim, j=j, emb_time_dim=emb_time_dim
        )

        # Project z history: d -> H
        self.z_hist_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        # Pad mask embedding only
        self.pad_mask_embed = nn.Linear(1, self.pad_mask_emb_dim)

        head_in_dim = self.context_producer_init.channels * self.hidden_dim

        self.gaussian_head = gaussian_head(
            in_features=head_in_dim,
            out_features=self.latent_dim,
        )

        # --- var posterior q_Φ(z_{-j+1:0} | z_{1:j}) ---
        # Takes z_{1:j} (flattened) and outputs mean/logvar for j aux (prev) latents
        # diagonal gaussian parameters
        aux_input_dim = self.latent_dim
        aux_hidden_dim = self.hidden_dim

        self.aux_proj = nn.Linear(aux_input_dim, aux_hidden_dim)
        aux_head_in_dim = self.context_producer_aux.channels * aux_hidden_dim

        self.aux_posterior_head = aux_posterior_head(
            in_features=aux_head_in_dim,
            out_features=self.latent_dim * self.j,
        )

        self.context_producer_init = maybe_compile(
            self.context_producer_init, dynamic=True
        )
        self.context_producer_aux = maybe_compile(
            self.context_producer_aux, dynamic=True
        )

    def aux_posterior_params(
        self,
        z_init: torch.Tensor,  # (B, d, j) or (BS, d, j)
        time_embed: torch.Tensor,  # (B, T, E_t)
        covariates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute q_Φ(z_{-j+1:0} | z_{1:j}) parameters.

        Returns:
            aux_mu: (B, d, j) mean for auxiliary latents
            aux_logvar: (B, d, j) log-variance for auxiliary latents
        """
        B = z_init.shape[0]

        time_init = time_embed[:, : self.j, :].permute(0, 2, 1)  # (B, E_t, j)
        if covariates is not None:
            covs = covariates[:, :, : self.j]  # (B, V, j)
            time_init = torch.cat([time_init, covs], dim=1)

        z_proj = self.aux_proj(z_init.permute(0, 2, 1))  # (B, j, H)
        z_proj = z_proj.permute(0, 2, 1)  # (B, H, j)

        # quiet dynamo recompile warning about unused mask input in this context producer, by passing
        # an empty mask tensor of the right shape (B, 0, j)
        empty_mask = torch.zeros(B, 0, self.j, device=z_init.device, dtype=z_init.dtype)

        h = self.context_producer_aux.forward(
            combined=z_proj,
            mask_embedded=empty_mask,
            hist_time_emb=time_init,
        )  # (B, C*tot_dim)

        aux_mu, aux_logvar = self.aux_posterior_head(h)

        # Reshape to (B, d, j)
        aux_mu = aux_mu.view(B, self.latent_dim, self.j)
        aux_logvar = aux_logvar.view(B, self.latent_dim, self.j)
        return aux_mu, aux_logvar

    def sample_aux_posterior(
        self,
        z_init: torch.Tensor,  # (B, d, j)
        time_embed: torch.Tensor,  # (B, T, E_t)
        covariates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample z_{-j+1:0} from q_Φ(· | z_{1:j}).

        Returns:
            z_aux: (B, d, j) sampled auxiliary latents
            aux_mu: (B, d, j)
            aux_logvar: (B, d, j)
        """
        aux_mu, aux_logvar = self.aux_posterior_params(
            z_init, time_embed, covariates=covariates
        )
        aux_sigma = (0.5 * aux_logvar).exp()
        eps = torch.randn_like(aux_mu)
        z_aux = aux_mu + aux_sigma * eps
        return z_aux, aux_mu, aux_logvar

    def hierarchical_kl(
        self,
        z_init: torch.Tensor,  # (B, S, d, j) VP samples
        time_embed: torch.Tensor,  # (B, T, E_t)
        start_idx: torch.Tensor,  # (B,)
        covariates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute hierarchical prior bound for initialization loss.

        This method implements the VHP term:

            KL[q_Φ(z_aux | z_init) || p(z_aux)] - E_q[log p_η(z_init | z_aux)]

        by sampling z_aux ~ q_Φ(z_aux | z_init), then computing the KL and
        log-prob terms.

        Args:
            z_init: (B, S, d, j) samples from encoder z_{1:j} ~ q_ϕ
            time_embed: (B, T, E_t) time embeddings

        Returns the negative ELBO contribution:
            KL[q_Φ(z_aux | z_init) || p(z_aux)] - E_q[log p_η(z_init | z_aux)]

        averaged over batch and samples.
        """
        B, S, d, j = z_init.shape
        device = z_init.device
        BS = B * S
        tdim = time_embed.size(1)

        # Flatten B, S
        z_init_bs = z_init.reshape(BS, d, j)  # (BS, d, j)
        time_embed_exp = (
            time_embed
            .unsqueeze(1)
            .expand(-1, S, -1, -1)
            .reshape(BS, tdim, self.emb_time_dim)
        )

        if covariates is not None:
            covariates_exp = (
                covariates
                .unsqueeze(1)
                .expand(-1, S, -1, -1)
                .reshape(BS, covariates.size(1), covariates.size(2))
            )
        else:
            covariates_exp = None

        start_idx_exp = start_idx.unsqueeze(1).expand(-1, S).reshape(BS)

        # Sample auxiliaries from q_Φ
        z_aux, aux_mu, aux_logvar = self.sample_aux_posterior(
            z_init_bs, time_embed_exp, covariates=covariates_exp
        )  # (BS, d, j)

        # KL[q_Φ(z_aux | z_init) || p_aux = N(0, I)]
        # = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))
        kl_aux = 0.5 * (aux_mu.pow(2) + aux_logvar.exp() - 1 - aux_logvar).sum(
            dim=(1, 2)
        )  # (BS,)

        # Compute log p_η(z_t | z_{t-j:t-1}) for t = 1..j
        # History starts with auxiliaries z_{-j+1:0}
        logp_init = torch.zeros(BS, device=device, dtype=z_init.dtype)
        z_hist = z_aux  # (BS, d, j)

        for step in range(j):
            t_idx = start_idx_exp + step  # (BS,)
            z_t = z_init_bs[:, :, step]  # (BS, d)

            # log p_η(z_t | z_{t-j:t-1})
            logp_t = self.log_prob(
                z=z_t,
                time_embed=time_embed_exp,
                time_idx=t_idx,
                z_hist=z_hist,
                covariates=covariates_exp,
            )  # (BS,)
            logp_init = logp_init + logp_t

            # Shift history: drop oldest, append z_t
            z_hist = torch.cat([z_hist[:, :, 1:], z_t.unsqueeze(-1)], dim=-1)

        # Negative ELBO: KL - log_prob
        neg_elbo = kl_aux - logp_init  # (BS,)

        return neg_elbo.view(B, S).mean()

    @property
    def is_gaussian_family(self) -> bool:
        return True

    def params(
        self,
        *,
        time_embed: torch.Tensor,  # (B, T, E_t)
        time_idx: torch.Tensor,  # (B,)
        z_hist: torch.Tensor,  # (B, d, k) where k <= j
        z_padding: Optional[torch.Tensor] = None,  # (B, d, j)
        covariates: Optional[torch.Tensor] = None,
    ) -> GaussianStats:
        """One-step prior: returns (mu_t, logvar_t) for p_η(z_t | z_hist, ·).
        Handles padding internally if z_hist length < j.
        """
        device = time_embed.device
        B, T, E_t = time_embed.shape
        assert E_t == self.emb_time_dim
        assert time_idx.shape == (B,)
        assert time_idx.dtype == torch.long

        # Handle variable history length
        k = z_hist.shape[-1]
        if k < self.j:
            assert z_padding is not None, "z_padding required when history length < j"
            num_pad = self.j - k

            pad_z = z_padding[:, :, -num_pad:]  # (B, d, num_pad)

            z_hist_full = torch.cat([pad_z, z_hist], dim=-1)  # (B, d, j)

            # Create mask: 0 for pad, 1 for real
            pad_mask = torch.cat(
                [
                    torch.zeros(B, num_pad, device=device, dtype=z_hist.dtype),
                    torch.ones(B, k, device=device, dtype=z_hist.dtype),
                ],
                dim=1,
            )
        else:
            z_hist_full = z_hist
            pad_mask = torch.ones(B, self.j, device=device, dtype=z_hist.dtype)

        # time embeddings for history only: [t-j, ..., t-1]
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

        # project z history
        z_hist_seq = z_hist_full.permute(0, 2, 1)  # (B, j, d)
        z_proj = self.z_hist_proj(z_hist_seq)  # (B, j, H)
        combined = z_proj.permute(0, 2, 1)  # (B, H, j)

        # time embeddings to (B, E_t, j)
        hist_time_emb = hist_time_emb.permute(0, 2, 1)  # (B, E_t, j)

        pad_mask_emb = self.pad_mask_embed(pad_mask.unsqueeze(-1))  # (B, j, E_pad)
        pad_mask_emb = pad_mask_emb.permute(0, 2, 1)  # (B, E_pad, j)

        x = self.context_producer_init.forward(
            combined=combined,
            mask_embedded=pad_mask_emb,
            hist_time_emb=hist_time_emb,
        )  # (B, C*tot_dim)

        mu_t, logvar_t = self.gaussian_head(x)
        return {"mus": mu_t, "logvars": logvar_t}

    def log_prob(
        self,
        z: torch.Tensor,  # (B, d) or (B*S, d)
        z_padding: Optional[torch.Tensor] = None,  # (B, d, j)
        *,
        time_embed: torch.Tensor,  # (B, T, E_t) or (B*S, T, E_t)
        time_idx: torch.Tensor,  # (B,) or (B*S,)
        z_hist: torch.Tensor,  # (B, d, j) or (B*S, d, j)
        pad_mask: Optional[torch.Tensor] = None,
        covariates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        stats = self.params(
            time_embed=time_embed,
            time_idx=time_idx,
            z_hist=z_hist,
            z_padding=z_padding,
            covariates=covariates,
        )
        mu_t = stats["mus"]
        logvar_t = stats["logvars"]
        return gaussian_log_prob(z, mu_t, logvar_t)

    def sample_initials(
        self,
        *,
        time_embed: torch.Tensor,  # (B, T, E_t)
        start_idx: torch.Tensor,  # (B,)
        S: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, GaussianStats]:
        """Generate the initial j latents autoregressively as a prior:

            z_1, ..., z_j ~ p_η(· | z_{-j+1:0})

        Returns:
            zs_init : (B, S, d, j)
            logps   : (B, S, j)
            stats   : GaussianStats with mus/logvars per time step, aggregated over S
        """
        device = time_embed.device
        B, T, E_t = time_embed.shape
        assert E_t == self.emb_time_dim
        assert start_idx.shape == (B,)
        assert start_idx.dtype == torch.long

        pad_z = self.latent_init.sample_pad_latents(
            time_embed=time_embed,
            time_idx=start_idx,
            S=S,
        )  # (B, S, d, j)

        # Vectorize S: expand inputs to (BS, ...)
        BS = B * S
        time_embed_expanded = (
            time_embed.unsqueeze(1).expand(-1, S, -1, -1).reshape(BS, T, E_t)
        )
        if covariates is not None:
            covariates_expanded = (
                covariates
                .unsqueeze(1)
                .expand(-1, S, -1, -1)
                .reshape(BS, covariates.size(1), T)
            )
        else:
            covariates_expanded = None

        pad_z_expanded = pad_z.reshape(B * S, self.latent_dim, self.j)  # (BS, d, j)
        start_idx_expanded = start_idx.unsqueeze(1).expand(-1, S).reshape(BS)

        zs_list = []
        logps_list = []
        mus_list = []
        logvars_list = []

        z_prev_paths = torch.zeros(
            BS, self.latent_dim, 0, device=device, dtype=time_embed.dtype
        )  # (BS, d, 0)

        for step in range(self.j):
            t_idx = start_idx_expanded + step  # (BS,)

            # Internal params handles padding now
            stats_t = self.params(
                time_embed=time_embed_expanded,
                time_idx=t_idx,
                z_hist=z_prev_paths,
                z_padding=pad_z_expanded,
                covariates=covariates_expanded,
            )
            mu_t = stats_t["mus"]  # (BS, d)
            logvar_t = stats_t["logvars"]  # (BS, d)
            sigma_t = (0.5 * logvar_t).exp()
            eps = torch.randn_like(mu_t)
            z_t = mu_t + sigma_t * eps  # (BS, d)
            logp_t = gaussian_log_prob(z_t, mu_t, logvar_t)  # (BS,)

            # update history
            z_prev_paths = torch.cat([z_prev_paths, z_t.unsqueeze(-1)], dim=-1)
            if z_prev_paths.shape[-1] > self.j:
                z_prev_paths = z_prev_paths[..., -self.j :]

            zs_list.append(z_t)
            logps_list.append(logp_t)
            mus_list.append(mu_t)
            logvars_list.append(logvar_t)

        # Stack time: (BS, d, j) or (BS, j)
        zs_bs = torch.stack(zs_list, dim=-1)
        logps_bs = torch.stack(logps_list, dim=-1)
        mus_bs = torch.stack(mus_list, dim=-1)
        logvars_bs = torch.stack(logvars_list, dim=-1)

        # Reshape back to (B, S, ...)
        zs_out = zs_bs.view(B, S, self.latent_dim, self.j)
        logps_out = logps_bs.view(B, S, self.j)
        mus_out = mus_bs.view(B, S, self.latent_dim, self.j)
        logvars_out = logvars_bs.view(B, S, self.latent_dim, self.j)

        stats: GaussianStats = {"mus": mus_out, "logvars": logvars_out}
        return zs_out, logps_out, stats

    def compute_init_loss(
        self,
        *,
        zs_init: torch.Tensor,  # (B, S, d, j)
        logq_init: torch.Tensor | None,  # (B, S, j) or None
        enc_stats: GaussianStats,
        time_embed: torch.Tensor,  # (B, T, E_t)
        start_idx: torch.Tensor,  # (B,)
        covariates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute initialization loss terms over the first j latents.

        This amounts to the VHP + entropy terms:


        Args:
            zs_init: (B, S, d, j)



        Returns a dict with keys:
          - 'loss': scalar tensor
          - 'entropy': scalar tensor (negative entropy contribution)
          - 'vhp': scalar tensor (0 if not applicable)
        """
        device = zs_init.device
        dtype = zs_init.dtype
        B, S, d, steps = zs_init.shape
        T = time_embed.size(1)

        # Expand time embeddings and start indices to BS

        BS = B * S
        time_embed_bs = (
            time_embed
            .unsqueeze(1)
            .expand(B, S, -1, -1)
            .reshape(BS, T, self.emb_time_dim)
        )
        start_idx_bs = start_idx.unsqueeze(1).expand(B, S).reshape(BS)

        # Pad latents for histories shorter than j
        pad_z = self.latent_init.sample_pad_latents(
            time_embed=time_embed,
            time_idx=start_idx,
            S=S,
        )  # (B, S, d, j)
        pad_z_flat = pad_z.reshape(BS, d, self.j)

        # Precompute negative entropy term
        if "logvars" in enc_stats:
            lv_init = enc_stats["logvars"][..., :steps]  # (B, S, d, steps)
            entropy = gaussian_entropy(lv_init)  # (B, S)
            L_ent = -entropy.mean()  # gaussian entropy returns the positive entropy
        else:
            assert logq_init is not None, (
                "logq_init required for MC entropy when logvars missing"
            )
            entropy = logq_init[..., :steps].sum(dim=2).mean(dim=1)  # (B,)
            L_ent = entropy.mean()  # this is the negative entropy already

        # Hierarchical prior path (VHP)
        L_vhp = self.hierarchical_kl(
            zs_init[..., :steps], time_embed, start_idx, covariates=covariates
        )

        total_loss = L_ent + L_vhp
        return {
            "loss": total_loss,
            "entropy": L_ent,
            "vhp": L_vhp,
        }


