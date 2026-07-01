"""This module implements encoders.
The encoders produce approximate posterior distributions
q_ϕ(z_{1:T} | x_{1:T}, u_{1:T}).
"""

import abc
from typing import Tuple, Literal, Callable, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ddssm.nn.futsum import FutureSummary, GRUFutureSummary
from ddssm.nn.fusions import ConcatLinearFusion
from ddssm.nn.diffnets import (
    ContextProducer as ContextProducer,  # re-exported (see transitions.py)
    ConvTimeLayer,
    CausalTransformerTimeLayer,
    ResidualBlock,
    build_feature_layer,
    )
from ddssm.nn.combiners import CompoundCombiner, BaseEncoderCombiner
from ddssm.nn.gaussians import (
    GaussianHead as GaussianHead,  # re-exported (see transitions.py / decoder.py)
    GaussianStats,
    gaussian_log_prob,
    gaussian_entropy,
)
from ddssm.nn.net_utils import TransformerEncoder, hist_abs_time_tokens
from ddssm.nn.dist_heads import BaseDistHead, GaussianDistHead
from ddssm.nn.aggregators import ContextProducerAggregator
from ddssm.nn.torch_compile import maybe_compile, maybe_compile_fn


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
        """Sample ``S`` latent paths and their per-step encoder log-densities.

        Returns:
            zs: ``(B, S, d, T)`` sampled latent paths.
            logq_paths: ``(B, S, T)`` per-step ``log q_ϕ(z_t^{(s)} | ·)``.
            stats: per-step distribution params (may be empty for
                non-Gaussian encoders).
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
    distribution parameterization lives in the ``dist_head`` slot. The
    defaults are a :class:`~ddssm.nn.combiners.CompoundCombiner`
    (``ContextProducerAggregator`` + ``ConcatLinearFusion``) and a
    :class:`~ddssm.nn.dist_heads.GaussianDistHead`.

    The name retains "Gaussian" even though the dist head is pluggable, to
    avoid mass-renaming downstream call sites.
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
        mu_mode: Literal["free", "additive"] = "additive",
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        # Gradient-checkpoint the future-summary (its one-shot O(B·T²) attention
        # is the encoder's largest activation at long histories). Recomputed in
        # backward instead of retained.
        self.grad_checkpoint = bool(grad_checkpoint)
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
        if mu_mode not in ("free", "additive"):
            raise ValueError(f"mu_mode must be 'free' or 'additive'; got {mu_mode!r}")
        self.mu_mode = mu_mode
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
        # Compile the per-step body of the autoregressive ``sample_paths`` loop —
        # it's the launch-bound bottleneck (~670 tiny ops/iter × T). Fuses the
        # combiner + dist_head into one graph per iteration. dynamic=True because
        # the latent-history length grows over the first j steps. Off-cluster (no
        # working triton) this falls back to eager; on the cluster it's ~1.3×.
        self._forward_with_stats = maybe_compile_fn(
            self._forward_with_stats, dynamic=True
        )

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

        # Distribution head: features -> (z, logq, step_params). Optional
        # persistence frame (μ = z_{t-1} + free μ); the anchor is the most-recent
        # *real* lag (zero at t=1, where pad_mask_hist[...,-1]=0).
        mean_offset = None
        if self.mu_mode == "additive":
            mean_offset = pad_mask_hist[:, -1:] * z_prev_full[..., -1]  # (B, d)
        z_t, logq_t, step_params = self.dist_head(features, mean_offset=mean_offset)
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

        obs_perm = observed_data.permute(0, 2, 1)  # (B, T, D)
        mask_perm = cond_mask.permute(0, 2, 1) if cond_mask is not None else None
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            h = checkpoint(
                self.fut_sum_module,
                obs_perm,
                mask_perm,
                h_time_embed,
                static_embed,
                use_reentrant=False,
                preserve_rng_state=False,
            )  # (B, T, C_summary)
        else:
            h = self.fut_sum_module(  # __call__ so the in-place torch.compile fires
                observed_data=obs_perm,
                observed_mask=mask_perm,
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


def _shift_right_time(x: torch.Tensor) -> torch.Tensor:
    """Shift along the last (time) axis by 1: prepend a zero column, drop the last.

    Maps a forward-time signal ``x_t`` to ``x_{t-1}`` with ``x_0 = 0`` — used both to
    feed the CausalNoiseNet only the *strict* noise prefix ``η_{<t}`` and to read the
    persistence anchor ``z_{t-1}`` out of the cumsum.
    """
    return torch.cat([x.new_zeros(*x.shape[:-1], 1), x[..., :-1]], dim=-1)


def arflow_cumsum(g, eta, logvar, clamp_logvar_min, clamp_logvar_max):
    """Additive/persistence flow over the noise. All tensors ``(..., d, T)``.

    ``z_t = z_{t-1} + g_t + σ_t⊙η_t`` (cumsum of the increment, ``z_0 = 0``);
    ``μ_t = z_{t-1} + g_t`` is the per-step conditional mean — the persistence anchor
    PLUS the innovation, so the transition's ``mu_hat = μ_t − z_{t-1} = g_t`` is the
    residual the diffusion denoises, and σ_data tracks ``Var[g]+E[σ²]`` not ``Var[z_t]``.

    Returns ``(z, mus, logvar_clamped)``.
    """
    logvar = logvar.clamp(clamp_logvar_min, clamp_logvar_max)
    sigma = (0.5 * logvar).exp()
    incr = g + sigma * eta
    z = torch.cumsum(incr, dim=-1)
    mus = _shift_right_time(z) + g
    return z, mus, logvar


class CausalNoiseNet(nn.Module):
    """IAF conditioner: ``(μ_t, logσ²_t)`` from strictly-causal noise ``n_{<t}`` and ``h``.

    The base noise ``n`` is shift-right-by-one (``n_{<t}``) and concatenated per ``(d,t)``
    with the data summary ``h_t``, projected to ``C`` channels, then run through ``L`` causal
    CSDI blocks (causal time mixer over T + feature mixer over d). The head emits
    ``(μ_t, logσ²_t)``. Because the conditioner sees only ``n_{<t}`` (the shift), the reparam
    ``z_t = μ_t + σ_t·n_t`` is a lower-triangular flow with diagonal ``∂z_t/∂n_t = σ_t`` — so
    ``log q = Σ_t gaussian_log_prob(z_t; μ_t, logσ²_t)`` is exact. ``h`` is data (bidirectional
    is fine); only the noise must be strictly causal.
    """

    def __init__(
        self,
        latent_dim: int,
        summary_dim: int,
        channels: int = 64,
        causal_layers: int = 2,
        nheads: int = 8,
        backbone: str = "transformer",
        init_logvar_bias: float = 0.0,
        stochastic_state: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.summary_dim = summary_dim
        self.channels = channels
        self.nheads = nheads
        self.backbone = backbone
        # stochastic_state=True: the IAF conditioner sees [n_{<t}, h] so μ,σ depend on the
        # noise history (q is an autoregressive flow conditioned on the realized path).
        # False: the deterministic causal encoder, μ,σ = f(h) only (z_hist amortized).
        self.stochastic_state = stochastic_state

        # Input: [n_{<t} (1, if stochastic), h_t (summary_dim)] per (d,t) → C channels.
        self.input_projection = nn.Linear(int(stochastic_state) + summary_dim, channels)

        def _make_time_layer() -> nn.Module:
            if backbone == "conv":
                return ConvTimeLayer(channels, causal=True)
            if backbone == "transformer":
                return CausalTransformerTimeLayer(channels, nheads=nheads, dropout=0.0)
            raise ValueError(
                f"backbone must be 'conv' or 'transformer'; got {backbone!r}"
            )

        # L causal CSDI blocks (ResidualBlock body ~verbatim): causal time mixer + feature
        # mixer over d + h_t as additive side-info (cond_projection). dropout=0 throughout.
        self.blocks = nn.ModuleList(
            ResidualBlock(
                side_dim=summary_dim,
                channels=channels,
                time_layer=_make_time_layer(),
                feature_layer=build_feature_layer(
                    "transformer", channels, nheads=nheads, n_layers=1, dropout=0.0
                ),
            )
            for _ in range(causal_layers)
        )

        # Head: C → 2 (μ, logσ²). Small-init weight so μ is data-responsive yet small at
        # init (q ≈ N(0, σ²), KL≈0 handoff); the data-at-input path trains μ immediately.
        self.head = nn.Conv1d(channels, 2, kernel_size=1)
        nn.init.xavier_uniform_(self.head.weight, gain=0.5)
        nn.init.zeros_(self.head.bias)
        self.head.bias.data[1] = init_logvar_bias

    def forward(self, n, h):
        """``n`` ``(BS, d, T)`` base noise; ``h`` ``(BS, T, summary_dim)`` → ``(μ, logσ²)`` ``(BS, d, T)``."""
        BS, d, T = n.shape
        # Input per (d,t): the data summary h_t, plus — for the stochastic IAF — the
        # strictly-causal noise n_{<t} (shift) so μ,σ condition on the realized path.
        h_grid = h.unsqueeze(1).expand(BS, d, T, self.summary_dim)  # (BS, d, T, M)
        if self.stochastic_state:
            n_shifted = _shift_right_time(n).unsqueeze(-1)  # (BS, d, T, 1) — n_{<t}
            inp = torch.cat([n_shifted, h_grid], dim=-1)  # (BS, d, T, 1+M)
        else:
            inp = h_grid  # (BS, d, T, M) — deterministic causal encoder
        x0 = self.input_projection(inp).permute(0, 3, 1, 2)  # (BS, C, d, T)

        # h_t as additive in-block side-info (extra data conditioning), broadcast over d.
        side_info = h.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, d, -1)

        x = x0
        skips = []
        for blk in self.blocks:
            x, skip = blk(x, side_info)
            skips.append(skip)
        x = torch.stack(skips, dim=0).sum(0) / (len(self.blocks) ** 0.5)
        x = x + x0  # skip: direct access to the (projected) input

        out = self.head(x.reshape(BS, self.channels, d * T)).reshape(BS, 2, d, T)
        return out[:, 0], out[:, 1]  # μ, logσ²


class ARFlowEncoder(BaseEncoder):
    """Parallel encoder over pre-sampled noise (drop-in for ``GaussianEncoder``).

    Draws a noise field ``η`` once, then ``z_t = μ_t + σ_t⊙η_t`` with
    ``(μ_t, logσ²_t) = CausalNoiseNet(η, c)`` in one parallel pass — no per-step
    loop. ``stochastic_state`` toggles the conditioner: ``True`` = IAF (sees the
    strictly-causal noise history ``η_{<t}`` → q is an autoregressive flow on the
    realized path); ``False`` = mean-field (``μ,σ = f(c)`` only). Identity/persistence
    baseline only (``μ_p`` handled by the transition's centering).

    ``forward_message`` chooses the data context ``c`` fed to the conditioner:

    * ``"none"`` — ``c = b_t`` (the backward summary ``b_s = F_ϕ(x_{s:T})``; default).
    * ``"fwd_data"`` — ``c = [f_t, b_t]``, adding a forward-causal data message
      ``f_t = F_ϕ(x_{1:t})`` (the clean past-only summary that the overlapping
      backward summaries cannot reconstruct).
    * ``"fwd_summary"`` — ``c = o_t``, a forward-causal pass over ``b`` (a
      deterministic analog of the AR latent path).

    The data context may use future information freely; only the NOISE stays
    strictly causal inside ``CausalNoiseNet``, so the IAF log-prob (lower-triangular
    Jacobian, diagonal ``σ_t``) is exact regardless of ``forward_message``.
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        j: int,
        emb_time_dim: int,
        use_mask: bool,
        hidden_dim: int = 64,
        covariate_dim: int = 0,
        static_covariate_dim: int = 0,
        fut_summary: Callable[..., FutureSummary] | None = None,
        channels: int = 64,
        causal_layers: int = 2,
        nheads: int = 8,
        backbone: Literal["conv", "transformer"] = "transformer",
        clamp_logvar_min: float = -7.0,
        clamp_logvar_max: float = 7.0,
        init_logvar_bias: float = 0.0,
        stochastic_state: bool = True,
        forward_message: Literal["none", "fwd_data", "fwd_summary"] = "none",
        fwd_summary: Callable[..., FutureSummary] | None = None,
        fwd_layers: int = 2,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.grad_checkpoint = bool(grad_checkpoint)
        if fut_summary is None:
            fut_summary = partial(GRUFutureSummary, summary_dim=64, num_layers=2)

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.j = j
        self.emb_time_dim = emb_time_dim
        self.covariate_dim = covariate_dim
        self.use_mask = use_mask
        self.hidden_dim = hidden_dim
        self.clamp_logvar_min = clamp_logvar_min
        self.clamp_logvar_max = clamp_logvar_max
        self.total_static_dim = static_covariate_dim
        # ARFlow assumes the persistence/identity baseline (μ_p = z_{t-1}); flag it so the
        # composition root (DDSSM_base) can reject a mismatched baseline.
        self.requires_persistence_baseline = True

        # Future summary — covariates folded into emb_time_dim exactly as GaussianEncoder
        # (encoder.py); static categoricals go straight to the summary (no combiner).
        self.fut_sum_module = fut_summary(
            data_dim=data_dim,
            emb_time_dim=emb_time_dim + covariate_dim,
            use_mask=use_mask,
            static_embed_dim=self.total_static_dim,
        )
        self.summary_dim = self.fut_sum_module.summary_dim

        # Optional forward DATA message folded into the conditioner context c. The
        # noise path stays strictly causal inside CausalNoiseNet, so a forward (or
        # backward) data context never breaks the IAF log-prob — causality is only
        # required in η, never in the data.
        self.forward_message = forward_message
        if forward_message == "fwd_data":
            # f_t = forward-causal summary of x_{1:t}; c = [f_t, b_t].
            if fwd_summary is None:
                fwd_summary = partial(
                    GRUFutureSummary, summary_dim=self.summary_dim,
                    num_layers=2, reverse_time=False,
                )
            self.fwd_sum_module = fwd_summary(
                data_dim=data_dim,
                emb_time_dim=emb_time_dim + covariate_dim,
                use_mask=use_mask,
                static_embed_dim=self.total_static_dim,
            )
            context_dim = self.summary_dim + self.fwd_sum_module.summary_dim
            self.fwd_sum_module = maybe_compile(self.fwd_sum_module, dynamic=True)
        elif forward_message == "fwd_summary":
            # o_t = forward-causal pass over the backward summary b; c = o_t.
            self.fwd_refiner = TransformerEncoder(
                d_model=self.summary_dim, nheads=nheads, num_layers=fwd_layers,
                dim_feedforward=max(self.summary_dim, 4 * self.summary_dim),
                dropout=0.0, causal=True, rope=True,
            )
            context_dim = self.summary_dim
            self.fwd_refiner = maybe_compile(self.fwd_refiner, dynamic=True)
        elif forward_message == "none":
            context_dim = self.summary_dim
        else:
            raise ValueError(
                "forward_message must be 'none', 'fwd_data', or 'fwd_summary'; "
                f"got {forward_message!r}"
            )

        self.causal_net = CausalNoiseNet(
            latent_dim=latent_dim,
            summary_dim=context_dim,  # dim of the context c the conditioner sees
            channels=channels,
            causal_layers=causal_layers,
            nheads=nheads,
            backbone=backbone,
            init_logvar_bias=init_logvar_bias,
            stochastic_state=stochastic_state,
        )

        self.fut_sum_module = maybe_compile(self.fut_sum_module, dynamic=True)
        self.causal_net = maybe_compile(self.causal_net, dynamic=True)

    @property
    def is_gaussian_family(self) -> bool:
        return True

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
        dtype = observed_data.dtype
        B, D, T = observed_data.shape
        d = self.latent_dim

        # Future summary h_t (parallel). Covariates ride in via the time-embed concat,
        # exactly as GaussianEncoder — skip this and h drops them.
        h_time_embed = time_embed
        if covariates is not None:
            h_time_embed = torch.cat(
                [h_time_embed, covariates.permute(0, 2, 1)], dim=-1
            )
        obs_perm = observed_data.permute(0, 2, 1)  # (B, T, D)
        mask_perm = cond_mask.permute(0, 2, 1) if cond_mask is not None else None
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            h = checkpoint(
                self.fut_sum_module,
                obs_perm,
                mask_perm,
                h_time_embed,
                static_embed,
                use_reentrant=False,
                preserve_rng_state=False,
            )  # (B, T, summary_dim)
        else:
            h = self.fut_sum_module(  # __call__ so the in-place torch.compile fires
                observed_data=obs_perm,
                observed_mask=mask_perm,
                t_emb=h_time_embed,
                static_embed=static_embed,
            )

        # Conditioner context c from the backward summary b (=h), optionally with a
        # forward DATA message. (Data context only — η enters causally in causal_net.)
        if self.forward_message == "fwd_data":
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                f = checkpoint(
                    self.fwd_sum_module, obs_perm, mask_perm, h_time_embed,
                    static_embed, use_reentrant=False, preserve_rng_state=False,
                )  # f_t = F_ϕ(x_{1:t})
            else:
                f = self.fwd_sum_module(
                    observed_data=obs_perm, observed_mask=mask_perm,
                    t_emb=h_time_embed, static_embed=static_embed,
                )
            context = torch.cat([f, h], dim=-1)  # (B, T, 2·summary_dim)
        elif self.forward_message == "fwd_summary":
            context = self.fwd_refiner(h)  # forward-causal pass over b → o_t
        else:
            context = h

        # Expand to S paths and draw η ONCE, per (B·S) path (not shared across S, so
        # forecast samples with S>1 differ).
        BS = B * S
        h_expanded = context.unsqueeze(1).expand(-1, S, -1, -1).reshape(BS, T, -1)
        eta = torch.randn(BS, d, T, device=device, dtype=dtype)

        mu, logvar = self.causal_net(eta, h_expanded)  # μ_t, logσ²_t  (⟂ η_{≥t})
        logvar = logvar.clamp(self.clamp_logvar_min, self.clamp_logvar_max)
        # IAF reparam with the SAME (unshifted) noise: z = F(η; x), diagonal ∂z_t/∂η_t = σ_t.
        z = mu + (0.5 * logvar).exp() * eta
        mus = mu

        # Per-step log q (sum over d): permute (BS, d, T) → (BS, T, d).
        logq = gaussian_log_prob(
            z.permute(0, 2, 1), mus.permute(0, 2, 1), logvar.permute(0, 2, 1)
        )  # (BS, T)

        zs = z.view(B, S, d, T)
        logqs = logq.view(B, S, T)
        stats: dict = {
            "mus": mus.view(B, S, d, T),
            "logvars": logvar.view(B, S, d, T),
        }
        return zs, logqs, stats

    def entropy_transition(self, stats: dict, j: int) -> torch.Tensor:
        return gaussian_entropy(stats["logvars"][..., j:]).mean()

    def entropy_init(self, stats: dict, steps: int) -> torch.Tensor:
        return gaussian_entropy(stats["logvars"][..., :steps]).mean()


class IdentityEncoder(BaseEncoder):
    """Pinned identity posterior ``q(z_t | x_t) = δ(z_t − x_t)`` (near-delta).

    Requires ``latent_dim == data_dim``: the latent frame IS the observation, so
    the diffusion transition denoises directly in OBSERVATION space — a CSDI-style
    obs-space model wrapped in the DDSSM pipeline, with the learnable encoder/decoder
    removed as a bottleneck. Use it to test whether the *latent pipeline* (not the
    transition) is what holds DDSSM below the obs-space CSDI reference.

    Param-free (contributes no optimizer group). The reported ``mus = x`` are the
    ABSOLUTE per-step means, so the transition's centered target is the persistence
    residual ``x_t − μ_p(x_{<t})``. A small FIXED log-variance keeps ``log q`` and the
    Gaussian entropy finite while the realized ``z ≈ x`` (σ ≈ 0.03 at the −7 clamp).
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        j: int,
        emb_time_dim: int,
        use_mask: bool = False,
        fixed_logvar: float = -7.0,
        **_unused,
    ) -> None:
        super().__init__()
        if latent_dim != data_dim:
            raise ValueError(
                "IdentityEncoder requires latent_dim == data_dim; got "
                f"latent_dim={latent_dim}, data_dim={data_dim}"
            )
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.j = j
        self.emb_time_dim = emb_time_dim
        self.use_mask = bool(use_mask)
        self.register_buffer(
            "fixed_logvar", torch.tensor(float(fixed_logvar)), persistent=False
        )

    @property
    def is_gaussian_family(self) -> bool:
        return True

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
        dtype = observed_data.dtype
        B, D, T = observed_data.shape
        d = self.latent_dim
        assert D == d, f"IdentityEncoder data_dim {D} != latent_dim {d}"
        BS = B * S

        mus = observed_data.unsqueeze(1).expand(B, S, d, T)  # (B,S,d,T) = x
        logvars = self.fixed_logvar.to(device=device, dtype=dtype).expand(B, S, d, T)
        sigma = (0.5 * logvars).exp()
        eta = torch.randn(B, S, d, T, device=device, dtype=dtype)
        z = mus + sigma * eta  # z ≈ x (near-delta)

        logq = gaussian_log_prob(
            z.reshape(BS, d, T).permute(0, 2, 1),
            mus.reshape(BS, d, T).permute(0, 2, 1),
            logvars.reshape(BS, d, T).permute(0, 2, 1),
        ).view(B, S, T)
        stats: dict = {"mus": mus, "logvars": logvars}
        return z, logq, stats

    def entropy_transition(self, stats: dict, j: int) -> torch.Tensor:
        return gaussian_entropy(stats["logvars"][..., j:]).mean()

    def entropy_init(self, stats: dict, steps: int) -> torch.Tensor:
        return gaussian_entropy(stats["logvars"][..., :steps]).mean()
