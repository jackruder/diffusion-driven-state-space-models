"""Core DDSSM model.

Owns the ELBO forward pass, encoder/decoder/transition dispatch, and the
autoregressive forecast rollout.
"""

import os
from types import SimpleNamespace
from typing import Any, final
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ddssm.model.losses import LossComponents
from ddssm.nn.net_utils import (
    time_embedding,
)
from ddssm.model.decoder import BaseDecoder
from ddssm.model.encoder import (
    BaseEncoder,
)
from ddssm.training.stages import LambdaRampConf, LrScheduleGroupConf
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.model.centering.baselines import BaseBaseline, PersistenceBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer


@dataclass
class ProbeBatch:
    """Detached latent-encoding payload reused by variance probes."""

    zs: torch.Tensor
    logq_paths: torch.Tensor
    enc_stats: dict
    time_embed: torch.Tensor
    covariates: torch.Tensor | None = None

    def as_kwargs(self) -> dict:
        """Return the payload as a keyword-argument dict for probe calls."""
        return {
            "enc_stats": self.enc_stats,
            "zs": self.zs,
            "logq_paths": self.logq_paths,
            "time_embed": self.time_embed,
            "covariates": self.covariates,
        }


def _require_persistence_baseline(encoder: nn.Module, baseline) -> None:
    """Reject a baseline mismatch for encoders that hard-code the persistence frame.

    An encoder may set ``requires_persistence_baseline = True`` to declare its posterior
    mean is framed on ``μ_p = z_{t-1}`` (e.g. ``ARFlowEncoder``'s additive cumsum, where
    the transition's ``mu_hat = mus − μ_p`` is the innovation only when ``μ_p`` is
    persistence). Composing it with any other baseline silently corrupts the ESM target
    and σ_data, so raise instead.
    """
    if not getattr(encoder, "requires_persistence_baseline", False):
        return
    if not isinstance(baseline, PersistenceBaseline):
        got = type(baseline).__name__ if baseline is not None else "None"
        raise NotImplementedError(
            f"{type(encoder).__name__} requires a PersistenceBaseline (its frame "
            f"hard-codes μ_p = z_{{t-1}}); got {got}."
        )


@final
class DDSSM_base(nn.Module):
    """Diffusion-Driven State Space Model (DDSSM).

    Implements the full variational model: encoder q_ϕ, decoder p_θ, and a
    pluggable transition p_ψ (Gaussian or diffusion-based). The initial-state
    term over the first ``j`` latents is the transition's hierarchical
    VHP-via-diffusion walk (ADR-0006), which requires an auxiliary posterior
    ``q_Φ``; there is no standalone init-prior module. The ``forward`` method
    returns the ELBO loss and its components; ``forecast`` autoregressively
    rolls out future latent states and decodes them.

    Args:
        encoder: Instantiated encoder module.
        decoder: Instantiated decoder module.
        transition: Instantiated transition module (the stage-2 slot).
        j: Number of history steps used by each module.
        data_dim: Observed data dimension D.
        latent_dim: Latent dimension d.
        emb_time_dim: Time embedding dimension. Set to ``0`` to disable the
            absolute-time conditioning path entirely (the default for the
            regular-timestep regime; reserved for future irregular-timestep
            relative-time conditioning when set ``> 0``). Every consumer
            branches on this Python int so the time-conditioning ops drop
            out of both eager and ``torch.compile`` graphs when off.
        covariate_dim: Dimension of time-varying covariates (0 = none).
        static_embed_dim: Per-feature categorical embedding size.
        num_classes_per_static: Vocabulary size per static categorical feature.
        use_observation_mask: Whether to use the observation mask in the encoder.
        mask_emb_dim: Mask embedding dimension (stored for reference).
        logvar_min: Min clamp for decoder/encoder log-variance.
        logvar_max: Max clamp for decoder/encoder log-variance.
        S: Number of Monte Carlo encoder samples.
        aux_posterior: Required ``q_Φ(z_aux | z_{1:j})`` for the init term.
        baseline: Optional (parameter-free) centering baseline for the
            transition.
        sigma_data: Optional per-t σ_data² buffer consumed by the transition.

    Raises:
        ValueError: If ``aux_posterior`` is ``None``.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        transition: nn.Module,
        j: int,
        data_dim: int,
        latent_dim: int,
        emb_time_dim: int = 16,
        covariate_dim: int = 0,
        static_embed_dim: int = 0,
        num_classes_per_static: list[int] | None = None,
        use_observation_mask: bool = True,
        mask_emb_dim: int = 8,
        logvar_min: float = -13.0,
        logvar_max: float = 13.0,
        S: int = 1,
        # --- VHP-via-diffusion + baseline-centering path (the only init path) ---
        aux_posterior: AuxPosterior | None = None,
        baseline: BaseBaseline | None = None,
        sigma_data: SigmaDataBuffer | None = None,
        # Reconstruction-loss vectorization knobs. The per-t decode is NOT
        # autoregressive (each x_t depends only on the sampled latent window),
        # so it batches over time exactly like the diffusion ESM loss.
        # ``recon_time_chunk`` = timesteps per batched decoder call (``None`` ⇒
        # all T at once); ``recon_grad_checkpoint`` checkpoints each chunk.
        recon_time_chunk: int | None = None,
        recon_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self._recon_time_chunk = recon_time_chunk
        self._recon_grad_checkpoint = bool(recon_grad_checkpoint)
        self.j = j
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.emb_time_dim = emb_time_dim

        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.S = S

        # Top level model parameters
        self.static_embed_dim = static_embed_dim
        self.num_classes_per_static = num_classes_per_static or []
        self.static_embeddings = nn.ModuleList()

        if self.static_embed_dim > 0 and self.num_classes_per_static:
            for num_classes in self.num_classes_per_static:
                emb = nn.Embedding(num_classes, self.static_embed_dim)
                # Same scale as the transition's feature embedding (1/√2),
                # replacing the default std-1 normal.
                nn.init.normal_(emb.weight, std=0.5**0.5)
                self.static_embeddings.append(emb)
            self.total_static_dim = (
                len(self.num_classes_per_static) * self.static_embed_dim
            )
        else:
            self.total_static_dim = 0

        # The init (first-j-states) term is the hierarchical VHP walk owned
        # by the transition (ADR-0006); it requires an aux posterior q_Φ.
        if aux_posterior is None:
            raise ValueError(
                "aux_posterior is required: DDSSM_base computes the initial-"
                "state term via the transition's hierarchical VHP walk "
                "(transition_kl_init), which needs q_Φ(z_aux | z_{1:j})."
            )

        # Sub-modules (already instantiated)
        self.encoder: BaseEncoder = encoder
        self.decoder = decoder
        self.transition = transition

        # --- model-v2 slots ---
        self.aux_posterior: AuxPosterior | None = aux_posterior
        self.baseline: BaseBaseline | None = baseline
        _require_persistence_baseline(encoder, baseline)
        self.sigma_data: SigmaDataBuffer | None = sigma_data

        # Consolidate the whole training-forward into a single compiled graph.
        # Wraps the outer ``forward`` (encoder → decoder → transition → loss
        # composition) so dynamo can see across sub-module boundaries and
        # fuse the small orchestration ops (reshape/detach/dict-writes)
        # that would otherwise sit at the CPU-dispatch layer between the
        # pre-compiled sub-modules.
        #
        # NOTE: ``compile_mode="reduce-overhead"`` (CUDA graphs) segfaults
        # here — likely a nested-compile conflict with the already-compiled
        # sub-modules (encoder.sample_paths, decoder.context_producer,
        # transition.diffmodel) whose independent CUDA-graph captures
        # collide with an outer capture. Keeping the default compile mode.
        from ddssm.nn.torch_compile import maybe_compile_fn as _mcf
        # inner=False marks this as the OUTER compile — it always fires
        # (subject to DDSSM_TORCH_COMPILE), whereas the sub-module compiles
        # (inner=True default) defer to DDSSM_TORCH_COMPILE_INNER. Set that
        # to 0 to make this compile the sole graph owner so mode=
        # "reduce-overhead" (CUDA graphs) can work without nested-capture
        # conflicts.
        _compile_mode_outer = os.environ.get(
            "DDSSM_TORCH_COMPILE_MODE", ""
        ).strip() or None
        # reduce-overhead / CUDA graphs require static shapes — force
        # dynamic=False when the caller asks for it. Otherwise keep
        # dynamic=True so the same compiled forward accepts training
        # AND forecast batches (they differ in T).
        _dynamic = _compile_mode_outer not in {"reduce-overhead", "max-autotune"}
        # fullgraph=True enforces zero graph breaks. Off by default because
        # ``DDSSM_base.forward`` returns a metrics dict of detached tensors
        # — inductor's generated backward gets ``None`` tangents for those
        # entries and hits ``copy_misaligned(None) → TypeError`` at the
        # bottom of the backward call. Fixing needs a forward refactor to
        # not return detached tensors (side-channel the metrics instead).
        # Opt in with DDSSM_TORCH_COMPILE_FULLGRAPH=1 to surface any new
        # graph breaks during development.
        _fullgraph = os.environ.get(
            "DDSSM_TORCH_COMPILE_FULLGRAPH", "0"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.forward = _mcf(
            self.forward,
            dynamic=_dynamic,
            compile_mode=_compile_mode_outer,
            fullgraph=_fullgraph,
            inner=False,
        )

    def _encode_latents(
        self,
        observed_data: torch.Tensor,  # (B,D,T)
        time_embed: torch.Tensor,  # (B,T,E_t)
        observation_mask: torch.Tensor | None,
        covariates: torch.Tensor | None = None,
        static_embed: torch.Tensor | None = None,  # (B,D,E_s)
    ):
        """Run encoder to obtain latent paths and optional Gaussian stats.

        Returns:
            zs        : (B, S, d, T)
            logq_paths: (B, S, T)
            enc_stats : dict, may contain 'mus', 'logvars' for Gaussian encoders
        """
        cond_mask = None
        if getattr(self.encoder, "use_mask", False):
            cond_mask = observation_mask
        zs, logq_paths, enc_stats = self.encoder.sample_paths(
            observed_data=observed_data,
            time_embed=time_embed,
            S=self.S,
            cond_mask=cond_mask,
            covariates=covariates,
            static_embed=static_embed,
        )
        return zs, logq_paths, enc_stats

    @torch.no_grad()
    def encode_for_probe(self, batch: dict) -> ProbeBatch:
        """Encode one batch and return detached tensors for variance probes."""
        observed_data = batch["observed_data"]
        observation_mask = batch["observation_mask"]
        timepoints = batch["timepoints"]
        covariates = batch.get("covariates")
        static_covariates = batch.get("static_covariates")

        te = time_embedding(timepoints, self.emb_time_dim, device=observed_data.device)
        static_embed = self._embed_static(static_covariates)
        zs, logq_paths, enc_stats = self._encode_latents(
            observed_data=observed_data,
            time_embed=te,
            observation_mask=observation_mask,
            covariates=covariates,
            static_embed=static_embed,
        )
        detached_stats = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in enc_stats.items()
        }
        return ProbeBatch(
            zs=zs.detach(),
            logq_paths=logq_paths.detach(),
            enc_stats=detached_stats,
            time_embed=te.detach(),
            covariates=None if covariates is None else covariates.detach(),
        )

    def _reconstruction_loss(
        self,
        observed_data: torch.Tensor,  # (B, D, T)
        time_embed: torch.Tensor,  # (B, T, E_t)
        zs: torch.Tensor,  # (B, S, d, T)
        observation_mask: torch.Tensor | None,
        covariates: torch.Tensor | None = None,
        static_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruction loss using decoder.log_likelihood.

        Implements:
            L_rec = sum_t E_{qϕ,S}[ -log p_θ(x_t | z_{t-j+1:t}, ·) ]

        Returns:
            L_rec : scalar
            ratio : calibration ratio E[(x−μ)^2] / E[exp(logvar)]
        """
        device = observed_data.device
        dtype = observed_data.dtype

        B, D, T = observed_data.shape
        Bz, S, d, Tz = zs.shape
        assert Bz == B and Tz == T

        if observation_mask is None:
            observation_mask = torch.ones_like(observed_data, device=device)
        assert observation_mask is not None
        BS = B * S
        j = self.j

        # Flatten the S dimension into the batch.
        zs_flat = zs.reshape(BS, d, T)
        obs_flat = observed_data.unsqueeze(1).expand(-1, S, -1, -1).reshape(BS, D, T)
        mask_flat = (
            observation_mask.unsqueeze(1).expand(-1, S, -1, -1).reshape(BS, D, T)
        )
        time_flat = time_embed.unsqueeze(1).expand(-1, S, -1, -1).reshape(BS, T, -1)
        cov_flat = (
            covariates
            .unsqueeze(1)
            .expand(-1, S, -1, -1)
            .reshape(BS, covariates.shape[1], T)
            if covariates is not None
            else None
        )
        static_flat = (
            static_embed
            .unsqueeze(1)
            .expand(-1, S, -1, -1)
            .reshape(BS, D, static_embed.shape[2])
            if static_embed is not None
            else None
        )

        # Per-t decode uses window z_{t-j+1:t} (length j). The per-t losses are
        # independent given the sampled latent path, so — exactly like the
        # diffusion ESM loss — we batch them over time CHUNKS rather than looping
        # 192× (the launch-bound bottleneck). Pre-pad the path with j-1 left
        # zeros and unfold: window[t] = [z_{t-j+1}, …, z_t], left-zero-padded for
        # t<j-1, IDENTICAL to the decoder's internal k<j pad (so the loss is
        # unchanged). (BS, d, T, j).
        if j > 1:
            pad = torch.zeros(BS, d, j - 1, device=device, dtype=zs_flat.dtype)
            zs_pad = torch.cat([pad, zs_flat], dim=-1)
        else:
            zs_pad = zs_flat
        windows = zs_pad.unfold(dimension=-1, size=j, step=1)  # (BS, d, T, j)

        total_neg_logp = torch.zeros((), device=device, dtype=dtype)
        total_obs = torch.zeros((), device=device, dtype=dtype)
        res2_sum = torch.zeros((), device=device, dtype=dtype)
        sigma2_sum = torch.zeros((), device=device, dtype=dtype)

        # Default = 1 (per-t): byte-identical to the legacy loop, including the
        # decoder-dropout RNG pattern (one call per t). Families that want the
        # speedup opt in via ``recon_time_chunk`` AND a deterministic decoder
        # (dropout=0) — required for both batch-invariance and the checkpoint.
        chunk = self._recon_time_chunk
        chunk = 1 if chunk is None else max(1, min(int(chunk), T))
        do_ckpt = (
            self._recon_grad_checkpoint and self.training and torch.is_grad_enabled()
        )

        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            cl = t1 - t0
            N = BS * cl
            # Stack the chunk's timesteps into the batch (row = bs*cl + c).
            x_c = obs_flat[:, :, t0:t1].permute(0, 2, 1).reshape(N, D)
            m_c = mask_flat[:, :, t0:t1].permute(0, 2, 1).reshape(N, D)
            zh_c = windows[:, :, t0:t1, :].permute(0, 2, 1, 3).reshape(N, d, j)
            # The decoder only ever gathers time / covariate tokens at
            # indices [t-j+1, t] (clamped at 0), so slice the shared tables
            # to the chunk's reachable range [ts, t1) and shift ``tidx``
            # into slice coordinates instead of replicating the full-T
            # tables per row (which materialised O(BS·T²·E)). Exactness:
            # when ts > 0 every unclamped gather index is ≥ ts, and when
            # the clamp can fire (t < j) ts == 0 — and the decoder's pad
            # mask ``clamp(t+1, max=j)`` saturates at j for all t ≥ j-1,
            # so the shift never changes it. Byte-identical output.
            ts = max(0, t0 - (j - 1))
            tidx = (
                torch
                .arange(t0 - ts, t1 - ts, device=device, dtype=torch.long)
                .view(1, cl)
                .expand(BS, cl)
                .reshape(N)
            )
            te_c = (
                time_flat[:, ts:t1]
                .unsqueeze(1)
                .expand(BS, cl, t1 - ts, -1)
                .reshape(N, t1 - ts, time_flat.shape[-1])
            )
            cov_c = (
                cov_flat[:, :, ts:t1]
                .unsqueeze(1)
                .expand(BS, cl, -1, -1)
                .reshape(N, cov_flat.shape[1], t1 - ts)
                if cov_flat is not None
                else None
            )
            st_c = (
                static_flat
                .unsqueeze(1)
                .expand(BS, cl, -1, -1)
                .reshape(N, D, static_flat.shape[2])
                if static_flat is not None
                else None
            )

            def _decode(x_c, zh_c, te_c, tidx, m_c, cov_c, st_c):
                return self.decoder.log_likelihood(
                    x_t=x_c,
                    z_hist=zh_c,
                    time_embed=te_c,
                    time_idx=tidx,
                    observation_mask_t=m_c,
                    covariates=cov_c,
                    static_embed=st_c,
                )

            if do_ckpt:
                logp, mu_x, logvar_x, obs_c = checkpoint(
                    _decode,
                    x_c,
                    zh_c,
                    te_c,
                    tidx,
                    m_c,
                    cov_c,
                    st_c,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                logp, mu_x, logvar_x, obs_c = _decode(
                    x_c, zh_c, te_c, tidx, m_c, cov_c, st_c
                )

            # Accumulate: mean over S, sum over (B, chunk) — same as the per-t loop.
            total_neg_logp = total_neg_logp - logp.view(B, S, cl).mean(dim=1).sum()
            total_obs = total_obs + obs_c.view(B, S, cl).mean(dim=1).sum()
            resid2 = (x_c - mu_x).pow(2)
            sigma2 = logvar_x.exp().clamp_min(1e-6)
            res2_sum = res2_sum + (resid2 * m_c).sum()
            sigma2_sum = sigma2_sum + (sigma2 * m_c).sum()

        total_obs = total_obs.clamp_min(1.0)

        # Per-sequence sum over OBSERVED entries — matches how init_kl /
        # trans_kl are aggregated (per-seq sums averaged over B) so the
        # ELBO stays a valid bound on the observed data. The previous
        # `(total_neg_logp / total_obs) * (D * T)` rescaling overstated
        # recon by `D*T / mean_obs_per_seq` under sparse masks, silently
        # breaking comparability across missingness fractions.
        L_rec = total_neg_logp / B

        # calibration ratio E[(x-μ)^2] / E[exp(logvar)]
        res2_mean = res2_sum / total_obs
        sigma2_mean = sigma2_sum / total_obs.clamp_min(1e-6)
        ratio = (res2_mean / sigma2_mean).clamp_min(1e-8)

        return L_rec, ratio

    def _init_kl_loss(
        self,
        zs: torch.Tensor,  # (B, S, d, T)
        logq_paths: torch.Tensor,  # (B, S, T)
        enc_stats: dict,  # GaussianStats
        time_embed: torch.Tensor,  # (B, T, E_t)
        covariates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Initialization loss for the first ``j`` latent steps.

        Pure pass-through to the active transition's hierarchical VHP init
        walk (:meth:`BaseTransition.transition_kl_init`). Per ADR-0006 the
        transition owns the per-step scoring, the encoder-entropy policy
        (``-H`` for closed-form, ``0`` for the ESM-cancelling diffusion
        path), and any σ_data update — the model does not gate on
        transition type. Returns ``{loss, entropy, vhp, kl_aux, loss_init,
        loss_psi}`` — ``return_psi=True`` is requested so the ψ-side
        (unit-weighted score-net) init loss is available for the split
        objective; transitions without a score net return zero for it.
        """
        del logq_paths  # the VHP path scores encoder moments, not sampled log-q
        return self.transition.transition_kl_init(
            enc_stats=enc_stats,
            zs=zs,
            aux_posterior=self.aux_posterior,
            time_embed=time_embed,
            sigma_data=self.sigma_data,
            covariates=covariates,
            return_psi=True,
        )

    def _compute_transition_kl(
        self,
        zs: torch.Tensor,  # (B, S, d, T)
        logq_paths: torch.Tensor,  # (B, S, T)
        enc_stats,
        time_embed: torch.Tensor,  # (B, T, E_t)
        covariates: torch.Tensor | None = None,
        static_covariates: torch.Tensor | None = None,
        mc_override: dict[str, Any] | None = None,
    ) -> dict:
        """Compute the transition KL term via ``self.transition``.

        The σ_data buffer is forwarded as a kwarg (the uniform ADR-0006
        interface); transitions that don't use it ignore it.

        Returns the transition's dict (at least ``"kl"``, plus any
        sub-components for logging).
        """
        # Uniform interface (ADR-0006): every transition's ``transition_kl``
        # accepts ``sigma_data``; the ones that don't use it ignore it.
        transition_kwargs: dict[str, Any] = {
            "enc_stats": enc_stats,
            "zs": zs,
            "logq_paths": logq_paths,
            "time_embed": time_embed,
            "sigma_data": self.sigma_data,
            "covariates": covariates,
        }
        if mc_override is not None:
            transition_kwargs["mc_override"] = mc_override
        return self.transition.transition_kl(**transition_kwargs)

    def _embed_static(
        self, static_covariates: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Centralized mapping of (B, D, V_s) IDs to (B, D, E_s) continuous vectors."""
        if static_covariates is None or not self.static_embeddings:
            return None
        embedded = []
        for i, emb_layer in enumerate(self.static_embeddings):
            embedded.append(emb_layer(static_covariates[..., i].long()))
        return torch.cat(embedded, dim=-1)

    @torch.no_grad()
    def log_prob(
        self,
        observed_data: torch.Tensor,
        observation_mask: torch.Tensor,
        timepoints: torch.Tensor,
        covariates: torch.Tensor | None = None,
        static_covariates: torch.Tensor | None = None,
        *,
        K: int | None = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "dopri5",
        divergence_mode: str = "exact",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Marginal log-likelihood ``log p_ψ(x_{1:T})`` via prob-flow IWAE.

        See ``model-v2.org`` § "Exact likelihood evaluation".  Composes
        the layer-1..3 primitives:

        * Trajectory proposal: ``q_φ(z_{1:T} | x_{1:T})`` via
          :meth:`_encode_latents` with ``K = self.S`` samples (override
          via the ``K`` arg).  Per-step ``log q`` already lives in
          ``logq_paths``; we sum across ``T`` for the trajectory total.
        * Per-transition: :meth:`DiffusionTransition.log_prob` for
          ``t = j..T-1`` via the probability-flow ODE.
        * Decoder: :meth:`BaseDecoder.log_likelihood` summed across
          ``t = 0..T-1``.
        * Initial state: ``log p_ψ(z_{1:j})`` via
          :meth:`DiffusionTransition.log_prob_init` (VHP IS under
          ``q_Φ``) when an ``aux_posterior`` is present; otherwise 0.

        Args:
            observed_data, observation_mask, timepoints, covariates, static_covariates:
                same shapes/semantics as :meth:`forward`.
            K: number of trajectory samples (defaults to ``self.S``).
            rtol, atol, method, divergence_mode, generator: forwarded to the
                per-transition prob-flow ODE solver.

        Returns:
            ``(B,)`` per-sequence log-likelihood estimate.
        """
        from ddssm.model.likelihood import iwae_log_likelihood

        device = observed_data.device
        dtype = observed_data.dtype
        B, _, T = observed_data.shape
        j = self.j

        static_embed = self._embed_static(static_covariates)
        time_embed = time_embedding(timepoints, self.emb_time_dim, device=device)
        zs, logq_paths, _enc_stats = self._encode_latents(
            observed_data=observed_data,
            time_embed=time_embed,
            observation_mask=observation_mask,
            covariates=covariates,
            static_embed=static_embed,
        )

        S = zs.shape[1]
        K_use = S if K is None else K
        if K_use > S:
            raise ValueError(
                f"requested K={K_use} > self.S={S}; increase model.S to draw more trajectories"
            )
        zs = zs[:, :K_use].contiguous()
        logq_paths = logq_paths[:, :K_use].contiguous()

        log_q_z = logq_paths.sum(dim=-1)

        log_p_dec = torch.zeros(B, K_use, device=device, dtype=dtype)
        for t in range(T):
            x_t = observed_data[:, :, t]
            m_t = observation_mask[:, :, t]
            t_idx = torch.full((B,), t, device=device, dtype=torch.long)
            for k in range(K_use):
                z_hist = zs[:, k, :, : t + 1]
                if z_hist.shape[-1] > j:
                    z_hist = z_hist[..., -j:]
                logp_t, _, _, _ = self.decoder.log_likelihood(
                    x_t=x_t,
                    z_hist=z_hist,
                    time_embed=time_embed,
                    time_idx=t_idx,
                    observation_mask_t=m_t,
                    covariates=covariates,
                    static_embed=static_embed,
                )
                log_p_dec[:, k] = log_p_dec[:, k] + logp_t

        transition = self.transition
        log_p_trans = torch.zeros(B, K_use, device=device, dtype=dtype)
        for t in range(j, T):
            if self.sigma_data is not None:
                sigma_d2 = (
                    self.sigma_data.read(t + 1).expand(B).to(device=device, dtype=dtype)
                )
            else:
                sigma_d2 = torch.ones(B, device=device, dtype=dtype)
            ctx = {
                "hist_time_emb": time_embed[:, t - j : t, :],
                "target_time_emb": time_embed[:, t : t + 1, :],
            }
            for k in range(K_use):
                logp_t = transition.log_prob(
                    z=zs[:, k, :, t],
                    z_hist=zs[:, k, :, t - j : t],
                    ctx=ctx,
                    sigma_d2=sigma_d2,
                    rtol=rtol,
                    atol=atol,
                    method=method,
                    divergence_mode=divergence_mode,
                    generator=generator,
                )
                log_p_trans[:, k] = log_p_trans[:, k] + logp_t

        if self.aux_posterior is not None and hasattr(transition, "log_prob_init"):
            log_p_init = transition.log_prob_init(
                zs=zs,
                aux_posterior=self.aux_posterior,
                time_embed=time_embed,
                sigma_data=self.sigma_data,
                covariates=covariates,
                rtol=rtol,
                atol=atol,
                method=method,
                divergence_mode=divergence_mode,
                generator=generator,
            )  # (B, K_use)
        else:
            log_p_init = torch.zeros(B, K_use, device=device, dtype=dtype)

        log_p_xz = log_p_init + log_p_trans + log_p_dec
        return iwae_log_likelihood(log_p_xz, log_q_z, dim=-1)

    def forward(
        self,
        observed_data: torch.Tensor,  # (B, D, T)
        observation_mask: torch.Tensor,  # (B, D, T)
        timepoints: torch.Tensor,  # (B, T)
        covariates: torch.Tensor | None = None,  # (B, V, T) or None
        static_covariates: torch.Tensor | None = None,  # (B, D, V_s) or None
        train: bool = True,
    ):
        """Compute ELBO loss and its components for a batch.

        Args:
            observed_data: Observed time-series, shape ``(B, D, T)``.
            observation_mask: Binary mask (1 = observed, 0 = missing), shape ``(B, D, T)``.
            timepoints: Integer or real timestamps, shape ``(B, T)``.
            covariates: Optional time-varying covariates, shape ``(B, V, T)``.
            static_covariates: Optional static categorical features, shape ``(B, D, V_s)``.
            train: If ``False``, also returns posterior samples/stats in ``stats``.

        Returns:
            components: ``LossComponents`` with unweighted per-term
                tensors (recon, init_kl, trans_kl). The loss object
                applies its own weights and produces the scalar that
                gets backpropped.
            metrics: Dict of scalar tensors for logging.
            stats: Empty dict during training; contains ``zs``, ``mus``,
                ``logvars`` when ``train=False``.
        """
        static_embed = self._embed_static(static_covariates)
        time_embed = time_embedding(
            timepoints, self.emb_time_dim, device=observed_data.device
        )  # (B, T, E_t)
        # encode latents: q_ϕ(z_{1:T}|·)
        zs, logq_paths, enc_stats = self._encode_latents(
            observed_data=observed_data,
            time_embed=time_embed,
            observation_mask=observation_mask,
            covariates=covariates,
            static_embed=static_embed,
        )  # zs: (B,S,d,T), logq_paths: (B,S,T)

        # optional Gaussian stats
        mus = enc_stats.get("mus", None)
        logvars = enc_stats.get("logvars", None)
        if logvars is not None:
            logvars = logvars.clamp(min=self.logvar_min, max=self.logvar_max)
            # Mutate enc_stats so downstream loss/KL terms see the clamped values.
            enc_stats["logvars"] = logvars

        # --- ELBO pieces ---

        L_rec, L_rec_calib = self._reconstruction_loss(
            observed_data=observed_data,
            time_embed=time_embed,
            zs=zs,
            observation_mask=observation_mask,
            covariates=covariates,
            static_embed=static_embed,
        )

        init_terms = self._init_kl_loss(
            zs,
            logq_paths,
            enc_stats,
            time_embed,
            covariates,
        )

        L_init = init_terms["loss"]
        # ψ-side (unit-weighted score-net) init loss. Requested via
        # ``return_psi=True`` in ``_init_kl_loss``; transitions without a
        # score net return an exact zero. Kept graph-connected (no detach)
        # for the split backward.
        L_init_psi = init_terms.get("loss_psi", torch.zeros_like(L_init))
        # ``torch.zeros((), device=...)`` allocates the scalar directly on
        # the target device — critical for CUDA graph capture, where
        # ``torch.tensor(0.0, device=cuda)`` would attempt a CPU→GPU copy
        # of an unpinned scalar and raise.
        L_vhp = init_terms.get(
            "vhp", torch.zeros((), device=observed_data.device)
        )
        L_ent_init = init_terms.get(
            "entropy", torch.zeros((), device=observed_data.device)
        )

        trans_terms = self._compute_transition_kl(
            zs=zs,
            logq_paths=logq_paths,
            enc_stats=enc_stats,
            time_embed=time_embed,
            covariates=covariates,
        )
        L_trans = trans_terms["kl"]
        # ψ-side (unit-weighted score-net) transition loss. Diffusion
        # transitions always return ``kl_psi``; non-diffusion transitions
        # (Gaussian / CSDI) return only ``"kl"`` (+ diagnostics), so the
        # zeros fallback is exact for them. Kept graph-connected (no
        # detach) for the split backward.
        L_trans_psi = trans_terms.get("kl_psi", torch.zeros_like(L_trans))
        trans_subterms = {k: v for k, v in trans_terms.items() if k != "kl"}

        distortion = L_rec
        # `loss` and `rate` in the metrics dict below are UNWEIGHTED
        # post-ADR-0004 — the loss object owns weights now.
        rate = L_init + L_trans
        loss = distortion + rate

        metrics = {
            "loss/total": loss.detach(),
            "loss/distortion/rec": L_rec.detach(),
            "loss/rate/init/tot": L_init.detach(),
            "loss/rate/init/vhp": L_vhp.detach(),
            "loss/rate/init/entropy": L_ent_init.detach(),
            "loss/rate/trans/kl": L_trans.detach(),
            "loss/rate/total": rate.detach(),
            "calib/ratio_res2_to_sigma2": L_rec_calib.detach(),
        }
        # Surface model-v2 init-term sub-components when present.
        if self.aux_posterior is not None:
            if "kl_aux" in init_terms:
                metrics["loss/rate/init/kl_aux"] = init_terms["kl_aux"].detach()
            if "loss_init" in init_terms:
                metrics["loss/rate/init/loss_init"] = init_terms["loss_init"].detach()
            if "loss_psi" in init_terms:
                metrics["loss/rate/init/loss_psi"] = init_terms["loss_psi"].detach()
        # Surface per-t σ_data²[t] buffer values whenever a buffer exists.
        # These feed the post-hoc ``sigma_data_drift`` metric's trajectory
        # plot (init-experiment.org § Headline metrics, metric 6); logged once
        # per step so the trajectory is recoverable from metrics.csv alone.
        if self.sigma_data is not None:
            buf = self.sigma_data.sigma_data2.detach()
            for slot, value in enumerate(buf):
                metrics[f"diag/sigma_data2/t={slot + 1}"] = value
        # Surface any optional transition sub-components (e.g. L_p, L_q) under
        # transition-driven keys.
        for key, val in trans_subterms.items():
            metrics[f"loss/rate/trans/{key}"] = val.detach()

        stats = {}  # return optional params
        if not train:  # Optionally return posterior stats/samples for analysis
            stats["zs"] = zs
            stats["mus"] = mus
            stats["logvars"] = logvars

        components = LossComponents(
            recon=L_rec,
            init_kl_phith=L_init,
            init_kl_psi=L_init_psi,
            trans_kl_phith=L_trans,
            trans_kl_psi=L_trans_psi,
        )
        return components, metrics, stats

    @torch.no_grad()
    def forecast(
        self,
        x_hist: torch.Tensor | None = None,  # (B, K, H)
        x_mask: torch.Tensor | None = None,  # (B, K, H)
        past_time: torch.Tensor | None = None,  # (B, H)
        future_time: torch.Tensor | None = None,  # (B, L2)
        past_covariates: torch.Tensor | None = None,  # (B, V, H) or None
        future_covariates: torch.Tensor | None = None,  # (B, V, L2) or None
        static_covariates: torch.Tensor | None = None,  # (B, V_s) or None
        *,
        # sampling controls
        num_samples: int = 32,
        use_vp_init: bool = False,
        s_churn: float = 0.0,
        s_noise: float = 1.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
    ) -> dict[str, torch.Tensor]:
        """Encode history, autoregressively roll out, and decode the future.

        Returns:
            Dict with ``pred_mean`` ``(B, D, L2)`` and ``pred_samples``
            ``(B, num_samples, D, L2)``.
        """
        assert (
            x_hist is not None
            and x_mask is not None
            and past_time is not None
            and future_time is not None
        ), "Need x_hist, x_mask, past_time, future_time."

        device = x_hist.device
        B, K, H = x_hist.shape
        L2 = int(future_time.size(1))
        d, j = int(self.latent_dim), int(self.j)

        # ----- time embeddings (past + future) -----
        time_all = torch.cat([past_time, future_time], dim=1)  # (B, H+L2)
        time_embed_all = time_embedding(
            time_all, self.emb_time_dim, device=device
        )  # (B, H+L2, E_t)
        time_embed_past = time_embed_all[:, :H, :]  # (B, H, E_t)

        covariates_all = None
        if past_covariates is not None and future_covariates is not None:
            # Assumes shape (B, V, T)
            covariates_all = torch.cat([past_covariates, future_covariates], dim=2)

        static_embed = self._embed_static(static_covariates)

        # Encode past
        # zs: (B, S, d, H)
        zs, _, stats = self.encoder.sample_paths(
            observed_data=x_hist,
            time_embed=time_embed_past,
            S=num_samples,
            cond_mask=x_mask if getattr(self.encoder, "use_mask", False) else None,
            covariates=past_covariates,
            static_embed=static_embed,
        )

        if use_vp_init and "mus" in stats:
            z_src = stats["mus"]  # (B, S, d, H)
        else:
            z_src = zs

        # Extract last j steps
        # z_src is (B, S, d, H), from VP draws.
        # need (B, S, d, j)
        if j <= H:
            z_hist = z_src[..., -j:]
        else:
            # Pad left
            pad_len = j - H
            pad = torch.zeros(
                B, num_samples, d, pad_len, device=device, dtype=z_src.dtype
            )
            z_hist = torch.cat([pad, z_src], dim=-1)

        # (B, S, d, j) -> (B*S, d, j)
        z_hist_flat = z_hist.reshape(B * num_samples, d, j)

        time_embed_all_bs = (
            time_embed_all
            .unsqueeze(1)
            .expand(B, num_samples, -1, -1)
            .reshape(B * num_samples, H + L2, -1)
        )

        if covariates_all is not None:
            covariates_all_bs = (
                covariates_all
                .unsqueeze(1)
                .expand(B, num_samples, -1, -1)
                .reshape(B * num_samples, covariates_all.size(1), H + L2)
            )
        else:
            covariates_all_bs = None

        if static_embed is not None:
            static_embed_bs = (
                static_embed
                .unsqueeze(1)
                .expand(B, num_samples, -1, -1)
                .reshape(B * num_samples, self.data_dim, self.total_static_dim)
            )
        else:
            static_embed_bs = None

        # ----- unroll each sample path and decode -----
        x_future_samples = []
        for t_step in range(L2):
            t_abs = H + t_step

            # Transition: z_t ~ p(z_t | z_{t-j:t-1})
            t_start = t_abs - j
            t_end = t_abs

            if t_start < 0:
                # Pad time embeddings
                pad_len = -t_start
                valid_emb = time_embed_all_bs[:, 0:t_end, :]
                pad_emb = torch.zeros(
                    B * num_samples,
                    pad_len,
                    self.emb_time_dim,
                    device=device,
                    dtype=time_embed_all.dtype,
                )
                hist_time_emb = torch.cat([pad_emb, valid_emb], dim=1)

                if covariates_all_bs is not None:
                    valid_cov = covariates_all_bs[:, :, 0:t_end]
                    pad_cov = torch.zeros(
                        B * num_samples,
                        covariates_all_bs.size(1),
                        pad_len,
                        device=device,
                        dtype=covariates_all_bs.dtype,
                    )
                    hist_covariates = torch.cat([pad_cov, valid_cov], dim=-1)
                else:
                    hist_covariates = None
            else:
                hist_time_emb = time_embed_all_bs[:, t_start:t_end, :]
                hist_covariates = (
                    covariates_all_bs[:, :, t_start:t_end]
                    if covariates_all_bs is not None
                    else None
                )

            # extract time step t
            target_time_emb = time_embed_all_bs[:, t_end : t_end + 1, :]
            ctx = {"hist_time_emb": hist_time_emb, "target_time_emb": target_time_emb}
            if self.sigma_data is not None:
                # Use the frozen per-t σ_data² buffer for the sampler's EDM
                # constants (model-v2.org § Practical considerations). The buffer
                # is 1-based; the target's 0-based position is ``t_abs``, so the
                # 1-based index is ``t_abs + 1``. sample() clamps to [1, T_max]
                # for constant extrapolation beyond the training horizon.
                ctx["sigma_data"] = self.sigma_data
                ctx["t"] = t_abs + 1
            if hist_covariates is not None:
                ctx["hist_covariates"] = hist_covariates.permute(
                    0, 2, 1
                )  # to (BS, j, V)

            if covariates_all_bs is not None:
                target_covariates = covariates_all_bs[:, :, t_end : t_end + 1]
                ctx["target_covariates"] = target_covariates.permute(
                    0, 2, 1
                )  # to (BS, 1, V)

            # transition.sample returns (BS, 1, d) because we pass S=1 (we already flattened S)
            z_t = self.transition.sample(
                z_hist_flat, S=1, ctx=ctx
            )  # (BS, 1, d)
            z_t = z_t.squeeze(1)  # (BS, d)

            # Decode: x_t ~ p(x_t | z_{t-j+1:t})
            # z_hist_flat is (BS, d, j). We append z_t.
            z_next_hist = torch.cat([z_hist_flat[..., 1:], z_t.unsqueeze(-1)], dim=-1)

            # Decoder needs z (BS, d, j) and time_idx.
            # time_idx for decoder is t_abs.
            t_idx_bs = torch.full(
                (B * num_samples,), t_abs, dtype=torch.long, device=device
            )

            # Sample x_t
            mu_x, logvar_x = self.decoder(
                z=z_next_hist,
                time_embed=time_embed_all_bs,
                time_idx=t_idx_bs,
                covariates=covariates_all_bs,
                static_embed=static_embed_bs,
            )
            eps_x = torch.randn_like(mu_x)
            x_t = mu_x + eps_x * torch.exp(0.5 * logvar_x)  # (BS, D)

            x_future_samples.append(x_t)

            # Update z_hist_flat for next step
            z_hist_flat = z_next_hist

        # Stack samples
        # x_future_samples: list of (BS, D) -> (BS, D, L2)
        x_future = torch.stack(x_future_samples, dim=-1)

        # (BS, D, L2) -> (B, S, D, L2)
        x_future = x_future.view(B, num_samples, self.data_dim, L2)

        pred_mean = x_future.mean(dim=1)  # (B, D, L2)

        return {"pred_mean": pred_mean, "pred_samples": x_future}


# ---------------------------------------------------------------------------
# Default hyperparams namespace (used when no hyperparams object is provided)
# ---------------------------------------------------------------------------


def _default_hyperparams():
    """Return a SimpleNamespace with default training hyperparameters.

    This is used when ``DDSSM_base`` is constructed without an explicit
    ``hyperparams`` argument (e.g. in tests or interactive use).
    """
    return SimpleNamespace(
        S=1,
        ema_decay=0.999,
        weight_decay=1e-4,
        weight_decay_psi=None,
        batch_size=16,
        grad_accum_steps=4,
        t_chunk=16,
        clip_grad_norm=1.0,
        psi_betas=None,
        enc_lr=5e-4,
        dec_lr=5e-4,
        trans_lr=5e-4,
        logvar_min=-13.0,
        logvar_max=13.0,
        lambda_ramp=None,
        lr_schedule=None,
        use_split_loss=False,
    )


# ---------------------------------------------------------------------------
# Hyperparameters and top-level DDSSMConf, co-located with DDSSM_base.
# ---------------------------------------------------------------------------


@dataclass
class DDSSMHyperParamsConf:
    """Training hyperparameters for DDSSM."""

    S: int = 1
    ema_decay: float = 0.999
    # AdamW weight decay for the single optimizer / φθ side. The ψ side
    # (score net) can be overridden independently via ``weight_decay_psi``;
    # ``None`` falls back to ``weight_decay``.
    weight_decay: float = 1e-4
    weight_decay_psi: float | None = None
    batch_size: int = 16
    grad_accum_steps: int = 4
    t_chunk: int = 16
    # Global grad-norm clip, applied after the non-finite-grad skip check
    # (see ``DDSSMTrainer._optimizer_step``). ``None`` disables clipping.
    clip_grad_norm: float | None = 1.0
    # Optional Adam betas for the score-net (ψ) param groups in single-loss
    # mode (list, not tuple, for OmegaConf; the trainer converts at use).
    # ``None`` (default) keeps today's optimizer topology exactly.
    psi_betas: list[float] | None = None

    enc_lr: float = 5e-4
    dec_lr: float = 5e-4
    trans_lr: float = 5e-4

    logvar_min: float = -13.0
    logvar_max: float = 13.0

    # Optional λ-ramp for the FullELBO rate weight on the φθ-side KL terms.
    # ``None`` (default) keeps the constant λ ≡ 1.0 behavior.
    lambda_ramp: LambdaRampConf | None = None
    # Optional per-role LR schedule (φθ / ψ). ``None`` keeps constant LR.
    # Enabling requires ``lambda_ramp`` set (the resolver anchors decay
    # windows to λ_end = lambda_ramp.delay + lambda_ramp.steps).
    lr_schedule: LrScheduleGroupConf | None = None
    # Enable split-loss training: the FullELBO returns a SplitLoss (φθ / ψ)
    # so the transition (ψ) and encoder/decoder (φθ) get separate optimizers
    # and separate objectives. Default False keeps the single-loss path.
    use_split_loss: bool = False
