"""Stage-2 diffusion transition for the model-v2 baseline-centering scheme.

``DiffusionTransition`` implements the model-v2 stage-2 transition along
three coupled axes from ``model-v2.org``:

1. **Centered ESM target.**  The score is matched in centered
   coordinates ``ẑ_t = z̃_t − μ_p(z_{t-1})``, where μ_p comes from a
   shared :class:`BaseBaseline` instance (the *same* baseline the
   stage-1 :class:`BaselineGaussianTransition` uses, so μ_p's
   parameters carry through the handoff).  The closed-form encoder
   marginal score in centered coords is
   ``ŝ_q = −(ẑ − μ̂_t) / Σ_t``, where ``μ̂_t = μ_t − μ_p`` and
   ``Σ_t = σ_t² + σ̃²`` (object 2 of the "three variance objects").

2. **σ_data(t)-driven EDM preconditioning.**  The EDM constants
   ``(c_skip, c_out, c_in)`` are recomputed *per call* from the
   current ``σ_data²(t)`` value stored in a :class:`SigmaDataBuffer`
   that lives on :class:`DDSSM_base`.  ``c_noise`` is σ_data-
   independent and stays cached.  When ``σ_data ≡ 1`` the constants
   reduce to the canonical EDM values (a unit test exercises this).

3. **VHP-via-diffusion at t = 1 … j.**  The *same* score network and
   schedule are reused at the initial j steps with auxiliary latents
   ``z_{-j+1:0} ∼ q_Φ(· | z_{1:j})`` in the prev-latent slot and a
   per-slot binary **padding mask** flagging the aux slots in the
   side-info tensor.  Encoder entropy at ``t = 1`` cancels with the
   ESM expansion (per § Entropy cancellation in stage 2), so the
   returned ``loss_init`` is the entropy-cancelled surrogate; ``DDSSM_base``
   does *not* add ``-H(q_φ)`` separately in stage 2.

The transition assumes the encoder is Gaussian with
``(mus, logvars)`` available — there is no MC fallback for the
centered ESM target.
"""

from __future__ import annotations

import math
from typing import Any, final
from functools import partial
from dataclasses import dataclass
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ddssm.nn.diffnets import CSDIUnet
from ddssm.nn.gaussians import GaussianStats
from ddssm.nn.net_utils import get_side_info
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.nn.torch_compile import maybe_compile
from ddssm.model.centering.baselines import BaseBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.transitions import BaseTransition


def _compute_vp_schedule_buffers(
    *,
    num_steps: int,
    beta_min: float,
    beta_max: float,
    tau_min: float,
    tau_max: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Precompute the VP-SDE grid quantities for a given schedule.

    Returns a dict of float32 tensors keyed by the names the rest of the
    module expects: ``alpha``, ``alpha2``, ``sigma_tilde``,
    ``one_minus_alpha2``, ``c_noise``, ``beta``, ``tau``,
    ``w_per_tau_unit``, ``wtilde_base``.

    Factored out of ``DiffusionTransition.__init__`` so the same precompute
    can be reused for the (independent) sampling schedule. The math is
    unchanged — only the wrapping into a helper is new.
    """
    if not (0.0 < tau_min < tau_max):
        raise ValueError(
            f"tau_min must be in (0, tau_max); got tau_min={tau_min}, tau_max={tau_max}"
        )
    if beta_max <= beta_min:
        raise ValueError(f"beta_max ({beta_max}) must be > beta_min ({beta_min})")

    dtype64 = torch.float64
    eps64 = torch.finfo(dtype64).eps
    K = int(num_steps)

    # K left-endpoint grid points {τ_min, …, τ_max − dτ} with spacing dτ.
    tau = torch.linspace(tau_min, tau_max, K + 1, dtype=dtype64)[:-1]
    beta = beta_min + (beta_max - beta_min) * tau
    int_beta = beta_min * tau + 0.5 * (beta_max - beta_min) * tau * tau
    alpha = torch.exp(-0.5 * int_beta)
    alpha2 = alpha * alpha
    one_minus_alpha2 = (1.0 - alpha2).clamp_min(eps64)
    sigma_tilde = torch.sqrt(one_minus_alpha2 / alpha2.clamp_min(eps64))
    c_noise = 0.25 * torch.log(sigma_tilde.clamp_min(eps64))

    # Per-tau g²(τ) for the ESM weight (σ_data-independent factor).
    # The σ_data-dependent piece c_out²(τ;t) is multiplied in per call.
    w_per_tau_unit = beta * alpha2 / one_minus_alpha2  # diagnostic
    dtau = (tau_max - tau_min) / float(K)
    # Bake the (1/2 * dtau) Riemann measure into ``wtilde_base``.
    # The full weight is ``wtilde_base[k] · σ_data²(t) / (σ̃² + σ_data²(t))``;
    # this gets applied in ``_esm_chunk_loss`` once σ_data is known.
    wtilde_base = 0.5 * dtau * beta / one_minus_alpha2

    return {
        "alpha": alpha.to(torch.float32),
        "alpha2": alpha2.to(torch.float32),
        "sigma_tilde": sigma_tilde.to(torch.float32),
        "one_minus_alpha2": one_minus_alpha2.to(torch.float32),
        "c_noise": c_noise.to(torch.float32),
        "beta": beta.to(torch.float32),
        "tau": tau.to(torch.float32),
        "w_per_tau_unit": w_per_tau_unit.to(torch.float32),
        "wtilde_base": wtilde_base.to(torch.float32),
    }


def _adaptive_is_density_meandom(
    sigma_tilde: torch.Tensor,  # (K,)
    sigma_d2: torch.Tensor,  # (n_t,)
    floor: float = 1e-12,
) -> torch.Tensor:
    """Mean-dominated adaptive IS density per (importance-sampling.org § Mean-
    dominated regime, line 342)::

        p*(s; σ_d²) ∝ s / (σ_d² + s²)²,   s_peak = σ_d/√3.

    Per-t-broadcast: returns shape ``(n_t, K)`` with each row normalised to
    sum to 1. Reduces exactly to the legacy ``s/(1+s²)²`` density at
    ``σ_d² = 1``. Sample-independent under the regularised regime where the
    encoder posterior variance stays near σ_d² across samples — the (σ²−σ_d²)²
    term in the full formula vanishes and the (σ²+s²) factor cancels.
    """
    s = sigma_tilde.to(torch.float32)  # (K,)
    sd2 = sigma_d2.to(torch.float32).clamp_min(floor).unsqueeze(-1)  # (n_t, 1)
    raw = (s / (sd2 + s * s).pow(2)).clamp_min(floor)  # (n_t, K)
    return raw / raw.sum(dim=-1, keepdim=True)


def _adaptive_is_density_full(
    sigma_tilde: torch.Tensor,  # (K,)
    sigma_d2: torch.Tensor,  # (N,)
    sigma2: torch.Tensor,  # (N,)
    mu_hat2: torch.Tensor,  # (N,)
    floor: float = 1e-12,
) -> torch.Tensor:
    """Full per-sample adaptive IS density (importance-sampling.org line 319,
    the boxed equation)::

        p*(s; σ_d², σ², μ̂²) ∝
            s · [μ̂²(σ²+s²) + (σ²-σ_d²)²]
              / [(σ_d²+s²)² · (σ²+s²)]

    Returns shape ``(N, K)`` with each row normalised to sum to 1. At
    ``σ²=σ_d²`` the second numerator term vanishes and the ``(σ²+s²)``
    factor cancels, collapsing to the mean-dom form modulo a per-row
    scalar (which is absorbed by the normalisation).
    """
    s = sigma_tilde.to(torch.float32)  # (K,)
    s2 = s * s  # (K,)
    sd2 = sigma_d2.to(torch.float32).clamp_min(floor).unsqueeze(-1)  # (N, 1)
    sg2 = sigma2.to(torch.float32).clamp_min(floor).unsqueeze(-1)  # (N, 1)
    mh2 = mu_hat2.to(torch.float32).unsqueeze(-1)  # (N, 1)
    num = s * (mh2 * (sg2 + s2) + (sg2 - sd2).pow(2))  # (N, K)
    den = (sd2 + s2).pow(2) * (sg2 + s2)  # (N, K)
    raw = (num / den.clamp_min(floor)).clamp_min(floor)
    return raw / raw.sum(dim=-1, keepdim=True)


@dataclass
class DiffusionScheduleConfig:
    """VP-SDE schedule configuration for :class:`DiffusionTransition`.

    The (σ_data-dependent) EDM constants are computed per call rather than
    precomputed because they vary with the current ``σ_data²(t)`` buffer
    value.
    """

    S_k: int = 1
    k_chunk: int = 1
    num_steps: int = 100
    beta_min: float = 0.1
    beta_max: float = 20.0
    tau_min: float = 1e-3
    # Importance-sampling mode for noise-level selection (training-time MC
    # over the diffusion τ axis). The IS reweighting in
    # ``_esm_chunk_loss`` keeps the estimator unbiased regardless of mode.
    #
    # - ``"adaptive_is"``: mean-dominated form of the loss-aware optimal
    #   IS density per importance-sampling.org § Mean-dominated regime
    #   (line 342)::  p_k ∝ s / (σ_d²(t) + s²)²,  s_peak = σ_d(t)/√3.
    #   Per-t adaptive — concentrates MC mass at the τ scale set by the
    #   live ``SigmaDataBuffer``. Default. Sample-independent in the
    #   regularised regime; cheap.
    # - ``"adaptive_is_full"``: full per-sample IS density per
    #   importance-sampling.org line 319 (the boxed equation),
    #   ``s·[μ̂²(σ²+s²) + (σ²-σ_d²)²] / [(σ_d²+s²)²·(σ²+s²)]``. Captures
    #   the bimodal regime when the per-sample posterior variance
    #   diverges from σ_d² (encoder over-confidence). More expensive
    #   per step; pick when you want sharper IS under encoder pathology.
    # - ``"lsgm_is"``: p_k ∝ β/(1−α²), the LSGM importance-sampling
    #   density. Schedule-aware but not σ_d-aware.
    # - ``"uniform"``: flat p_k, preserved as a baseline + for the
    #   variance probe.
    k_sampling_mode: str = "adaptive_is"
    pk_gamma: float = 1.0
    pk_floor: float = 1e-12
    # Timesteps processed per score-net call in ``transition_kl`` (the per-t ESM
    # losses are independent → chunking is pure batching, loss-invariant). With
    # gradient checkpointing the retained memory is one chunk's d²-attention, so
    # larger chunks = fewer/faster calls; tune to the memory budget. ``None`` ⇒ 1
    # (per-timestep: minimal memory, slowest).
    time_chunk_size: int | None = None


@dataclass
class DiffusionSamplingScheduleConfig:
    """Inference-time (rollout) VP-SDE schedule. Independent of
    :class:`DiffusionScheduleConfig` (which is the training-loss schedule).

    The training schedule must be **wide and σ_d=1-centred** because σ_d(t)
    is evolving during training — we don't know where it'll land. The
    sampling schedule is set post-training when σ_d has stabilised and can
    be narrower and σ_d-centred (the user is responsible for picking
    values consistent with the converged σ_d — this config does *not*
    read the live buffer at sample-time).

    Default ``num_steps=50`` is smaller than the training default (100)
    because inference walks the τ grid deterministically — fewer steps
    = faster rollout. ``tau_min``, ``tau_max``, ``beta_*`` are deliberate
    user knobs, not σ_d-relative. See importance-sampling.org §
    "Revised practical implications" for the σ_d-aware floor recommendations
    when calibrating these by hand.
    """

    num_steps: int = 50
    tau_min: float = 1e-3
    tau_max: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 20.0


@final
class DiffusionTransition(BaseTransition):
    """Centered ESM/EDM transition with σ_data(t) tracking and VHP at t = 1 … j.

    Args:
        baseline: Shared :class:`BaseBaseline` instance.  *Same* instance
            as the stage-1 :class:`BaselineGaussianTransition`'s
            baseline (μ_p parameters carry over the handoff).
        latent_dim: Latent dimension ``d``.
        j: Latent history length.
        emb_time_dim: Time embedding dimension (matches the model's).
        covariate_dim: Optional time-varying covariate dim.
        T_max: Maximum 1-based timestep the σ_data buffer covers.
            Should match ``sigma_data.T_max``; used here to validate
            t-index lookups.
        unet: Builder for the score network ``F_ψ``.  Built with
            ``zero_init_output=True`` so ``D_ψ ≈ c_skip·z̃`` at
            stage-2 start.
        schedule: VP-SDE schedule.
    """

    def __init__(
        self,
        baseline: BaseBaseline,
        latent_dim: int,
        j: int,
        emb_time_dim: int,
        T_max: int,
        covariate_dim: int = 0,
        unet: Callable[..., CSDIUnet] | None = None,
        schedule: DiffusionScheduleConfig | None = None,
        grad_checkpoint: bool = False,
        sampling_schedule: DiffusionSamplingScheduleConfig | None = None,
        emb_feature_dim: int | None = None,
        sampler: str = "edm",
        edm_s_churn: float = 0.0,
        edm_s_noise: float = 1.0,
        edm_s_tmin: float = 0.0,
        edm_s_tmax: float = float("inf"),
        edm_rho: float = 7.0,
        edm_sigma_max_rel: float | None = None,
        edm_sigma_min_rel: float | None = None,
    ) -> None:
        super().__init__()
        # Gradient-checkpoint the score-net call in the per-chunk ESM loss. The
        # score-net's d²-attention activations are otherwise retained across all
        # T−j timestep chunks (kl_sum accumulates), which is the latent=512
        # memory long-pole; checkpointing recomputes them in backward instead.
        self.grad_checkpoint = bool(grad_checkpoint)
        if int(baseline.latent_dim) != int(latent_dim):
            raise ValueError(
                f"baseline.latent_dim={baseline.latent_dim} != latent_dim={latent_dim}"
            )
        if int(baseline.j) != int(j):
            raise ValueError(f"baseline.j={baseline.j} != j={j}")
        if T_max <= j:
            raise ValueError(f"T_max ({T_max}) must be > j ({j})")

        self.baseline = baseline
        self.latent_dim = int(latent_dim)
        self.j = int(j)
        self.emb_time_dim = int(emb_time_dim)
        self.covariate_dim = int(covariate_dim)
        self.T_max = int(T_max)

        if schedule is None:
            schedule = DiffusionScheduleConfig()
        self.schedule = schedule
        self.S_k = schedule.S_k
        self.num_steps = schedule.num_steps

        # Feature + side-info dim — +1 for cond_mask, +1 for padding_mask
        # (precursor (iii) from init-experiment.org § Implementation precursors).
        # emb_feature_dim historically tracked emb_time_dim; it is now decoupled so
        # the per-channel feature embedding can be enabled (CSDI-style) while time
        # conditioning stays off (emb_time_dim=0).
        self.emb_feature_dim = (
            int(emb_time_dim) if emb_feature_dim is None else int(emb_feature_dim)
        )
        self.side_dim = (
            self.emb_time_dim + self.covariate_dim + self.emb_feature_dim + 2
        )

        # Sampler selection + EDM (Karras 2022) knobs. ``edm`` (default) is the
        # σ-space Heun sampler with optional stochastic churn (deterministic at
        # the default edm_s_churn=0); ``pf_ode`` is the legacy deterministic VP
        # probability-flow Euler sampler.
        self.sampler = str(sampler)
        self.edm_s_churn = float(edm_s_churn)
        self.edm_s_noise = float(edm_s_noise)
        self.edm_s_tmin = float(edm_s_tmin)
        self.edm_s_tmax = float(edm_s_tmax)
        self.edm_rho = float(edm_rho)
        self.edm_sigma_max_rel = (
            float(edm_sigma_max_rel) if edm_sigma_max_rel is not None else None
        )
        self.edm_sigma_min_rel = (
            float(edm_sigma_min_rel) if edm_sigma_min_rel is not None else None
        )

        if unet is None:
            unet = partial(
                CSDIUnet,
                channels=64,
                n_layers=4,
                embedding_dim=128,
            )
        # Side-info channel layout is [time(+cov), feat, cond_mask, padding_mask],
        # so the conditioning mask is the second-to-last channel. Pass the index
        # explicitly so the U-Net never has to guess it from a relative offset.
        cond_mask_channel = (
            self.emb_time_dim + self.covariate_dim + self.emb_feature_dim
        )
        self.diffmodel = unet(
            output_len=1,
            diffusion_steps=schedule.num_steps,
            latent_dim=self.latent_dim,
            latent_history_len=self.j,
            side_dim=self.side_dim,
            zero_init_output=True,
            cond_mask_channel=cond_mask_channel,
        )
        self.diffmodel = maybe_compile(self.diffmodel)

        self.embed_layer = nn.Embedding(
            num_embeddings=self.latent_dim, embedding_dim=self.emb_feature_dim
        )

        # ---------- VP-SDE precompute (σ_data-independent quantities) ----------
        # Training schedule: every loss-side buffer (alpha, beta, …) plus
        # the IS p_k are derived from this. ``sample_*`` mirrors register
        # the sampling-schedule grid the rollout reads in
        # ``_vp_pf_sample_centered``.
        precomp_train = _compute_vp_schedule_buffers(
            num_steps=int(schedule.num_steps),
            beta_min=float(schedule.beta_min),
            beta_max=float(schedule.beta_max),
            tau_min=float(schedule.tau_min),
            tau_max=1.0,
        )
        for name, t in precomp_train.items():
            self.register_buffer(name, t)

        # Sampling-schedule buffers. If ``sampling_schedule`` is provided,
        # build a second VP-SDE grid; otherwise alias to the training
        # buffers so ``sample()`` can read ``self.sample_*`` unconditionally.
        if sampling_schedule is not None:
            precomp_sample = _compute_vp_schedule_buffers(
                num_steps=int(sampling_schedule.num_steps),
                beta_min=float(sampling_schedule.beta_min),
                beta_max=float(sampling_schedule.beta_max),
                tau_min=float(sampling_schedule.tau_min),
                tau_max=float(sampling_schedule.tau_max),
            )
            for name, t in precomp_sample.items():
                self.register_buffer(f"sample_{name}", t)
            self.sample_num_steps = int(sampling_schedule.num_steps)
        else:
            # Alias training buffers as plain Python attributes — assigning
            # an already-registered buffer back as a buffer would clone and
            # waste memory; an attribute alias is free and PyTorch handles
            # the duplicate name cleanly on save/load.
            for name in precomp_train.keys():
                setattr(self, f"sample_{name}", getattr(self, name))
            self.sample_num_steps = int(self.num_steps)
        self.sampling_schedule = sampling_schedule

        # Importance-sampling distribution p_k. Static modes
        # (``uniform``, ``lsgm_is``) register a ``(K,)`` buffer that
        # ``_esm_chunk_loss`` broadcasts to per-row. Adaptive modes
        # (``adaptive_is``, ``adaptive_is_full``) compute p_k per-row at
        # loss-time from the live ``SigmaDataBuffer``; there is no static
        # buffer to register, so ``self.p_k`` is left ``None``.
        ismode = schedule.k_sampling_mode
        self.gamma = float(schedule.pk_gamma)
        self.gfloor = float(schedule.pk_floor)
        if ismode == "lsgm_is":
            beta_t = self.beta.to(torch.float32)
            om_a2 = self.one_minus_alpha2.to(torch.float32)
            p_k = (beta_t / om_a2).clamp_min(self.gfloor)
            if self.gamma != 1.0:
                p_k = p_k.pow(self.gamma)
            p_k = p_k / p_k.sum()
            self.register_buffer("p_k", p_k)
        elif ismode == "uniform":
            p_k = torch.full(
                (self.num_steps,), 1.0 / self.num_steps, dtype=torch.float32
            )
            self.register_buffer("p_k", p_k)
        elif ismode in ("adaptive_is", "adaptive_is_full"):
            # Per-row p_k is computed inside ``_esm_chunk_loss`` from the
            # live σ_d²(t) (and per-sample stats for ``adaptive_is_full``).
            self.p_k = None
        else:
            raise ValueError(
                f"Unknown k_sampling_mode={ismode!r}; "
                f"expected 'uniform', 'lsgm_is', 'adaptive_is', or "
                f"'adaptive_is_full'"
            )
        self.k_sampling_mode = ismode

    # ------------------------------------------------------------------
    # Per-transition log-density via probability-flow ODE.
    # ------------------------------------------------------------------
    def log_prob(
        self,
        z: torch.Tensor,
        z_hist: torch.Tensor,
        ctx: dict[str, Any] | None = None,
        mc_override: dict[str, Any] | None = None,
        *,
        sigma_d2: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "dopri5",
        divergence_mode: str = "exact",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Native-coord per-transition log-density ``log p_ψ^ode(z | z_hist)``.

        Layer 1 of the exact-likelihood evaluator (model-v2.org §
        "Exact likelihood evaluation"): integrates the augmented
        probability-flow ODE in native coordinates and returns the
        Liouville log-density, using :func:`solve_prob_flow_logdensity`
        with ``self.score`` as the score callable.

        Args:
            z: ``(B, d)`` evaluation point in native coords.
            z_hist: ``(B, d, j)`` conditioning latents.
            ctx: side-info dict; must include ``hist_time_emb`` and
                ``target_time_emb``.
            mc_override: ignored (kept for ``BaseTransition`` parity).
            sigma_d2: ``(B,)`` ``σ_data²(t)`` per row.  Defaults to ones
                (matches the diffusion sampler's ``sigma_data ≡ 1`` fallback).
            padding_mask: ``(B, j+1)`` padding-mask channel; defaults
                to zeros (no padding).
            rtol, atol, method: torchdiffeq adaptive-solver controls.
            divergence_mode: ``"exact"`` (cycle 2) or ``"hutchinson"``
                (cycle 3).

        Returns:
            ``(B,)`` per-row log-density.
        """
        del mc_override
        from ddssm.model.likelihood import solve_prob_flow_logdensity

        if ctx is None:
            raise ValueError("DiffusionTransition.log_prob requires ctx")
        B, d = z.shape
        if sigma_d2 is None:
            sigma_d2 = torch.ones(B, device=z.device, dtype=z.dtype)

        def score_fn(z_curr: torch.Tensor, tau_curr: torch.Tensor) -> torch.Tensor:
            tau_b = (
                tau_curr.expand(z_curr.shape[0]) if tau_curr.dim() == 0 else tau_curr
            )
            return self.score(
                z=z_curr,
                tau=tau_b,
                z_hist=z_hist,
                ctx=ctx,
                sigma_d2=sigma_d2,
                padding_mask=padding_mask,
            )

        return solve_prob_flow_logdensity(
            score_fn=score_fn,
            z0=z,
            beta_min=float(self.schedule.beta_min),
            beta_max=float(self.schedule.beta_max),
            tau_min=float(self.schedule.tau_min),
            rtol=rtol,
            atol=atol,
            method=method,
            divergence_mode=divergence_mode,
            generator=generator,
        )

    # ------------------------------------------------------------------
    # transition_kl  (t = j+1 … T)
    # ------------------------------------------------------------------
    def transition_kl(
        self,
        enc_stats: GaussianStats,
        zs: torch.Tensor,  # (B, S, d, T)
        logq_paths: torch.Tensor,  # (B, S, T)  — unused
        time_embed: torch.Tensor,  # (B, T, E_t)
        sigma_data: SigmaDataBuffer,
        covariates: torch.Tensor | None = None,
        mc_override: dict[str, Any] | None = None,
        return_per_sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Centered ESM/EDM loss over ``t = j+1 … T``.

        Returns ``{"kl": ..., "kl_phith": ..., "kl_psi": ...}``.  ``"kl"``
        is the IS-weighted ELBO-side loss (unchanged single-loss value);
        ``"kl_phith"`` is the same tensor (alias) and ``"kl_psi"`` is the
        unit-weighted score-net-side loss (same ``/(B·S)`` denominator,
        NOT detached — it shares the score-net forward graph).  ``R_μp``
        is added at the :class:`DDSSM_base.forward` level via the free
        function — diffusion is a pure loss-computer per the plan's
        ownership decisions.

        With ``return_per_sample`` (the variance-probe path) also returns
        ``L_p`` (the unnormalised summed ESM loss) and ``L_p_per_sample``
        (the per-sequence-sample loss, summed over target timesteps) —
        both wrapping the phith side — plus ``kl_psi_per_sample``. The
        default per-timestep chunking gives ``N = B·S`` rows per chunk, so
        the per-sample vectors align across timesteps and sum elementwise.
        """
        del logq_paths
        if (
            "mus" not in enc_stats
            or "logvars" not in enc_stats
            or enc_stats["mus"] is None
            or enc_stats["logvars"] is None
        ):
            raise ValueError(
                "DiffusionTransition.transition_kl requires Gaussian (mus, logvars)."
            )

        B, S, d, T = zs.shape
        j = self.j
        if d != self.latent_dim:
            raise ValueError(f"zs latent dim {d} != self.latent_dim {self.latent_dim}")

        device = zs.device
        dtype = zs.dtype
        kl_sum = torch.zeros((), device=device, dtype=dtype)
        kl_sum_psi = torch.zeros((), device=device, dtype=dtype)
        per_sample_acc: torch.Tensor | None = None
        per_sample_psi_acc: torch.Tensor | None = None
        n_target_steps = max(0, T - j)
        if n_target_steps == 0:
            if return_per_sample:
                zeros = torch.zeros(B * S, device=device, dtype=dtype)
                return {
                    "kl": kl_sum,
                    "kl_phith": kl_sum,
                    "kl_psi": kl_sum_psi,
                    "L_p": kl_sum,
                    "L_p_per_sample": zeros,
                    "kl_psi_per_sample": torch.zeros_like(zeros),
                }
            return {"kl": kl_sum, "kl_phith": kl_sum, "kl_psi": kl_sum_psi}

        mus = enc_stats["mus"]
        logvars = enc_stats["logvars"]

        for (
            B_,
            S_,
            chunk_len,
            t_start,
            t_end,
            _z_target_flat,
            z_hist_flat,  # (N, d, j)
            ctx,
        ) in self._iter_window_chunks(
            zs,
            time_embed,
            time_chunk_size=self.schedule.time_chunk_size,
            covariates=covariates,
        ):
            N = B_ * S_ * chunk_len
            # Slice encoder stats for the chunk's targets.
            mu_t_flat = mus[..., t_start:t_end].permute(0, 1, 3, 2).reshape(N, d)
            sigma2_t_flat = (
                logvars[..., t_start:t_end].exp().permute(0, 1, 3, 2).reshape(N, d)
            )

            # Per-row σ_data²(t).  Row r → c = r % chunk_len → t = t_start + c (0-based).
            t_idx = torch.arange(
                t_start + 1, t_end + 1, device=device, dtype=torch.long
            )  # 1-based, (chunk_len,)
            sigma_d2_per_t = sigma_data.read(t_idx).to(dtype=dtype)  # (chunk_len,)
            sigma_d2_per_row = (
                sigma_d2_per_t
                .view(1, 1, chunk_len)
                .expand(B_, S_, chunk_len)
                .reshape(N)
            )

            # Padding mask: all-zeros for t ≥ j+1 (no aux slots).
            padding_mask = torch.zeros(N, j + 1, device=device, dtype=dtype)

            chunk_loss, chunk_loss_psi, mu_hat = self._esm_chunk_loss(
                mu_t=mu_t_flat,
                sigma2_t=sigma2_t_flat,
                z_hist=z_hist_flat,
                ctx=ctx,
                sigma_d2_per_row=sigma_d2_per_row,
                padding_mask=padding_mask,
                mc_override=mc_override,
                return_per_sample=return_per_sample,
            )
            if return_per_sample:
                # chunk_loss is the per-sample (N=B*S,) vector for this t.
                per_sample_acc = (
                    chunk_loss
                    if per_sample_acc is None
                    else per_sample_acc + chunk_loss
                )
                per_sample_psi_acc = (
                    chunk_loss_psi
                    if per_sample_psi_acc is None
                    else per_sample_psi_acc + chunk_loss_psi
                )
                kl_sum = kl_sum + chunk_loss.sum()
                kl_sum_psi = kl_sum_psi + chunk_loss_psi.sum()
            else:
                kl_sum = kl_sum + chunk_loss
                kl_sum_psi = kl_sum_psi + chunk_loss_psi

            # σ_data update at this chunk's timesteps, reusing the centered
            # mean already computed inside ``_esm_chunk_loss``.
            _update_sigma_data_blocked(
                sigma_data=sigma_data,
                mu_hat=mu_hat,
                sigma2_t=sigma2_t_flat,
                B=B_,
                S=S_,
                chunk_len=chunk_len,
                d=d,
                t_start_external=t_start + 1,
            )

        # Match the BaselineGaussian convention: sum over target timesteps,
        # mean over (B, S).  Dividing additionally by n_target_steps would
        # silently shrink stage-2's KL by factor (T-j) relative to stage-1,
        # biasing the ELBO toward reconstruction across the centering handoff.
        denom = float(B * S)
        kl = kl_sum / denom
        kl_psi = kl_sum_psi / denom  # same /(B·S) denominator on both sides
        if return_per_sample:
            return {
                "kl": kl,
                "kl_phith": kl,
                "kl_psi": kl_psi,
                "L_p": kl_sum,  # unnormalised summed ESM loss (phith side)
                "L_p_per_sample": per_sample_acc,
                "kl_psi_per_sample": per_sample_psi_acc,
            }
        return {"kl": kl, "kl_phith": kl, "kl_psi": kl_psi}

    # ------------------------------------------------------------------
    # transition_kl_init  (t = 1 … j)
    # ------------------------------------------------------------------
    def _init_entropy_term(self, enc_stats: GaussianStats) -> torch.Tensor:
        """Stage-2 ESM expansion already cancels the encoder entropy → 0.

        Overrides the base default (`-H(q_φ)`); per ``model-v2.org``
        § Entropy cancellation in stage 2 the ESM ``loss_init`` is the
        entropy-cancelled surrogate.
        """
        lv = enc_stats["logvars"]
        return torch.zeros((), device=lv.device, dtype=lv.dtype)

    def _score_init_step(
        self,
        *,
        step: int,
        z_t: torch.Tensor,  # (BS, d)  unused (history already shifted by base)
        z_hist: torch.Tensor,  # (BS, d, j)
        enc_stats: GaussianStats,
        time_embed: torch.Tensor,  # (B, T, E_t)
        sigma_data: SigmaDataBuffer,
        B: int,
        S: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Centered ESM/EDM surrogate for one init step (summed over B·S).

        Returns the ``(phith, psi)`` pair from :meth:`_esm_chunk_loss`:
        the IS-weighted ELBO-side loss and the unit-weighted score-net-side
        loss (graph-connected, not detached).  The padding-mask channel
        flags the ``j - step`` aux slots. Also updates ``sigma_data`` at
        this init ``t``.
        """
        del z_t  # the base walk owns the history shift
        BS = B * S
        d = self.latent_dim
        dtype = z_hist.dtype
        device = z_hist.device

        mu_t_flat = enc_stats["mus"][:, :, :, step].reshape(BS, d)
        sigma2_t_flat = enc_stats["logvars"][:, :, :, step].exp().reshape(BS, d)

        t_external = step + 1
        sigma_d2_per_row = sigma_data.read(t_external).to(dtype=dtype).expand(BS)

        # Padding mask: first ``j - step`` slots are aux (1.0); target slot 0.
        j = self.j
        padding_mask = torch.zeros(BS, j + 1, device=device, dtype=dtype)
        n_aux_slots = j - step
        if n_aux_slots > 0:
            padding_mask[:, :n_aux_slots] = 1.0

        ctx_step = self._init_step_time_ctx(step, time_embed, B, S, T)

        chunk_loss, chunk_loss_psi, mu_hat = self._esm_chunk_loss(
            mu_t=mu_t_flat,
            sigma2_t=sigma2_t_flat,
            z_hist=z_hist,
            ctx=ctx_step,
            sigma_d2_per_row=sigma_d2_per_row,
            padding_mask=padding_mask,
        )

        sigma_data.update(
            t_idx=torch.tensor([t_external], device=device),
            mu_hat_batch=mu_hat,
            sigma_t2_batch=sigma2_t_flat,
        )
        return chunk_loss, chunk_loss_psi

    # ------------------------------------------------------------------
    # VHP initial-state log-density (model-v2.org § Exact likelihood, Layer 4).
    # ------------------------------------------------------------------
    def log_prob_init(
        self,
        zs: torch.Tensor,
        aux_posterior: AuxPosterior,
        time_embed: torch.Tensor,
        sigma_data: SigmaDataBuffer | None = None,
        covariates: torch.Tensor | None = None,
        *,
        J: int = 1,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "dopri5",
        divergence_mode: str = "exact",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """VHP importance-sampled ``log p_ψ(z_{1:j})`` per trajectory.

        Layer 4 of the exact-likelihood evaluator (model-v2.org § "VHP
        initial state").  Mirrors :meth:`transition_kl_init`'s
        mixed-history walk over the first ``j`` steps, but instead of the
        ESM/EDM surrogate it accumulates the probability-flow ODE
        log-densities ``log p_ψ^ode(z_step | z_hist_step)`` (via
        :meth:`log_prob`) with ``z_0 ∼ q_Φ`` in the aux slots, then
        reduces the importance weights::

            log p_ψ(z_{1:j}) ≈ logmeanexp_J[
                Σ_step log p_ψ^ode(z_step | z_hist)
                + log N(z_0; 0, I) − log q_Φ(z_0 | z_{1:j})
            ]

        via :func:`ddssm.model.likelihood.vhp.vhp_log_prob_init`.

        Unlike :meth:`transition_kl_init` this is an evaluation-only
        path: ``sigma_data`` is **read, never updated**, and no encoder
        moments are needed (the target is the realised ``z_step`` from
        ``zs``, not an ESM regression).

        Args:
            zs: ``(B, S, d, T)`` encoder latent samples; ``S`` is the
                IWAE trajectory axis.
            aux_posterior: ``q_Φ(z_0 | z_{1:j})``.
            time_embed: ``(B, T, E_t)``.
            sigma_data: ``σ_data²(t)`` buffer (read-only); ``None`` →
                ``σ_data ≡ 1``.
            covariates: ignored (baseline-centering doesn't condition on
                them); kept for signature parity.
            J: number of VHP importance draws.
            rtol, atol, method, divergence_mode, generator: prob-flow ODE
                controls forwarded to :meth:`log_prob`.

        Returns:
            ``(B, S)`` per-trajectory ``log p_ψ(z_{1:j})``.
        """
        del covariates
        from ddssm.model.likelihood import vhp_log_prob_init

        B, S, d, T = zs.shape
        j = self.j
        device = zs.device
        dtype = zs.dtype
        if d != self.latent_dim:
            raise ValueError(f"zs latent dim {d} != self.latent_dim {self.latent_dim}")
        if j > T:
            raise ValueError(f"zs has T={T} < j={j}")
        BS = B * S
        log_2pi = math.log(2.0 * math.pi)

        z_init = zs[..., :j].mean(dim=1)  # (B, d, j)

        log_p_draws: list[torch.Tensor] = []
        log_q_draws: list[torch.Tensor] = []
        log_prior_draws: list[torch.Tensor] = []
        for _ in range(J):
            z_aux, aux_mu, aux_logvar = aux_posterior.sample(z_init)  # (B, d, j) each

            aux_var = aux_logvar.exp()
            log_q = -0.5 * (
                (z_aux - aux_mu).pow(2) / aux_var + aux_logvar + log_2pi
            ).sum(dim=(1, 2))  # (B,)
            log_prior = -0.5 * (z_aux.pow(2) + log_2pi).sum(dim=(1, 2))  # (B,)

            z_hist = z_aux.unsqueeze(1).expand(B, S, d, j).reshape(BS, d, j).clone()
            log_p = torch.zeros(BS, device=device, dtype=dtype)
            for step in range(j):
                if sigma_data is not None:
                    sigma_d2 = sigma_data.read(step + 1).to(dtype=dtype).expand(BS)
                else:
                    sigma_d2 = torch.ones(BS, device=device, dtype=dtype)

                padding_mask = torch.zeros(BS, j + 1, device=device, dtype=dtype)
                n_aux_slots = j - step
                if n_aux_slots > 0:
                    padding_mask[:, :n_aux_slots] = 1.0

                tgt_idx = step
                hist_idx = torch.arange(
                    tgt_idx - j, tgt_idx, device=device, dtype=torch.long
                ).clamp(min=0, max=T - 1)
                hist_te = time_embed.index_select(1, hist_idx)  # (B, j, E_t)
                tgt_te = time_embed[:, tgt_idx : tgt_idx + 1, :]  # (B, 1, E_t)
                time_win = (
                    torch
                    .cat([hist_te, tgt_te], dim=1)
                    .unsqueeze(1)
                    .expand(B, S, j + 1, self.emb_time_dim)
                    .reshape(BS, j + 1, self.emb_time_dim)
                )
                ctx_step: dict[str, torch.Tensor] = {
                    "hist_time_emb": time_win[:, :j, :],
                    "target_time_emb": time_win[:, j : j + 1, :],
                }

                z_t = zs[:, :, :, step].reshape(BS, d)
                log_p = log_p + self.log_prob(
                    z=z_t,
                    z_hist=z_hist,
                    ctx=ctx_step,
                    sigma_d2=sigma_d2,
                    padding_mask=padding_mask,
                    rtol=rtol,
                    atol=atol,
                    method=method,
                    divergence_mode=divergence_mode,
                    generator=generator,
                )

                if j > 1:
                    z_hist = torch.cat([z_hist[:, :, 1:], z_t.unsqueeze(-1)], dim=-1)
                else:
                    z_hist = z_t.unsqueeze(-1)

            log_p_draws.append(log_p.reshape(B, S))
            log_q_draws.append(log_q.unsqueeze(1).expand(B, S))
            log_prior_draws.append(log_prior.unsqueeze(1).expand(B, S))

        log_p_stack = torch.stack(log_p_draws, dim=-1)  # (B, S, J)
        log_q_stack = torch.stack(log_q_draws, dim=-1)  # (B, S, J)
        log_prior_stack = torch.stack(log_prior_draws, dim=-1)  # (B, S, J)
        return vhp_log_prob_init(
            log_p_stack, log_q_stack, log_prior_stack, dim=-1
        )  # (B, S)

    # ------------------------------------------------------------------
    # Per-chunk ESM loss in centered coordinates.
    # ------------------------------------------------------------------
    def _esm_chunk_loss(
        self,
        mu_t: torch.Tensor,  # (N, d)
        sigma2_t: torch.Tensor,  # (N, d)
        z_hist: torch.Tensor,  # (N, d, j)
        ctx: dict[str, torch.Tensor],
        sigma_d2_per_row: torch.Tensor,  # (N,)
        padding_mask: torch.Tensor,  # (N, j+1)
        mc_override: dict[str, Any] | None = None,
        return_per_sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Centered ESM regression: ``E_τ E_q[w · ‖F_ψ − F*‖²]``.

        Returns ``(loss_phith, loss_psi, mu_hat_t)``.  Both losses share
        one score-net forward and fork at the unweighted squared error:

        * ``loss_phith`` — the IS-weighted (``w̃/p_k``) squared error,
          *summed over N* (caller normalises), or the per-sample ``(N,)``
          vector when ``return_per_sample`` (used by the variance probe).
          Bit-identical to the pre-split single loss.
        * ``loss_psi`` — the same squared error *unit-weighted* (no
          ``w̃/p_k``), with the identical reduction and ``S_k``
          normalisation.  NOT detached: it descends from the same
          ``F_pred`` graph so a selective backward can route it to the
          score net ψ.

        ``mu_hat_t`` is the centered mean ``μ̂ = μ_t − μ_p(z_hist)`` so
        callers can feed the σ_data update without re-running the
        baseline.  The sampler draws ``ẑ_t^(τ) = μ̂ + √(σ_t² + σ̃²)·ε``
        in centered coords.
        """
        N, d = mu_t.shape
        device = mu_t.device
        dtype = mu_t.dtype
        eps_dtype = torch.finfo(dtype).eps

        # μ_p(z_hist) — gradient flows through the baseline + (during
        # learnable mode) through the encoder via the chunked z_hist.
        mu_p_t = self.baseline.mean(z_hist)  # (N, d)
        mu_hat_t = mu_t - mu_p_t

        # Side-info window — j+1 slots (j history + 1 target).
        if "hist_time_emb" not in ctx or "target_time_emb" not in ctx:
            raise ValueError(
                "DiffusionTransition requires hist_time_emb and target_time_emb in ctx"
            )
        hist_time = ctx["hist_time_emb"]
        tgt_time = ctx["target_time_emb"]
        if "hist_covariates" in ctx:
            hist_time = torch.cat([hist_time, ctx["hist_covariates"]], dim=-1)
        if "target_covariates" in ctx:
            tgt_time = torch.cat([tgt_time, ctx["target_covariates"]], dim=-1)
        time_win = torch.cat([hist_time, tgt_time], dim=1)  # (N, j+1, E+V)

        cond_mask = torch.ones(N, d, self.j + 1, device=device, dtype=dtype)
        cond_mask[..., -1] = 0.0
        side_win = get_side_info(
            data_dim=self.latent_dim,
            time_embed=time_win,
            embed_layer=self.embed_layer,
            cond_mask=cond_mask,
            device=device,
            padding_mask=padding_mask,
        )  # (N, side_dim, d, j+1)

        # ---- chunk over S_k ----
        k_chunk = max(1, min(int(self.schedule.k_chunk), int(self.S_k)))
        total_sqerr_phith = torch.zeros(N, device=device, dtype=dtype)
        total_sqerr_psi = torch.zeros(N, device=device, dtype=dtype)

        override_k_idx = None
        override_eps = None
        override_p_k = None
        if mc_override is not None:
            override_k_idx = mc_override.get("k_idx")
            override_eps = mc_override.get("eps")
            override_p_k = mc_override.get("p_k")

        # Build per-row p_k once per chunk-loss call. Static modes
        # (uniform / lsgm_is) get a no-copy expand view of self.p_k; the
        # two adaptive modes recompute from the live σ_d² (and per-sample
        # stats for ``adaptive_is_full``) per row. Reweighting at the
        # ``weights = wtilde / p_k`` step below does a row-aware gather.
        if override_p_k is not None:
            # The IS correction must divide by the density the caller's
            # ``k_idx`` was actually drawn from (or, for forced-k sweeps,
            # the mode's sampling density). Recomputing the live adaptive
            # density here would mismatch the caller's proposal and bias
            # the estimator by q(k)/p(k) per row.
            override_p_k = override_p_k.to(device=device, dtype=torch.float32)
            p_k_per_row = (
                override_p_k
                if override_p_k.dim() == 2
                else override_p_k.unsqueeze(0).expand(N, -1)
            )  # (N, K)
        elif self.k_sampling_mode == "adaptive_is":
            p_k_per_row = _adaptive_is_density_meandom(
                self.sigma_tilde,
                sigma_d2_per_row,
                floor=self.gfloor,
            )  # (N, K)
        elif self.k_sampling_mode == "adaptive_is_full":
            # Per-coordinate means so these match σ_d²'s per-coordinate scale
            # (sigma_data tracks residual variance per coordinate). Using sum
            # would over-scale by ~d and break the collapse to mean-dom at
            # real calibration for d > 1.
            mu_hat2_row = mu_hat_t.pow(2).mean(dim=1)  # (N,)
            sigma2_row = sigma2_t.mean(dim=1)  # (N,)
            p_k_per_row = _adaptive_is_density_full(
                self.sigma_tilde,
                sigma_d2=sigma_d2_per_row,
                sigma2=sigma2_row,
                mu_hat2=mu_hat2_row,
                floor=self.gfloor,
            )  # (N, K)
        else:
            # Static mode: broadcast view (zero-copy).
            p_k_per_row = self.p_k.unsqueeze(0).expand(N, -1)  # (N, K)

        remaining_k = int(self.S_k)
        k_cursor = 0
        while remaining_k > 0:
            kc = min(k_chunk, remaining_k)
            remaining_k -= kc

            if override_k_idx is not None:
                k_idx = override_k_idx[:, k_cursor : k_cursor + kc]
            else:
                # Per-row sampling: each of N rows draws kc τ-bins from its
                # own p_k. ``torch.multinomial`` natively handles the 2D
                # input, returning shape (N, kc).
                k_idx = torch.multinomial(p_k_per_row, kc, replacement=True)
            if override_eps is not None:
                eps_n = override_eps[:, :, k_cursor : k_cursor + kc]
            else:
                eps_n = torch.randn(N, d, kc, device=device, dtype=dtype)
            k_cursor += kc

            z_in, F_target = self._vp_precondition(
                mu_hat_t=mu_hat_t,
                sigma2_t=sigma2_t,
                k_idx=k_idx,
                eps=eps_n,
                sigma_d2_per_row=sigma_d2_per_row,
            )  # (N, d, kc) each

            # Latent window: [z_hist, z_in].
            z_hist_rep = z_hist.unsqueeze(-1).expand(N, d, self.j, kc)
            z_in_exp = z_in.unsqueeze(2)  # (N, d, 1, kc)
            latent_w = torch.cat([z_hist_rep, z_in_exp], dim=2)  # (N, d, j+1, kc)
            latent_w = (
                latent_w.permute(0, 3, 1, 2).reshape(N * kc, d, self.j + 1).contiguous()
            )

            side_w = (
                side_win
                .unsqueeze(1)
                .expand(N, kc, -1, -1, -1)
                .reshape(N * kc, self.side_dim, d, self.j + 1)
                .contiguous()
            )

            k_flat = k_idx.reshape(N * kc)
            c_noise_flat = self.c_noise[k_flat]  # (N*kc,)

            # Per-(N*kc) σ_data² values for the weight & EDM constants.
            sd2_flat = sigma_d2_per_row.view(N, 1).expand(N, kc).reshape(N * kc)
            wtilde_base_flat = self.wtilde_base[k_flat]
            st2_flat = self.sigma_tilde[k_flat].pow(2)  # σ̃²
            # σ_data-dependent weight:
            #   w(τ;t) = g²(τ) · σ_data²(t) / (α_τ² · σ̃² · (σ̃² + σ_data²(t)))
            # We stored wtilde_base = (1/2 · dτ) · β / (1 − α²) = (1/2 · dτ) ·
            # g²(τ) / (1 − α²); multiplying by σ_data² / (σ̃² + σ_data²)
            # then dividing by σ̃² · α² yields the full w(τ;t).  But we also
            # have wtilde_base[k] = (1/2 · dτ) · β / (1 − α²); we still need
            # to factor in σ_data² / (σ̃² + σ_data²) · (α² · σ̃² · ...).
            # Simpler: derive the full ``wtilde`` here.
            # Recall: σ̃² = (1 − α²) / α², so α² · σ̃² = 1 − α².  Thus
            # the σ_data ≡ 1 V2 weight ``β · α² / (1 − α²)`` is exactly
            # ``β / σ̃² · α²``; equivalently
            # ``wtilde[k] = (1/2·dτ) · β · α² / (1−α²)``.
            # With σ_data dependence:
            #   w(τ;t) = (β / (α² · σ̃² · (σ̃² + σd²))) · σ_d²
            # Plug ``α² · σ̃² = 1 − α²`` in:
            #   w(τ;t) = (β · σ_d²) / ((1 − α²) · (σ̃² + σ_d²))
            # We store wtilde_base = (1/2·dτ) · β / (1 − α²), so:
            #   wtilde(k;t) = wtilde_base[k] · σ_d² / (σ̃² + σ_d²)
            wtilde_full = (
                wtilde_base_flat * sd2_flat / (st2_flat + sd2_flat).clamp_min(eps_dtype)
            )
            # Importance-sampling correction. ``wtilde_base`` already bakes in
            # the (½·dτ) Riemann measure, so this is the standard estimator of
            # the sum ∫ ≈ Σ_k (½·dτ·w_k)·X_k  by drawing  k ~ p_k:  divide the
            # baked weight by the sampling mass p_k (one-over-density × weight,
            # per model-v2.org § Importance Sampling). Do NOT also divide by K —
            # that double-counts the τ-measure (dτ ≈ 1/K already in wtilde_base)
            # and shrinks the per-t ESM loss by a factor of K. ``gather``
            # picks each row's p_k at the row-specific sampled k.
            p_k_at_sample = p_k_per_row.gather(1, k_idx).reshape(N * kc)
            weights = (wtilde_full / p_k_at_sample.clamp_min(eps_dtype)).detach()

            if (
                self.grad_checkpoint
                and self.training
                and torch.is_grad_enabled()
                and not return_per_sample
            ):
                # Recompute the score-net (and its d²-attention activations) in
                # backward instead of retaining it across every timestep chunk.
                # use_reentrant=False is the torch.compile-compatible variant; the
                # score-net is deterministic given inputs (RNG is the multinomial/
                # randn above), so no RNG-state handling is needed here.
                F_pred = checkpoint(
                    self.diffmodel,
                    latent_w,
                    side_w,
                    c_noise_flat,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )  # (N*kc, d, 1)
            else:
                F_pred = self.diffmodel(latent_w, side_w, c_noise_flat)  # (N*kc, d, 1)
            F_pred = F_pred.squeeze(-1)
            F_tgt_flat = F_target.permute(0, 2, 1).reshape(N * kc, d)

            # Fork at the unweighted squared error: phith gets the IS
            # weights (arithmetic order identical to the pre-split code),
            # psi gets the same sqerr unit-weighted.  Both share F_pred.
            sqerr = (F_pred - F_tgt_flat).pow(2).sum(dim=1)
            total_sqerr_phith = total_sqerr_phith + (sqerr * weights).view(N, kc).sum(
                dim=1
            )
            total_sqerr_psi = total_sqerr_psi + sqerr.view(N, kc).sum(dim=1)

        # S_k normalisation applied symmetrically to both accumulators.
        per_sample_phith = total_sqerr_phith / float(self.S_k)
        per_sample_psi = total_sqerr_psi / float(self.S_k)
        if return_per_sample:
            return per_sample_phith, per_sample_psi, mu_hat_t
        return per_sample_phith.sum(), per_sample_psi.sum(), mu_hat_t

    # ------------------------------------------------------------------
    # σ_data-aware EDM preconditioning.
    # ------------------------------------------------------------------
    def _vp_precondition(
        self,
        mu_hat_t: torch.Tensor,  # (N, d) — CENTERED mean
        sigma2_t: torch.Tensor,  # (N, d)
        k_idx: torch.Tensor,  # (N, S_k)
        eps: torch.Tensor,  # (N, d, S_k)
        sigma_d2_per_row: torch.Tensor,  # (N,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(z_in, F_target)`` in centered coordinates.

        EDM constants vary with σ_data²(t) per row:
          ``c_skip = σ_d² / (σ̃² + σ_d²)``
          ``c_out  = σ̃·σ_d / √(σ̃² + σ_d²)``
          ``c_in   = 1 / √(σ̃² + σ_d²)``
        Reduce to V2's σ_data ≡ 1 constants in the unit limit.
        """
        eps_dtype = torch.finfo(mu_hat_t.dtype).eps
        sigma_tilde = self.sigma_tilde[k_idx]  # (N, S_k)
        sigma_tilde2 = sigma_tilde * sigma_tilde

        sd2 = sigma_d2_per_row.view(-1, 1).clamp_min(eps_dtype)  # (N, 1)
        sd = sd2.sqrt()
        denom = (sigma_tilde2 + sd2).clamp_min(eps_dtype)  # (N, S_k)
        sqrt_denom = denom.sqrt()

        c_skip = sd2 / denom  # (N, S_k)
        c_out = (sigma_tilde * sd) / sqrt_denom  # (N, S_k)
        c_in = 1.0 / sqrt_denom  # (N, S_k)

        # Broadcast to (N, 1, S_k) for the latent dim.
        st2_ = sigma_tilde2.unsqueeze(1)
        cskip_ = c_skip.unsqueeze(1)
        cout_ = c_out.unsqueeze(1).clamp_min(eps_dtype)
        cin_ = c_in.unsqueeze(1)

        sigma2_t_ = sigma2_t.unsqueeze(-1)  # (N, d, 1)
        mu_hat_t_ = mu_hat_t.unsqueeze(-1)  # (N, d, 1)

        var_total = (sigma2_t_ + st2_).clamp_min(eps_dtype)  # (N, d, S_k)
        z_hat = mu_hat_t_ + var_total.sqrt() * eps  # centered residual

        s_q_hat = -(z_hat - mu_hat_t_) / var_total
        D_star = z_hat + st2_ * s_q_hat
        F_target = (D_star - cskip_ * z_hat) / cout_
        z_in = cin_ * z_hat

        return z_in, F_target

    # ------------------------------------------------------------------
    # Native-coord score (model-v2.org § Exact likelihood, Layer 2).
    # ------------------------------------------------------------------
    def score(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        z_hist: torch.Tensor,
        ctx: dict[str, Any],
        sigma_d2: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Native-coordinate score ``s_ψ(z, τ, z_{t-1})`` for the prob-flow ODE.

        Composes the EDM-parameterized denoiser ``F_ψ`` into a
        native-coordinate score per model-v2.org § Native-coordinate
        score reconstruction::

            ẑ      = z / α_τ − μ_p(z_hist)
            D_ψ    = c_skip · ẑ + c_out · F_ψ(c_in · ẑ, τ, z_hist)
            s_ψ(z) = (D_ψ − ẑ) / (α_τ · σ̃_τ²)

        α(τ), σ̃(τ), c_noise(τ) are evaluated closed-form from the
        VP-SDE schedule for arbitrary continuous τ ∈ (0, 1] — the buffer
        grid is only used for ``transition_kl`` / sampling, not here.
        The σ_data-dependent EDM constants follow the same per-row form
        as :meth:`_vp_precondition`.

        Args:
            z: ``(B, d)`` latent in native coordinates.
            tau: ``(B,)`` continuous τ value per row.
            z_hist: ``(B, d, j)`` previous latents (conditioning).
            ctx: must include ``hist_time_emb (B, j, E_t)`` and
                ``target_time_emb (B, 1, E_t)``; optionally
                ``hist_covariates``/``target_covariates``.
            sigma_d2: ``(B,)`` σ_data²(t) per row.
            padding_mask: ``(B, j+1)`` padding-mask channel; defaults
                to zeros (no padding) matching the forecast-sampler path.

        Returns:
            ``(B, d)`` native-coordinate score.
        """
        B, d = z.shape
        if d != self.latent_dim:
            raise ValueError(f"z latent dim {d} != self.latent_dim {self.latent_dim}")
        device = z.device
        dtype = z.dtype
        eps_dtype = torch.finfo(dtype).eps

        beta_min = float(self.schedule.beta_min)
        beta_max = float(self.schedule.beta_max)
        int_beta = beta_min * tau + 0.5 * (beta_max - beta_min) * tau.pow(2)
        alpha = torch.exp(-0.5 * int_beta)
        alpha2 = alpha.pow(2).clamp_min(eps_dtype)
        sigma_tilde2 = (1.0 - alpha2) / alpha2
        sigma_tilde = sigma_tilde2.clamp_min(eps_dtype).sqrt()
        c_noise = 0.25 * torch.log(sigma_tilde.clamp_min(eps_dtype))

        sd2 = sigma_d2.clamp_min(eps_dtype)
        denom = (sigma_tilde2 + sd2).clamp_min(eps_dtype)
        sqrt_denom = denom.sqrt()
        c_skip = sd2 / denom
        c_out = (sigma_tilde * sd2.sqrt()) / sqrt_denom
        c_in = 1.0 / sqrt_denom

        alpha_col = alpha.unsqueeze(-1).clamp_min(eps_dtype)
        mu_p = self.baseline.mean(z_hist)
        z_hat = z / alpha_col - mu_p

        if "hist_time_emb" not in ctx or "target_time_emb" not in ctx:
            raise ValueError(
                "DiffusionTransition.score requires hist/target time embeddings in ctx"
            )
        hist_time = ctx["hist_time_emb"]
        tgt_time = ctx["target_time_emb"]
        if "hist_covariates" in ctx:
            hist_time = torch.cat([hist_time, ctx["hist_covariates"]], dim=-1)
        if "target_covariates" in ctx:
            tgt_time = torch.cat([tgt_time, ctx["target_covariates"]], dim=-1)
        time_win = torch.cat([hist_time, tgt_time], dim=1)

        cond_mask = torch.ones(B, d, self.j + 1, device=device, dtype=dtype)
        cond_mask[..., -1] = 0.0
        if padding_mask is None:
            padding_mask = torch.zeros(B, self.j + 1, device=device, dtype=dtype)
        side_win = get_side_info(
            data_dim=self.latent_dim,
            time_embed=time_win,
            embed_layer=self.embed_layer,
            cond_mask=cond_mask,
            device=device,
            padding_mask=padding_mask,
        )

        z_in = (c_in.unsqueeze(-1) * z_hat).unsqueeze(-1)
        latent_w = torch.cat([z_hist, z_in], dim=2)
        F_pred = self.diffmodel(latent_w, side_win, c_noise).squeeze(-1)

        D_psi = c_skip.unsqueeze(-1) * z_hat + c_out.unsqueeze(-1) * F_pred
        s_tilde = (D_psi - z_hat) / sigma_tilde2.unsqueeze(-1).clamp_min(eps_dtype)
        return s_tilde / alpha_col

    # ------------------------------------------------------------------
    # Sampling: VP probability-flow ODE in centered coords.
    # ------------------------------------------------------------------
    def sample(
        self,
        z_hist: torch.Tensor,  # (B, d, j)
        S: int = 1,
        ctx: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Reverse-SDE sample at ``t ≥ j+1`` in centered coords + add μ_p back.

        Args:
            z_hist: ``(B, d, j)`` real previous latents.
            S: ignored (returns one trajectory per row).
            ctx: must include ``hist_time_emb`` and ``target_time_emb``,
                and optionally ``sigma_data`` and ``t`` (1-based).  When
                ``sigma_data`` is missing we fall back to σ_data ≡ 1.
        """
        del S
        if ctx is None:
            raise ValueError("DiffusionTransition.sample requires ctx")
        if "hist_time_emb" not in ctx or "target_time_emb" not in ctx:
            raise ValueError(
                "DiffusionTransition.sample requires hist/target time embeddings"
            )
        B, d, j_in = z_hist.shape
        if j_in != self.j:
            raise ValueError(f"Expected history j={self.j}, got {j_in}")

        device = z_hist.device
        dtype = z_hist.dtype

        # σ_data² lookup for this t.
        sigma_data: SigmaDataBuffer | None = ctx.get("sigma_data")
        t_external = int(ctx.get("t", self.j + 1))
        if sigma_data is not None:
            # Constant extrapolation beyond the trained horizon
            # (model-v2.org:1442-1459): F_ψ does not take t as input, so for
            # t > T reuse σ_data²[T] — keeps the EDM constants in the regime the
            # network was calibrated against. read() is strict, so clamp here.
            t_clamped = max(1, min(t_external, int(sigma_data.T_max)))
            sd2_scalar = float(sigma_data.read(t_clamped).item())
        else:
            sd2_scalar = 1.0

        # Build side-info window.
        hist_time = ctx["hist_time_emb"]
        tgt_time = ctx["target_time_emb"]
        if "hist_covariates" in ctx:
            hist_time = torch.cat([hist_time, ctx["hist_covariates"]], dim=-1)
        if "target_covariates" in ctx:
            tgt_time = torch.cat([tgt_time, ctx["target_covariates"]], dim=-1)
        time_win = torch.cat([hist_time, tgt_time], dim=1)  # (B, j+1, E+V)
        cond_mask = torch.ones(B, d, self.j + 1, device=device, dtype=dtype)
        cond_mask[..., -1] = 0.0
        # Padding mask for forecast sampling (history is real, target is target):
        # all zeros.
        padding_mask = torch.zeros(B, self.j + 1, device=device, dtype=dtype)
        side_win = get_side_info(
            data_dim=self.latent_dim,
            time_embed=time_win,
            embed_layer=self.embed_layer,
            cond_mask=cond_mask,
            device=device,
            padding_mask=padding_mask,
        )

        mu_p = self.baseline.mean(z_hist)  # (B, d) — added back at end
        if self.sampler == "edm":
            z_hat_sample = self._edm_sample_centered(
                z_hist=z_hist,
                side_win=side_win,
                sigma_d2=sd2_scalar,
            )
        else:
            z_hat_sample = self._vp_pf_sample_centered(
                z_hist=z_hist,
                side_win=side_win,
                sigma_d2=sd2_scalar,
            )
        z_sample = z_hat_sample + mu_p
        return z_sample.unsqueeze(1)  # (B, 1, d)

    @torch.no_grad()
    def _vp_pf_sample_centered(
        self,
        z_hist: torch.Tensor,  # (B, d, j)
        side_win: torch.Tensor,  # (B, side_dim, d, j+1)
        sigma_d2: float,
    ) -> torch.Tensor:
        """Reverse probability-flow Euler sampler in centered coords."""
        B, d, _ = z_hist.shape
        device = z_hist.device
        dtype = z_hist.dtype
        eps_dtype = torch.finfo(dtype).eps

        # Prior at the top grid step (τ ≈ 1) in centered coords. The unit-diffusion
        # VP forward marginal is N(0, α²·σ_d² + (1−α²)): the OU process drives
        # any data variance toward the stationary variance 1 at τ=1, so the
        # terminal is ≈ N(0, I) regardless of σ_data. Initialise at that exact
        # marginal (reduces to N(0, I) at σ_data ≡ 1 and matches the prob-flow
        # likelihood's N(0, I) terminal prior). The previous
        # N(0, max(σ_d², 1)) over-dispersed forecast samples when σ_data² > 1.
        #
        # Sampling-schedule buffers (``self.sample_*``) are separate from
        # the training grid; when no sampling_schedule was provided the
        # ``sample_*`` aliases reference the training buffers (so this
        # branch is unchanged for that case).
        a2_top = float(self.sample_alpha2[-1].item())
        om_a2_top = float(self.sample_one_minus_alpha2[-1].item())
        var_init = max(a2_top * sigma_d2 + om_a2_top, eps_dtype)
        x = math.sqrt(var_init) * torch.randn(B, d, device=device, dtype=dtype)

        K = self.sample_num_steps
        for i in range(K - 1, 0, -1):
            alpha_i = float(self.sample_alpha[i].item())
            sigma_tilde_i = float(self.sample_sigma_tilde[i].item())
            sigma_tilde2_i = sigma_tilde_i * sigma_tilde_i
            denom = sigma_tilde2_i + sigma_d2
            sqrt_denom = math.sqrt(max(denom, eps_dtype))
            c_skip_i = sigma_d2 / max(denom, eps_dtype)
            c_out_i = (sigma_tilde_i * math.sqrt(sigma_d2)) / sqrt_denom
            c_in_i = 1.0 / sqrt_denom
            c_noise_i = float(self.sample_c_noise[i].item())

            # Convert native x to VE-coords ẑ_tilde (drives the network).
            z_tilde = x / max(alpha_i, eps_dtype)
            z_in = (c_in_i * z_tilde).unsqueeze(2)  # (B, d, 1)
            latent_w = torch.cat([z_hist, z_in], dim=2)
            c_noise_vec = torch.full((B,), c_noise_i, device=device, dtype=dtype)
            F_pred = self.diffmodel(latent_w, side_win, c_noise_vec).squeeze(-1)
            D_pred = c_skip_i * z_tilde + c_out_i * F_pred
            # Score in VE coords; native-coord score = s_tilde / α.
            s_tilde = (D_pred - z_tilde) / max(sigma_tilde2_i, eps_dtype)
            s_native = s_tilde / max(alpha_i, eps_dtype)

            beta_i = float(self.sample_beta[i].item())
            d_tau = float((self.sample_tau[i - 1] - self.sample_tau[i]).item())  # < 0
            drift = -0.5 * beta_i * x - 0.5 * beta_i * s_native
            x = x + d_tau * drift

        return x

    @torch.no_grad()
    def _edm_sample_centered(
        self,
        z_hist: torch.Tensor,  # (B, d, j)
        side_win: torch.Tensor,  # (B, side_dim, d, j+1)
        sigma_d2: float,
    ) -> torch.Tensor:
        """EDM (Karras 2022) Heun sampler with optional stochastic churn.

        Operates in VE/centered coords: the network sees ẑ_tilde with marginal
        N(z_centered, σ̃²) and the EDM preconditioning maps F→D(ẑ;σ̃). The σ̃ grid
        reuses the trained ``sample_sigma_tilde`` buffer's endpoints; the
        intermediate nodes follow the EDM ρ-power schedule. The final x at σ→0 is
        the centered native clean latent (= z_hat_sample); ``sample()`` adds μ_p.
        """
        B, d, _ = z_hist.shape
        device = z_hist.device
        dtype = z_hist.dtype
        eps_dtype = torch.finfo(dtype).eps
        sd2 = max(float(sigma_d2), eps_dtype)
        sd = math.sqrt(sd2)
        sigma_max = float(self.sample_sigma_tilde[-1].item())
        sigma_min = float(self.sample_sigma_tilde[0].item())
        if self.edm_sigma_max_rel is not None:
            sigma_max = min(sigma_max, self.edm_sigma_max_rel * sd)
        if self.edm_sigma_min_rel is not None:
            sigma_min = max(sigma_min, self.edm_sigma_min_rel * sd)
        N = int(self.sample_num_steps)
        rho = float(self.edm_rho)

        def denoise(x_tilde: torch.Tensor, sigma: float) -> torch.Tensor:
            sigma2 = sigma * sigma
            denom = sigma2 + sd2
            sqrt_denom = math.sqrt(denom)
            c_skip = sd2 / denom
            c_out = (sigma * sd) / sqrt_denom
            c_in = 1.0 / sqrt_denom
            c_noise = 0.25 * math.log(max(sigma, eps_dtype))
            z_in = (c_in * x_tilde).unsqueeze(2)  # (B, d, 1)
            latent_w = torch.cat([z_hist, z_in], dim=2)
            c_noise_vec = torch.full((B,), c_noise, device=device, dtype=dtype)
            F_pred = self.diffmodel(latent_w, side_win, c_noise_vec).squeeze(-1)
            return c_skip * x_tilde + c_out * F_pred

        # EDM ρ-power σ schedule, descending to a final 0.0 node.
        if N > 1:
            ramp = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
        else:
            ramp = torch.zeros(1, dtype=torch.float64)
        min_inv = sigma_min ** (1.0 / rho)
        max_inv = sigma_max ** (1.0 / rho)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
        sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)]).tolist()

        gamma_max = math.sqrt(2.0) - 1.0
        churn = (
            min(self.edm_s_churn / max(N, 1), gamma_max)
            if self.edm_s_churn > 0
            else 0.0
        )

        # Init at the EDM-coords prior: ẑ_tilde ~ N(0, σ_max² + σ_d²) (the VE marginal
        # of centered data perturbed to σ_max). Matches the c_skip/c_out contract.
        x = math.sqrt(sigma_max * sigma_max + sd2) * torch.randn(
            B, d, device=device, dtype=dtype
        )
        for i in range(N):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]
            gamma = churn if (self.edm_s_tmin <= sigma_cur <= self.edm_s_tmax) else 0.0
            sigma_hat = min(sigma_cur * (1.0 + gamma), sigma_max)
            if sigma_hat > sigma_cur:
                noise = torch.randn(B, d, device=device, dtype=dtype)
                x = (
                    x
                    + math.sqrt(max(sigma_hat * sigma_hat - sigma_cur * sigma_cur, 0.0))
                    * self.edm_s_noise
                    * noise
                )
            d_cur = (x - denoise(x, sigma_hat)) / max(sigma_hat, eps_dtype)
            x_next = x + (sigma_next - sigma_hat) * d_cur
            if sigma_next > 0.0:
                d_next = (x_next - denoise(x_next, sigma_next)) / sigma_next
                x_next = x + (sigma_next - sigma_hat) * 0.5 * (d_cur + d_next)
            x = x_next

        return x


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _update_sigma_data_blocked(
    *,
    sigma_data: SigmaDataBuffer,
    mu_hat: torch.Tensor,  # (B*S*chunk_len, d) — (B, S, chunk_len) order
    sigma2_t: torch.Tensor,  # (B*S*chunk_len, d)
    B: int,
    S: int,
    chunk_len: int,
    d: int,
    t_start_external: int,
) -> None:
    """Reorganise (B, S, chunk_len) flattened input into (chunk_len, B*S) blocks."""
    mu_hat_b = (
        mu_hat
        .view(B, S, chunk_len, d)
        .permute(2, 0, 1, 3)
        .reshape(chunk_len * B * S, d)
    )
    sigma2_b = (
        sigma2_t
        .view(B, S, chunk_len, d)
        .permute(2, 0, 1, 3)
        .reshape(chunk_len * B * S, d)
    )
    t_idx = torch.arange(
        t_start_external,
        t_start_external + chunk_len,
        device=mu_hat.device,
        dtype=torch.long,
    )
    sigma_data.update(t_idx=t_idx, mu_hat_batch=mu_hat_b, sigma_t2_batch=sigma2_b)
