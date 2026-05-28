"""Stage-2 diffusion transition for the model-v2 baseline-centering scheme.

``DiffusionV3Transition`` extends :class:`DiffusionV2Transition` along
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
   reduce to V2's hardcoded values (a unit test exercises this).

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
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, final

import torch
import torch.nn as nn

from ..aux_posterior import AuxPosterior
from ..centering.baselines import BaseBaseline
from ..centering.sigma_data import SigmaDataBuffer
from ..diffnets import CSDIUnet
from ..gaussians import GaussianStats
from ..net_utils import get_side_info
from ..torch_compile import maybe_compile
from .transitions import BaseTransition


@dataclass
class DiffusionV3ScheduleConfig:
    """VP-SDE schedule configuration for :class:`DiffusionV3Transition`.

    Mirrors :class:`DiffusionV2ScheduleConfig`; the (σ_data-dependent)
    EDM constants are computed per call rather than precomputed because
    they vary with the current ``σ_data²(t)`` buffer value.
    """

    S_k: int = 1
    k_chunk: int = 1
    num_steps: int = 100
    beta_min: float = 0.1
    beta_max: float = 20.0
    tau_min: float = 1e-3
    k_sampling_mode: str = "uniform"
    pk_gamma: float = 1.0
    pk_floor: float = 1e-12


@final
class DiffusionV3Transition(BaseTransition):
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
        schedule: DiffusionV3ScheduleConfig | None = None,
    ) -> None:
        super().__init__()
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
            schedule = DiffusionV3ScheduleConfig()
        self.schedule = schedule
        self.S_k = schedule.S_k
        self.num_steps = schedule.num_steps

        # Feature + side-info dim — +1 for cond_mask, +1 for padding_mask
        # (precursor (iii) from init-experiment.org § Implementation precursors).
        self.emb_feature_dim = emb_time_dim
        self.side_dim = (
            self.emb_time_dim + self.covariate_dim + self.emb_feature_dim + 2
        )

        if unet is None:
            unet = partial(
                CSDIUnet, channels=64, n_layers=4, embedding_dim=128,
            )
        self.diffmodel = unet(
            output_len=1,
            diffusion_steps=schedule.num_steps,
            latent_dim=self.latent_dim,
            latent_history_len=self.j,
            side_dim=self.side_dim,
            zero_init_output=True,
        )
        self.diffmodel = maybe_compile(self.diffmodel)

        self.embed_layer = nn.Embedding(
            num_embeddings=self.latent_dim, embedding_dim=self.emb_feature_dim
        )

        # ---------- VP-SDE precompute (σ_data-independent quantities) ----------
        dtype64 = torch.float64
        eps64 = torch.finfo(dtype64).eps
        K = self.num_steps
        beta_min = float(schedule.beta_min)
        beta_max = float(schedule.beta_max)
        tau_min = float(schedule.tau_min)
        if not (0.0 < tau_min < 1.0):
            raise ValueError(f"tau_min must be in (0, 1); got {tau_min}")
        if beta_max <= beta_min:
            raise ValueError(f"beta_max ({beta_max}) must be > beta_min ({beta_min})")

        tau = torch.linspace(tau_min, 1.0, K, dtype=dtype64)
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
        dtau = (1.0 - tau_min) / float(K)
        # Bake the (1/2 * dtau) Riemann measure into ``wtilde_base``.
        # The full weight is ``wtilde_base[k] · σ_data²(t) / (σ̃² + σ_data²(t))``;
        # this gets applied in ``_esm_chunk_loss`` once σ_data is known.
        wtilde_base = 0.5 * dtau * beta / one_minus_alpha2

        self.register_buffer("alpha", alpha.to(torch.float32))
        self.register_buffer("alpha2", alpha2.to(torch.float32))
        self.register_buffer("sigma_tilde", sigma_tilde.to(torch.float32))
        self.register_buffer("one_minus_alpha2", one_minus_alpha2.to(torch.float32))
        self.register_buffer("c_noise", c_noise.to(torch.float32))
        self.register_buffer("beta", beta.to(torch.float32))
        self.register_buffer("tau", tau.to(torch.float32))
        self.register_buffer("w_per_tau_unit", w_per_tau_unit.to(torch.float32))
        self.register_buffer("wtilde_base", wtilde_base.to(torch.float32))

        # Importance-sampling distribution p_k.
        ismode = schedule.k_sampling_mode
        self.gamma = float(schedule.pk_gamma)
        self.gfloor = float(schedule.pk_floor)
        if ismode == "lsgm_is":
            p_k = (beta / one_minus_alpha2).to(torch.float32).clamp_min(self.gfloor)
            if self.gamma != 1.0:
                p_k = p_k.pow(self.gamma)
            p_k = p_k / p_k.sum()
        elif ismode == "uniform":
            p_k = torch.full(
                (self.num_steps,), 1.0 / self.num_steps, dtype=torch.float32
            )
        else:
            raise ValueError(
                f"Unknown k_sampling_mode={ismode!r}; expected 'uniform' or 'lsgm_is'"
            )
        self.register_buffer("p_k", p_k)
        self.k_sampling_mode = ismode

    # ------------------------------------------------------------------
    # Per-transition log-density via probability-flow ODE.
    # ------------------------------------------------------------------
    def log_prob(
        self,
        z: torch.Tensor,
        z_hist: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None,
        mc_override: Optional[Dict[str, Any]] = None,
        *,
        sigma_d2: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "dopri5",
        divergence_mode: str = "exact",
        generator: Optional[torch.Generator] = None,
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
                (matches the V3 sampler's ``sigma_data ≡ 1`` fallback).
            padding_mask: ``(B, j+1)`` padding-mask channel; defaults
                to zeros (no padding).
            rtol, atol, method: torchdiffeq adaptive-solver controls.
            divergence_mode: ``"exact"`` (cycle 2) or ``"hutchinson"``
                (cycle 3).

        Returns:
            ``(B,)`` per-row log-density.
        """
        del mc_override
        from ..likelihood import solve_prob_flow_logdensity

        if ctx is None:
            raise ValueError("DiffusionV3Transition.log_prob requires ctx")
        B, d = z.shape
        if sigma_d2 is None:
            sigma_d2 = torch.ones(B, device=z.device, dtype=z.dtype)

        def score_fn(z_curr: torch.Tensor, tau_curr: torch.Tensor) -> torch.Tensor:
            tau_b = tau_curr.expand(z_curr.shape[0]) if tau_curr.dim() == 0 else tau_curr
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
        zs: torch.Tensor,             # (B, S, d, T)
        logq_paths: torch.Tensor,     # (B, S, T)  — unused
        time_embed: torch.Tensor,     # (B, T, E_t)
        sigma_data: SigmaDataBuffer,
        covariates: Optional[torch.Tensor] = None,
        mc_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Centered ESM/EDM loss over ``t = j+1 … T``.

        Returns ``{"kl": ...}``.  ``R_μp`` is added at the
        :class:`DDSSM_base.forward` level via the free function — V3
        is a pure loss-computer per the plan's ownership decisions.
        """
        del logq_paths
        if (
            "mus" not in enc_stats
            or "logvars" not in enc_stats
            or enc_stats["mus"] is None
            or enc_stats["logvars"] is None
        ):
            raise ValueError(
                "DiffusionV3Transition.transition_kl requires Gaussian (mus, logvars)."
            )

        B, S, d, T = zs.shape
        j = self.j
        if d != self.latent_dim:
            raise ValueError(f"zs latent dim {d} != self.latent_dim {self.latent_dim}")

        device = zs.device
        dtype = zs.dtype
        kl_sum = torch.zeros((), device=device, dtype=dtype)
        n_target_steps = max(0, T - j)
        if n_target_steps == 0:
            return {"kl": kl_sum}

        mus = enc_stats["mus"]
        logvars = enc_stats["logvars"]

        for (
            B_,
            S_,
            chunk_len,
            t_start,
            t_end,
            _z_target_flat,
            z_hist_flat,   # (N, d, j)
            ctx,
        ) in self._iter_window_chunks(zs, time_embed, covariates=covariates):
            N = B_ * S_ * chunk_len
            # Slice encoder stats for the chunk's targets.
            mu_t_flat = (
                mus[..., t_start:t_end].permute(0, 1, 3, 2).reshape(N, d)
            )
            sigma2_t_flat = (
                logvars[..., t_start:t_end]
                .exp()
                .permute(0, 1, 3, 2)
                .reshape(N, d)
            )

            # Per-row σ_data²(t).  Row r → c = r % chunk_len → t = t_start + c (0-based).
            t_idx = torch.arange(
                t_start + 1, t_end + 1, device=device, dtype=torch.long
            )  # 1-based, (chunk_len,)
            sigma_d2_per_t = sigma_data.read(t_idx).to(dtype=dtype)  # (chunk_len,)
            sigma_d2_per_row = (
                sigma_d2_per_t.view(1, 1, chunk_len)
                .expand(B_, S_, chunk_len)
                .reshape(N)
            )

            # Padding mask: all-zeros for t ≥ j+1 (no aux slots).
            padding_mask = torch.zeros(N, j + 1, device=device, dtype=dtype)

            chunk_loss = self._esm_chunk_loss(
                mu_t=mu_t_flat,
                sigma2_t=sigma2_t_flat,
                z_hist=z_hist_flat,
                ctx=ctx,
                sigma_d2_per_row=sigma_d2_per_row,
                padding_mask=padding_mask,
                mc_override=mc_override,
            )
            kl_sum = kl_sum + chunk_loss

            # σ_data update at this chunk's timesteps.
            mu_p_chunk = self.baseline.mean(z_hist_flat)  # (N, d)
            mu_hat = mu_t_flat - mu_p_chunk
            _update_sigma_data_blocked(
                sigma_data=sigma_data,
                mu_hat=mu_hat,
                sigma2_t=sigma2_t_flat,
                B=B_, S=S_, chunk_len=chunk_len, d=d,
                t_start_external=t_start + 1,
            )

        denom = float(B * S * n_target_steps)
        kl = kl_sum / denom
        return {"kl": kl}

    # ------------------------------------------------------------------
    # transition_kl_init  (t = 1 … j)
    # ------------------------------------------------------------------
    def transition_kl_init(
        self,
        enc_stats: GaussianStats,
        zs: torch.Tensor,
        aux_posterior: AuxPosterior,
        time_embed: torch.Tensor,
        sigma_data: SigmaDataBuffer,
        covariates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stage-2 VHP init term — same ESM/EDM surrogate at t = 1 … j.

        Mirrors :meth:`BaselineGaussianTransition.transition_kl_init`'s
        mixed-history walk but uses the centered ESM/EDM surrogate
        (with the padding-mask channel flagging aux slots) instead of
        the closed-form Gaussian log-density.

        Per ``model-v2.org`` § Entropy cancellation in stage 2, the
        returned ``loss_init`` is the entropy-cancelled surrogate;
        :class:`DDSSM_base` does *not* add ``-H(q_φ)`` separately.

        Returns ``{"loss_init", "kl_aux"}``.
        """
        del covariates  # baseline-centering doesn't condition on them
        if (
            "mus" not in enc_stats
            or "logvars" not in enc_stats
            or enc_stats["mus"] is None
            or enc_stats["logvars"] is None
        ):
            raise ValueError(
                "transition_kl_init requires Gaussian encoder stats (mus, logvars)."
            )

        B, S, d, T = zs.shape
        j = self.j
        device = zs.device
        dtype = zs.dtype
        if d != self.latent_dim:
            raise ValueError(f"zs latent dim {d} != self.latent_dim {self.latent_dim}")
        if T < j:
            raise ValueError(f"zs has T={T} < j={j}")

        # Sample aux latents from q_Φ(z_aux | z_{1:j}), conditioned on the
        # S-averaged encoder samples.
        z_init = zs[..., :j].mean(dim=1)  # (B, d, j)
        z_aux, aux_mu, aux_logvar = aux_posterior.sample(z_init)  # (B, d, j) each

        BS = B * S
        # Broadcast z_aux per-S so every S sample sees the same aux draw.
        z_hist = (
            z_aux.unsqueeze(1)
            .expand(B, S, d, j)
            .reshape(BS, d, j)
            .clone()
        )

        # Build per-step time-window context lazily.  At each init step we
        # need (BS, j+1, E_t) for the side-info construction.
        total_loss = torch.zeros((), device=device, dtype=dtype)
        for step in range(j):
            # z_t encoder stats, (BS, d).
            mu_t_flat = enc_stats["mus"][:, :, :, step].reshape(BS, d)
            sigma2_t_flat = enc_stats["logvars"][:, :, :, step].exp().reshape(BS, d)

            # σ_data lookup at 1-based t = step + 1.
            t_external = step + 1
            sigma_d2_value = sigma_data.read(t_external).to(dtype=dtype)
            sigma_d2_per_row = sigma_d2_value.expand(BS)

            # Padding mask over the (j+1) slots of the score-net window:
            # the first ``j - step`` slots are aux (1.0), the next
            # ``step`` slots are real (0.0), and the last slot is the
            # target (0.0).
            padding_mask = torch.zeros(BS, j + 1, device=device, dtype=dtype)
            n_aux_slots = j - step
            if n_aux_slots > 0:
                padding_mask[:, :n_aux_slots] = 1.0

            # Build the (BS, j+1, E_t) time window for the init step.
            # The j history slots correspond to abstract timesteps
            # ``t - j … t - 1`` (1-based ``t``).  We clamp the
            # abstract-time index to ``[0, T - 1]`` so the side-info
            # tensor can use the encoder's time_embed grid.
            tgt_idx = step  # 0-based code index of z_t
            hist_idx = torch.arange(
                tgt_idx - j, tgt_idx, device=device, dtype=torch.long,
            ).clamp(min=0, max=T - 1)
            # (j, E_t)
            hist_te_per_batch = time_embed.index_select(1, hist_idx)  # (B, j, E_t)
            tgt_te_per_batch = time_embed[:, tgt_idx : tgt_idx + 1, :]  # (B, 1, E_t)
            time_win_batch = torch.cat(
                [hist_te_per_batch, tgt_te_per_batch], dim=1
            )  # (B, j+1, E_t)
            time_win = (
                time_win_batch.unsqueeze(1)
                .expand(B, S, j + 1, self.emb_time_dim)
                .reshape(BS, j + 1, self.emb_time_dim)
            )

            ctx_step: Dict[str, torch.Tensor] = {
                "hist_time_emb": time_win[:, :j, :],
                "target_time_emb": time_win[:, j : j + 1, :],
            }

            chunk_loss = self._esm_chunk_loss(
                mu_t=mu_t_flat,
                sigma2_t=sigma2_t_flat,
                z_hist=z_hist,
                ctx=ctx_step,
                sigma_d2_per_row=sigma_d2_per_row,
                padding_mask=padding_mask,
            )
            total_loss = total_loss + chunk_loss

            # σ_data update at this init t.
            mu_p_step = self.baseline.mean(z_hist)  # (BS, d)
            mu_hat = mu_t_flat - mu_p_step
            sigma_data.update(
                t_idx=torch.tensor([t_external], device=device),
                mu_hat_batch=mu_hat,
                sigma_t2_batch=sigma2_t_flat,
            )

            # Shift history: drop oldest, append the real z_t sample.
            z_t = zs[:, :, :, step].reshape(BS, d)
            if j > 1:
                z_hist = torch.cat([z_hist[:, :, 1:], z_t.unsqueeze(-1)], dim=-1)
            else:
                z_hist = z_t.unsqueeze(-1)

        loss_init = total_loss / float(BS)
        kl_aux = aux_posterior.kl_against_standard_normal(aux_mu, aux_logvar)
        return {"loss_init": loss_init, "kl_aux": kl_aux}

    # ------------------------------------------------------------------
    # VHP initial-state log-density (model-v2.org § Exact likelihood, Layer 4).
    # ------------------------------------------------------------------
    def log_prob_init(
        self,
        zs: torch.Tensor,
        aux_posterior: AuxPosterior,
        time_embed: torch.Tensor,
        sigma_data: Optional[SigmaDataBuffer] = None,
        covariates: Optional[torch.Tensor] = None,
        *,
        J: int = 1,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "dopri5",
        divergence_mode: str = "exact",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """VHP importance-sampled ``log p_ψ(z_{1:j})`` per trajectory.

        Layer 4 of the exact-likelihood evaluator (model-v2.org § "VHP
        initial state").  Mirrors :meth:`transition_kl_init`'s
        mixed-history walk over the first ``j`` steps, but instead of the
        ESM/EDM surrogate it accumulates the probability-flow ODE
        log-densities ``log p_ψ^ode(z_step | z_hist_step)`` (via
        :meth:`log_prob`) with ``z_0 ∼ q_Φ`` in the aux slots, then
        reduces the importance weights

            log p_ψ(z_{1:j}) ≈ logmeanexp_J[
                Σ_step log p_ψ^ode(z_step | z_hist)
                + log N(z_0; 0, I) − log q_Φ(z_0 | z_{1:j})
            ]

        via :func:`ddssm.likelihood.vhp.vhp_log_prob_init`.

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
        from ..likelihood import vhp_log_prob_init

        B, S, d, T = zs.shape
        j = self.j
        device = zs.device
        dtype = zs.dtype
        if d != self.latent_dim:
            raise ValueError(f"zs latent dim {d} != self.latent_dim {self.latent_dim}")
        if T < j:
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

            z_hist = (
                z_aux.unsqueeze(1).expand(B, S, d, j).reshape(BS, d, j).clone()
            )
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
                    torch.cat([hist_te, tgt_te], dim=1)
                    .unsqueeze(1)
                    .expand(B, S, j + 1, self.emb_time_dim)
                    .reshape(BS, j + 1, self.emb_time_dim)
                )
                ctx_step: Dict[str, torch.Tensor] = {
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
        mu_t: torch.Tensor,           # (N, d)
        sigma2_t: torch.Tensor,       # (N, d)
        z_hist: torch.Tensor,         # (N, d, j)
        ctx: Dict[str, torch.Tensor],
        sigma_d2_per_row: torch.Tensor,  # (N,)
        padding_mask: torch.Tensor,   # (N, j+1)
        mc_override: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Centered ESM regression: ``E_τ E_q[w · ‖F_ψ − F*‖²]``.

        Returns the *summed-over-N* weighted squared error (caller
        normalises).  Centering: ``μ̂ = μ_t − μ_p(z_hist)``; the sampler
        draws ``ẑ_t^(τ) = μ̂ + √(σ_t² + σ̃²)·ε`` in centered coords.
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
                "DiffusionV3Transition requires hist_time_emb and target_time_emb in ctx"
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
        total_sqerr = torch.zeros(N, device=device, dtype=dtype)

        override_k_idx = None
        override_eps = None
        if mc_override is not None:
            override_k_idx = mc_override.get("k_idx")
            override_eps = mc_override.get("eps")

        remaining_k = int(self.S_k)
        k_cursor = 0
        while remaining_k > 0:
            kc = min(k_chunk, remaining_k)
            remaining_k -= kc

            if override_k_idx is not None:
                k_idx = override_k_idx[:, k_cursor : k_cursor + kc]
            else:
                k_idx = torch.multinomial(self.p_k, N * kc, replacement=True).view(
                    N, kc
                )
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
                latent_w.permute(0, 3, 1, 2)
                .reshape(N * kc, d, self.j + 1)
                .contiguous()
            )

            side_w = (
                side_win.unsqueeze(1)
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
                wtilde_base_flat
                * sd2_flat
                / (st2_flat + sd2_flat).clamp_min(eps_dtype)
            )
            # IS correction divides by K · p_k.
            weights = (
                wtilde_full
                / (self.num_steps * self.p_k[k_flat].clamp_min(eps_dtype))
            ).detach()

            F_pred = self.diffmodel(latent_w, side_w, c_noise_flat)  # (N*kc, d, 1)
            F_pred = F_pred.squeeze(-1)
            F_tgt_flat = F_target.permute(0, 2, 1).reshape(N * kc, d)

            sqerr = (F_pred - F_tgt_flat).pow(2).sum(dim=1) * weights
            total_sqerr = total_sqerr + sqerr.view(N, kc).sum(dim=1)

        per_sample = total_sqerr / float(self.S_k)
        return per_sample.sum()

    # ------------------------------------------------------------------
    # σ_data-aware EDM preconditioning.
    # ------------------------------------------------------------------
    def _vp_precondition(
        self,
        mu_hat_t: torch.Tensor,        # (N, d) — CENTERED mean
        sigma2_t: torch.Tensor,        # (N, d)
        k_idx: torch.Tensor,           # (N, S_k)
        eps: torch.Tensor,             # (N, d, S_k)
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
        sigma_tilde = self.sigma_tilde[k_idx]               # (N, S_k)
        sigma_tilde2 = sigma_tilde * sigma_tilde

        sd2 = sigma_d2_per_row.view(-1, 1).clamp_min(eps_dtype)  # (N, 1)
        sd = sd2.sqrt()
        denom = (sigma_tilde2 + sd2).clamp_min(eps_dtype)  # (N, S_k)
        sqrt_denom = denom.sqrt()

        c_skip = sd2 / denom                            # (N, S_k)
        c_out = (sigma_tilde * sd) / sqrt_denom         # (N, S_k)
        c_in = 1.0 / sqrt_denom                         # (N, S_k)

        # Broadcast to (N, 1, S_k) for the latent dim.
        st2_ = sigma_tilde2.unsqueeze(1)
        cskip_ = c_skip.unsqueeze(1)
        cout_ = c_out.unsqueeze(1).clamp_min(eps_dtype)
        cin_ = c_in.unsqueeze(1)

        sigma2_t_ = sigma2_t.unsqueeze(-1)              # (N, d, 1)
        mu_hat_t_ = mu_hat_t.unsqueeze(-1)              # (N, d, 1)

        var_total = (sigma2_t_ + st2_).clamp_min(eps_dtype)  # (N, d, S_k)
        z_hat = mu_hat_t_ + var_total.sqrt() * eps           # centered residual

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
        ctx: Dict[str, Any],
        sigma_d2: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
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
                "DiffusionV3Transition.score requires hist/target time embeddings in ctx"
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
        z_hist: torch.Tensor,           # (B, d, j)
        S: int = 1,
        ctx: Optional[Dict[str, Any]] = None,
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
            raise ValueError("DiffusionV3Transition.sample requires ctx")
        if "hist_time_emb" not in ctx or "target_time_emb" not in ctx:
            raise ValueError(
                "DiffusionV3Transition.sample requires hist/target time embeddings"
            )
        B, d, j_in = z_hist.shape
        if j_in != self.j:
            raise ValueError(f"Expected history j={self.j}, got {j_in}")

        device = z_hist.device
        dtype = z_hist.dtype

        # σ_data² lookup for this t.
        sigma_data: Optional[SigmaDataBuffer] = ctx.get("sigma_data")
        t_external = int(ctx.get("t", self.j + 1))
        if sigma_data is not None:
            sd2_scalar = float(sigma_data.read(t_external).item())
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
        z_hat_sample = self._vp_pf_sample_centered(
            z_hist=z_hist, side_win=side_win, sigma_d2=sd2_scalar,
        )
        z_sample = z_hat_sample + mu_p
        return z_sample.unsqueeze(1)  # (B, 1, d)

    @torch.no_grad()
    def _vp_pf_sample_centered(
        self,
        z_hist: torch.Tensor,    # (B, d, j)
        side_win: torch.Tensor,  # (B, side_dim, d, j+1)
        sigma_d2: float,
    ) -> torch.Tensor:
        """Reverse probability-flow Euler sampler in centered coords."""
        B, d, _ = z_hist.shape
        device = z_hist.device
        dtype = z_hist.dtype
        eps_dtype = torch.finfo(dtype).eps

        # Prior at τ=1 in centered coords is N(0, σ_d² · I) under VP
        # (the marginal variance at τ=1 is σ_data²-scaled).  Conservative
        # initialisation: sample N(0, max(σ_d², 1) · I) — keeps the V2
        # behaviour at σ_data ≡ 1.
        x = math.sqrt(max(sigma_d2, 1.0)) * torch.randn(B, d, device=device, dtype=dtype)

        K = self.num_steps
        for i in range(K - 1, 0, -1):
            alpha_i = float(self.alpha[i].item())
            sigma_tilde_i = float(self.sigma_tilde[i].item())
            sigma_tilde2_i = sigma_tilde_i * sigma_tilde_i
            denom = sigma_tilde2_i + sigma_d2
            sqrt_denom = math.sqrt(max(denom, eps_dtype))
            c_skip_i = sigma_d2 / max(denom, eps_dtype)
            c_out_i = (sigma_tilde_i * math.sqrt(sigma_d2)) / sqrt_denom
            c_in_i = 1.0 / sqrt_denom
            c_noise_i = float(self.c_noise[i].item())

            # Convert native x to VE-coords ẑ_tilde (drives the network).
            z_tilde = x / max(alpha_i, eps_dtype)
            z_in = (c_in_i * z_tilde).unsqueeze(2)  # (B, d, 1)
            latent_w = torch.cat([z_hist, z_in], dim=2)
            c_noise_vec = torch.full(
                (B,), c_noise_i, device=device, dtype=dtype
            )
            F_pred = self.diffmodel(latent_w, side_win, c_noise_vec).squeeze(-1)
            D_pred = c_skip_i * z_tilde + c_out_i * F_pred
            # Score in VE coords; native-coord score = s_tilde / α.
            s_tilde = (D_pred - z_tilde) / max(sigma_tilde2_i, eps_dtype)
            s_native = s_tilde / max(alpha_i, eps_dtype)

            beta_i = float(self.beta[i].item())
            d_tau = float((self.tau[i - 1] - self.tau[i]).item())  # < 0
            drift = -0.5 * beta_i * x - 0.5 * beta_i * s_native
            x = x + d_tau * drift

        return x


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _update_sigma_data_blocked(
    *,
    sigma_data: SigmaDataBuffer,
    mu_hat: torch.Tensor,       # (B*S*chunk_len, d) — (B, S, chunk_len) order
    sigma2_t: torch.Tensor,     # (B*S*chunk_len, d)
    B: int,
    S: int,
    chunk_len: int,
    d: int,
    t_start_external: int,
) -> None:
    """Reorganise (B, S, chunk_len) flattened input into (chunk_len, B*S) blocks."""
    mu_hat_b = (
        mu_hat.view(B, S, chunk_len, d)
        .permute(2, 0, 1, 3)
        .reshape(chunk_len * B * S, d)
    )
    sigma2_b = (
        sigma2_t.view(B, S, chunk_len, d)
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
