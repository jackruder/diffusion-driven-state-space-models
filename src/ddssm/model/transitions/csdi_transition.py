"""Transition that wraps the *literal* ermongroup CSDI as the DDSSM prior.

Motivation: every other DDSSM transition (Gaussian, our CSDI-style
``DiffusionTransition``) plateaus at ~25 % MARG→FLOOR on the p=0.85 nlblmv
head-to-head while the standalone ermongroup CSDI hits 58 %. The identity-encoder
test (obs-space, *our* ``DiffusionTransition``) also stalls at ~22 %, exonerating
the learned latent frame. That leaves the transition itself — but our diffusion
code differs from CSDI in *both* objective (centered entropy-cancelled ESM/EDM
surrogate + adaptive-IS noise sampling) and conditioning (no masked-input stack).

This module removes that ambiguity by dropping the **verbatim** CSDI denoiser
(:mod:`._csdi_vendor`) into the transition slot: the DDPM ε-MSE loss
(``calc_loss``), the ancestral sampler (``impute``), and the masked-conditioning
side-info builder (``get_side_info``) are called directly. Paired with the
identity encoder/decoder and ``j == HIST`` this reproduces the standalone CSDI
forecaster *inside* the DDSSM ELBO pipeline. A pass (≈58 %) indicts our
transition code; a stall (≈22 %) indicts the surrounding pipeline (forecast
rollout / ELBO coupling / data plumbing), since the frame is already cleared.

Mapping CSDI's (B, K, L) conditional-imputation API onto the DDSSM transition:

* **K = latent_dim** (features), **L = j + 1** (a history window plus the
  one-step-ahead target). Each ``(sample, time-origin)`` pair is its own CSDI
  "row", so ``calc_loss`` draws an independent diffusion step per window.
* **cond_mask** = 1 on the first ``j`` columns (the history is observed) and 0 on
  the last (the target is imputed) — exactly CSDI-Forecasting's HIST/PRED split.
* The CSDI loss (an averaged ε-MSE) is returned as the ELBO ``"kl"`` (the rate
  ``L_trans``); minimizing it *is* training the CSDI denoiser. ``transition_kl_init``
  returns zeros — CSDI models only the conditional, not the j-step init marginal
  (which forecasting takes from data anyway).
* Per-feature standardization (CSDI's data recipe) is reproduced as a one-shot
  lazy calibration over the first batch's latents, frozen into buffers so it
  survives checkpoint reload. With the identity encoder z≈x, so this matches the
  standalone baseline; if upstream already standardized, it is ~a no-op.
"""

from __future__ import annotations

from typing import Any

import torch

from ddssm.model.transitions.transitions import BaseTransition
from ddssm.model.transitions._csdi_vendor.main_model import CSDI_Forecasting


class CSDITransition(BaseTransition):
    """Literal ermongroup CSDI in the DDSSM stage-2 transition slot."""

    def __init__(
        self,
        *,
        latent_dim: int,
        j: int,
        emb_time_dim: int = 0,
        T_max: int = 192,
        channels: int = 64,
        layers: int = 4,
        nheads: int = 8,
        diffusion_embedding_dim: int = 128,
        num_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.5,
        schedule: str = "quad",
        timeemb: int = 128,
        featureemb: int = 16,
        standardize: bool = True,
    ) -> None:
        super().__init__()
        # BaseTransition window helpers read these (unused here — we override the
        # public methods — but set for interface parity / diagnostics).
        self.j = int(j)
        self.latent_dim = int(latent_dim)
        self.emb_time_dim = int(emb_time_dim)
        self.standardize = bool(standardize)

        config = {
            "diffusion": {
                "layers": int(layers),
                "channels": int(channels),
                "nheads": int(nheads),
                "diffusion_embedding_dim": int(diffusion_embedding_dim),
                "beta_start": float(beta_start),
                "beta_end": float(beta_end),
                "num_steps": int(num_steps),
                "schedule": str(schedule),
                "is_linear": False,
            },
            "model": {
                "is_unconditional": 0,  # conditional (masked-input) CSDI
                "timeemb": int(timeemb),
                "featureemb": int(featureemb),
                "target_strategy": "test",
                # == target_dim ⇒ sample_features never fires; we use the full K.
                "num_sample_features": int(latent_dim),
            },
        }
        # Built on CPU; alpha_torch / self.device are repaired at call time (see
        # _ensure_device) because CSDI_base stores them as a plain attr + string,
        # which nn.Module.to() does not move.
        self.csdi = CSDI_Forecasting(config, torch.device("cpu"), target_dim=latent_dim)

        # Per-feature standardization stats (CSDI's data recipe). Persistent so
        # eval/probe in a fresh process restores the exact frozen scale.
        self.register_buffer("feat_mean", torch.zeros(latent_dim))
        self.register_buffer("feat_std", torch.ones(latent_dim))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))

    # --- helpers ----------------------------------------------------------
    def _ensure_device(self, ref: torch.Tensor) -> None:
        """Point the vendored CSDI at ``ref``'s device (alpha_torch is not a buffer)."""
        dev = ref.device
        self.csdi.device = dev
        if self.csdi.alpha_torch.device != dev:
            self.csdi.alpha_torch = self.csdi.alpha_torch.to(dev)

    def _calibrate(self, zs: torch.Tensor) -> None:
        """One-shot per-feature mean/std over (B, S, T) — frozen into buffers."""
        if not self.standardize or bool(self.calibrated):
            return
        # zs: (B, S, d, T) -> per-feature over everything but d. In-place copy_ so
        # the buffer keeps its identity (a stable EMA shadow / state_dict slot).
        z = (
            zs.detach().permute(0, 1, 3, 2).reshape(-1, zs.shape[2]).float()
        )  # (B*S*T, d)
        self.feat_mean.copy_(z.mean(dim=0))
        self.feat_std.copy_(z.std(dim=0) + 1e-6)
        self.calibrated.fill_(True)

    def _standardize(self, w: torch.Tensor) -> torch.Tensor:
        """(N, d, L) -> standardized per feature d."""
        if not self.standardize:
            return w
        m = self.feat_mean.to(w.dtype)[None, :, None]
        s = self.feat_std.to(w.dtype)[None, :, None]
        return (w - m) / s

    def _destandardize(self, w: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return w
        m = self.feat_mean.to(w.dtype)
        s = self.feat_std.to(w.dtype)
        return w * s + m

    # --- ELBO rate (training) --------------------------------------------
    def transition_kl(
        self,
        enc_stats: Any,
        zs: torch.Tensor,  # (B, S, d, T)
        logq_paths: torch.Tensor,  # (B, S, T) — unused (CSDI cancels entropy)
        time_embed: torch.Tensor,  # (B, T, E_t) — unused (CSDI builds its own pos-emb)
        sigma_data: Any = None,
        covariates: torch.Tensor | None = None,
        mc_override: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        del enc_stats, logq_paths, time_embed, sigma_data, covariates, mc_override
        B, S, d, T = zs.shape
        j = self.j
        if T - j <= 0:
            return {"kl": zs.new_zeros(())}

        self._ensure_device(zs)
        self._calibrate(zs)

        L = j + 1
        # All (B*S) trajectories, all time-origins -> one CSDI row per window.
        windows = zs.unfold(dimension=-1, size=L, step=1)  # (B, S, d, T-j, L)
        windows = windows.permute(0, 1, 3, 2, 4).reshape(-1, d, L)  # (N, d, L)
        observed_data = self._standardize(windows)

        N = observed_data.shape[0]
        cond_mask = torch.ones(N, d, L, device=zs.device, dtype=observed_data.dtype)
        cond_mask[..., -1] = 0.0  # last column = one-step-ahead target (imputed)
        observed_mask = torch.ones_like(cond_mask)
        observed_tp = (
            torch
            .arange(L, device=zs.device, dtype=observed_data.dtype)
            .unsqueeze(0)
            .expand(N, -1)
        )

        side_info = self.csdi.get_side_info(observed_tp, cond_mask)
        loss = self.csdi.calc_loss(
            observed_data, cond_mask, observed_mask, side_info, is_train=1
        )
        return {"kl": loss}

    def transition_kl_init(
        self,
        enc_stats: Any,
        zs: torch.Tensor,
        aux_posterior: Any = None,
        time_embed: torch.Tensor | None = None,
        sigma_data: Any = None,
        covariates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # CSDI models only the conditional p(z_t | z_{t-j:t-1}); the j-step init
        # marginal is supplied from data at forecast time. Contribute nothing.
        del enc_stats, aux_posterior, time_embed, sigma_data, covariates
        z = zs.new_zeros(())
        return {"loss": z, "entropy": z, "vhp": z, "kl_aux": z, "loss_init": z}

    # --- forecast rollout -------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        z_hist: torch.Tensor,  # (BS, d, j)
        S: int = 1,
        ctx: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del S, ctx  # CSDI conditions on the window only; positional time is internal.
        self._ensure_device(z_hist)
        BS, d, j = z_hist.shape
        L = j + 1

        hist_std = self._standardize(z_hist)  # (BS, d, j)
        pad = hist_std.new_zeros(BS, d, 1)
        observed_data = torch.cat([hist_std, pad], dim=-1)  # (BS, d, L)
        cond_mask = torch.ones(
            BS, d, L, device=z_hist.device, dtype=observed_data.dtype
        )
        cond_mask[..., -1] = 0.0
        observed_tp = (
            torch
            .arange(L, device=z_hist.device, dtype=observed_data.dtype)
            .unsqueeze(0)
            .expand(BS, -1)
        )

        side_info = self.csdi.get_side_info(observed_tp, cond_mask)
        samples = self.csdi.impute(observed_data, cond_mask, side_info, n_samples=1)
        z_pred = samples[:, 0, :, -1]  # (BS, d) standardized target column
        z_pred = self._destandardize(z_pred)
        return z_pred.unsqueeze(1)  # (BS, 1, d)
