"""Stage-1 Gaussian transition for the model-v2 baseline-centering scheme.

Implements the stage-1 transition prior
``p_ψ^(stage-1)(z_t | z_{t-j:t-1}) = N(μ_p(z_{t-j:t-1}), diag(σ_p²(z_{t-j:t-1})))``
from ``model-v2.org`` § Assembled losses for stage 1 / § State-conditional prior
variance.  Functionally a refactoring of :class:`ddssm.model.transitions.transitions.GaussianTransition`
that delegates the ``(μ_p, log σ_p²)`` heads to a shared :class:`ddssm.model.centering.baselines.BaseBaseline`
module — letting the stage-2 :class:`ddssm.model.transitions.diffusion.DiffusionTransition`
read the same μ_p instance for the centering shift.

Provides two entry points:

* :meth:`transition_kl` for ``t = j+1 … T`` (closed-form Gaussian KL,
  chunked exactly like ``GaussianTransition.transition_kl``'s
  closed-form path).
* :meth:`transition_kl_init` for the VHP-via-diffusion ``t = 1 … j``
  loop: samples auxiliary latents from ``q_Φ`` and computes the
  cross-entropy MC estimate ``-log p_ψ(z_t | z_hist_t)`` against the
  encoder sample, with the history mixing from all-aux at t=1 to one-
  aux-plus-(j-1)-real at t=j (the same hierarchical walk the
  now-removed legacy InitPrior performed; see ADR-0006).

Both entry points update the shared :class:`~ddssm.model.centering.sigma_data.SigmaDataBuffer`
when one is supplied — passive accumulation per ``model-v2.org``
§ Stage-1 → stage-2 handoff step 1.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ddssm.nn.gaussians import GaussianStats, gaussian_kl_divergence
from ddssm.model.centering.baselines import BaseBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.transitions import BaseTransition


class BaselineGaussianTransition(BaseTransition):
    """``p_ψ(z_t | z_hist) = N(μ_p(z_hist), diag(σ_p²(z_hist)))`` (stage 1).

    Args:
        baseline: Shared :class:`BaseBaseline` instance.  The *same*
            instance is also passed to the stage-2
            :class:`DiffusionTransition` so μ_p's parameters are
            shared across the two stages.
        latent_dim: Latent dimension ``d``.
        j: History length.
        emb_time_dim: Reserved for API compatibility with
            :class:`BaseTransition`; the closed-form Gaussian KL in
            this transition does not consume time embeddings (μ_p and
            σ_p are functions of ``z_hist`` only per the doc's
            construction).
        covariate_dim: Reserved for API compatibility (ditto).
    """

    def __init__(
        self,
        baseline: BaseBaseline,
        latent_dim: int,
        j: int,
        emb_time_dim: int = 0,
        covariate_dim: int = 0,
    ) -> None:
        super().__init__()
        if int(baseline.latent_dim) != int(latent_dim):
            raise ValueError(
                f"baseline.latent_dim={baseline.latent_dim} != latent_dim={latent_dim}"
            )
        if int(baseline.j) != int(j):
            raise ValueError(f"baseline.j={baseline.j} != j={j}")
        self.baseline = baseline
        self.latent_dim = int(latent_dim)
        self.j = int(j)
        self.emb_time_dim = int(emb_time_dim)
        self.covariate_dim = int(covariate_dim)

    # ------------------------------------------------------------------
    # prior_params / sample / log_prob delegate to the baseline.
    # ------------------------------------------------------------------
    def prior_params(
        self,
        z_hist: torch.Tensor,
        ctx: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(μ_p, log σ_p²)`` from :meth:`BaseBaseline.mean_and_logvar`.

        ``ctx`` is accepted for API compatibility but ignored — μ_p / σ_p
        are functions of ``z_hist`` only per ``model-v2.org`` § State-
        conditional prior variance.
        """
        del ctx
        return self.baseline.mean_and_logvar(z_hist)

    def sample(
        self,
        z_hist: torch.Tensor,
        S: int = 1,
        ctx: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Draw ``z_t = μ_p + σ_p · ε``, ``ε ∼ N(0, I)``.

        Returns ``(B, S, d)``.
        """
        del ctx
        mu, logvar = self.prior_params(z_hist)
        sigma = (0.5 * logvar).exp()
        B, d = mu.shape
        eps = torch.randn(B, S, d, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(1) + sigma.unsqueeze(1) * eps

    def log_prob(
        self,
        z: torch.Tensor,
        z_hist: torch.Tensor,
        ctx: dict[str, Any] | None = None,
        mc_override: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Closed-form Gaussian log-density of ``z`` under the baseline prior.

        Supports ``z`` shaped ``(B, d)`` or ``(B, S, d)``; returns
        respectively ``(B,)`` or ``(B, S)``.
        """
        del ctx, mc_override
        B = z_hist.shape[0]
        p_mu, p_logvar = self.prior_params(z_hist)
        assert p_mu.shape == p_logvar.shape == (B, self.latent_dim)
        return _gaussian_log_prob(z, p_mu, p_logvar, latent_dim=self.latent_dim)

    # ------------------------------------------------------------------
    # transition_kl for t = j+1 … T  (closed-form Gaussian KL)
    # ------------------------------------------------------------------
    def transition_kl(
        self,
        enc_stats: GaussianStats,
        zs: torch.Tensor,  # (B, S, d, T)
        logq_paths: torch.Tensor,  # (B, S, T)  unused on the closed-form path
        time_embed: torch.Tensor,  # (B, T, E_t)
        sigma_data: SigmaDataBuffer | None = None,
        covariates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Closed-form Gaussian KL plus passive σ_data update.

        For every chunked target ``t = j+1 … T`` (1-based; ``t_start =
        j`` to ``t_end ≤ T`` in 0-based code indexing):

          1. ``p_mu, p_logvar = baseline.mean_and_logvar(z_hist)``.
          2. Closed-form ``KL(q || p)`` via :func:`gaussian_kl_divergence`.
          3. If ``sigma_data`` is supplied, compute centered moments
             ``μ̂_t = μ_q − μ_p`` and update the buffer at the chunk's
             external 1-based t indices.

        Returns ``{"kl": ..., "log_sigma_p2_mean": ...}``.  The
        log-σ_p² diagnostic is reported every step so the global
        regularizer :func:`r_sigma_p_loss` can be cross-checked.
        """
        if (
            "mus" not in enc_stats
            or "logvars" not in enc_stats
            or enc_stats["mus"] is None
            or enc_stats["logvars"] is None
        ):
            raise ValueError(
                "BaselineGaussianTransition.transition_kl requires the "
                "encoder to expose Gaussian (mus, logvars) — the legacy "
                "MC-fallback path is intentionally not supported here."
            )

        B, S, d, T = zs.shape
        j = self.j
        device, dtype = zs.device, zs.dtype
        if d != self.latent_dim:
            raise ValueError(f"zs latent dim {d} != self.latent_dim {self.latent_dim}")

        mus = enc_stats["mus"]
        logvars = enc_stats["logvars"]
        assert mus.shape == logvars.shape == (B, S, d, T)

        total_kl = torch.zeros(B, device=device, dtype=dtype)
        total_logvar_p_sum = torch.zeros((), device=device, dtype=dtype)
        total_logvar_p_count = 0
        total_steps = T - j

        if total_steps > 0:
            for (
                B_,
                S_,
                chunk_len,
                t_start,
                t_end,
                _z_target_flat,
                z_hist_flat,
                _ctx,
            ) in self._iter_window_chunks(
                zs,
                time_embed,
                covariates=covariates,
            ):
                # Shape: (N, d) where N = B*S*chunk_len
                p_mu, p_logvar = self.baseline.mean_and_logvar(z_hist_flat)

                # Slice encoder stats for the chunk's targets.
                q_mu = mus[..., t_start:t_end].permute(0, 1, 3, 2).reshape(-1, d)
                q_logvar = (
                    logvars[..., t_start:t_end].permute(0, 1, 3, 2).reshape(-1, d)
                )

                kl_flat = gaussian_kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
                kl = kl_flat.view(B_, S_, chunk_len)
                total_kl = total_kl + kl.sum(dim=2).mean(dim=1)

                # Track logvar_p diagnostic.
                total_logvar_p_sum = total_logvar_p_sum + p_logvar.sum()
                total_logvar_p_count += p_logvar.numel()

                # σ_data buffer update (per-chunk, blocked by t).
                if sigma_data is not None:
                    _update_sigma_data_from_chunk(
                        sigma_data=sigma_data,
                        q_mu=q_mu,
                        q_logvar=q_logvar,
                        p_mu=p_mu,
                        B=B_,
                        S=S_,
                        chunk_len=chunk_len,
                        d=d,
                        t_start_external=t_start + 1,  # 1-based per the doc
                    )

        kl_scalar = total_kl.mean()
        log_sigma_p2_mean = (
            total_logvar_p_sum / float(total_logvar_p_count)
            if total_logvar_p_count > 0
            else torch.zeros((), device=device, dtype=dtype)
        )

        return {"kl": kl_scalar, "log_sigma_p2_mean": log_sigma_p2_mean.detach()}

    # ------------------------------------------------------------------
    # transition_kl_init for t = 1 … j  (VHP-via-diffusion init term)
    # ------------------------------------------------------------------
    def _score_init_step(
        self,
        *,
        step: int,
        z_t: torch.Tensor,  # (BS, d)
        z_hist: torch.Tensor,  # (BS, d, j)
        enc_stats: GaussianStats,
        time_embed: torch.Tensor,  # unused — baseline doesn't condition on time
        sigma_data: SigmaDataBuffer | None,
        B: int,
        S: int,
        T: int,
    ) -> torch.Tensor:
        """Closed-form cross-entropy ``-log p_ψ(z_t | z_hist)`` (summed over B·S).

        Per § State-conditional prior variance the σ_p head is shared with
        μ_p; the log-density uses both heads. Also updates ``sigma_data``
        at this init ``t`` (1-based) from the centered encoder moments. The
        shared :meth:`BaseTransition._init_entropy_term` default adds
        ``-H(q_φ)`` so the assembled init term is the full stage-1 loss.
        """
        del time_embed  # baseline does not condition on time
        BS = B * S
        d = self.latent_dim
        p_mu, p_logvar = self.baseline.mean_and_logvar(z_hist)
        log_p = _gaussian_log_prob_flat(z_t, p_mu, p_logvar)  # (BS,)

        if sigma_data is not None:
            q_mu = enc_stats["mus"][:, :, :, step].reshape(BS, d)
            q_logvar = enc_stats["logvars"][:, :, :, step].reshape(BS, d)
            mu_hat = q_mu - p_mu
            sigma_data.update(
                t_idx=torch.tensor([step + 1], device=z_t.device),
                mu_hat_batch=mu_hat,
                sigma_t2_batch=q_logvar.exp(),
            )

        return (-log_p).sum()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_log_prob_flat(
    z: torch.Tensor,  # (N, d)
    mu: torch.Tensor,  # (N, d)
    logvar: torch.Tensor,  # (N, d)
) -> torch.Tensor:
    """Closed-form Gaussian log-density summed over the ``d`` axis."""
    var = logvar.exp()
    diff = z - mu
    per_dim = -0.5 * (diff * diff / var + logvar + math.log(2 * math.pi))
    return per_dim.sum(dim=-1)


def _gaussian_log_prob(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    latent_dim: int,
) -> torch.Tensor:
    """Dispatch over ``z`` shape ``(B, d)`` or ``(B, S, d)``."""
    B = mu.shape[0]
    if z.dim() == 2:
        return _gaussian_log_prob_flat(z, mu, logvar)
    if z.dim() == 3:
        S = z.shape[1]
        mu_exp = mu.unsqueeze(1).expand(B, S, latent_dim)
        lv_exp = logvar.unsqueeze(1).expand(B, S, latent_dim)
        per_dim = -0.5 * (
            (z - mu_exp).pow(2) / lv_exp.exp() + lv_exp + math.log(2 * math.pi)
        )
        return per_dim.sum(dim=-1)  # (B, S)
    raise ValueError(f"z must be (B,d) or (B,S,d); got shape {tuple(z.shape)}")


def _update_sigma_data_from_chunk(
    *,
    sigma_data: SigmaDataBuffer,
    q_mu: torch.Tensor,  # (N=B*S*chunk_len, d)
    q_logvar: torch.Tensor,  # (N, d)
    p_mu: torch.Tensor,  # (N, d)
    B: int,
    S: int,
    chunk_len: int,
    d: int,
    t_start_external: int,
) -> None:
    """Reorganise ``(N, d)`` into ``(chunk_len * B*S, d)`` blocked by t.

    The chunked layout in :meth:`_iter_window_chunks` is
    ``(B, S, chunk_len, d)`` flattened — i.e. ``chunk_len`` varies as
    the *innermost* of the leading dims.  :meth:`SigmaDataBuffer.update`
    expects ``(chunk_len, B*S, d)``-blocked input where chunk_len varies
    *slowest*, so we permute.
    """
    mu_hat = q_mu - p_mu
    sigma2 = q_logvar.exp()
    # (B, S, chunk_len, d) -> (chunk_len, B, S, d) -> (chunk_len * B * S, d)
    mu_hat = (
        mu_hat
        .view(B, S, chunk_len, d)
        .permute(2, 0, 1, 3)
        .reshape(chunk_len * B * S, d)
    )
    sigma2 = (
        sigma2
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
    sigma_data.update(t_idx=t_idx, mu_hat_batch=mu_hat, sigma_t2_batch=sigma2)
