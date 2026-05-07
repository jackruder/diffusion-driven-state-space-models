"""VP-SDE diffusion transition (V2) with explicit-score-matching objective.

Implements the model described in ``model-v2.org``.  Compared to V1
(:class:`ddssm.transitions.diffusion.DiffusionTransition`), V2 differs in:

* **Schedule.** A discretised VP-SDE on ``tau in [tau_min, 1]``::

      beta(tau) = beta_min + (beta_max - beta_min) * tau
      alpha(tau) = exp(-1/2 * (beta_min*tau + 1/2*(beta_max-beta_min)*tau**2))
      sigma(tau)**2 = 1 - alpha(tau)**2
      sigma_tilde(tau) = sigma(tau)/alpha(tau)
      g(tau)**2 = beta(tau)
      w(tau) = beta(tau) * alpha(tau)**2 / (1 - alpha(tau)**2)

  EDM-style preconditioning constants follow with ``sigma_data = 1``::

      c_skip = alpha**2,  c_out = sqrt(1 - alpha**2),
      c_in   = alpha,     c_noise = (1/4) * log(sigma_tilde).

* **Objective (Explicit Score Matching).** The closed-form marginal score
  ``s_q`` is computed analytically from the encoder statistics
  ``(mu_t, sigma_t**2)`` at the *target* step.  We sample ``z_tilde`` from the
  marginal ``N(mu_t, sigma_t**2 + sigma_tilde**2)``; ``z_t`` itself is
  integrated out (Gaussian convolution).  The regression target follows
  EDM with the closed-form score::

      s_q_tilde   = -(z_tilde - mu_t) / (sigma_t**2 + sigma_tilde**2)
      D_star      = z_tilde + sigma_tilde**2 * s_q_tilde
      F_star      = (D_star - c_skip * z_tilde) / c_out
      loss        = (1/2) * dtau * w(tau) / p(tau)
                    * || F_psi(c_in*z_tilde, c_noise, z_hist) - F_star ||**2

  where ``dtau = (1 - tau_min) / K`` is the Riemann measure for the
  uniform tau grid, and the leading ``1/2`` comes from the KL definition
  ``KL = (1/2) * E_tau E_q[ g**2(tau) * ||s_q - s_p||**2 ]``.  Both
  factors are baked into ``wtilde`` so that callers obtain a correctly
  scaled per-step KL in nats (matching the dict contract of the V1
  transition)..

  When the encoder does not expose ``mus`` / ``logvars`` we fall back to the
  degenerate case ``mu_t = z_t, sigma_t**2 = 0``, which recovers the standard
  DSM target around the sampled latent.

* **Log-likelihood / boundary KL.** Not implemented in V2; the relevant
  ``log_likelihood`` / ``forward_kl_loss`` / ``log_prob`` methods raise
  :class:`NotImplementedError`.  The ESM derivation in ``model-v2.org`` makes
  the V1-style ``L_K`` boundary term not directly applicable; deferred to a
  follow-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, final

import torch
import torch.nn as nn

from hydra_zen import builds

from ..diffnets import CSDIUnet, CSDIUnetConf
from ..gaussians import GaussianStats, gaussian_entropy
from ..net_utils import get_side_info
from ..torch_compile import maybe_compile
from .transitions import BaseTransition, _mc_entropy_from_logq


@dataclass
class DiffusionV2ScheduleConfig:
    """VP-SDE diffusion schedule configuration for ``DiffusionV2Transition``."""

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
class DiffusionV2Transition(BaseTransition):
    """VP-SDE diffusion transition p(z_t | z_{t-j:t-1}) trained with ESM.

    Wraps :class:`CSDIUnet` and overrides :meth:`transition_kl` directly so it
    can pull encoder ``(mus, logvars)`` for the closed-form ESM target — the
    inherited :meth:`seq_log_prob` / :meth:`log_prob` chain only forwards the
    sampled ``z_t`` and is bypassed.
    """

    def __init__(
        self,
        latent_dim: int,
        j: int,
        emb_time_dim: int,
        covariate_dim: int = 0,
        unet: Callable[..., CSDIUnet] | None = None,
        schedule: DiffusionV2ScheduleConfig | None = None,
    ) -> None:
        super().__init__()

        if unet is None:
            unet = partial(CSDIUnet, channels=64, n_layers=4, embedding_dim=128)
        if schedule is None:
            schedule = DiffusionV2ScheduleConfig()

        self.j = j
        self.latent_dim = latent_dim
        self.emb_time_dim = emb_time_dim
        self.covariate_dim = covariate_dim

        # Feature embedding dimension mirrors the time embedding dimension.
        self.emb_feature_dim = emb_time_dim
        self.side_dim = (
            self.emb_time_dim + self.covariate_dim + self.emb_feature_dim + 1
        )

        self.diffmodel = unet(
            output_len=1,
            diffusion_steps=schedule.num_steps,
            latent_dim=self.latent_dim,
            latent_history_len=self.j,
            side_dim=self.side_dim,
        )
        self.diffmodel = maybe_compile(self.diffmodel)

        self.embed_layer = nn.Embedding(
            num_embeddings=self.latent_dim, embedding_dim=self.emb_feature_dim
        )

        self.S_k = schedule.S_k
        self.num_steps = schedule.num_steps

        # ----- VP-SDE precompute on a uniform tau-grid in [tau_min, 1] -----
        dtype64 = torch.float64
        eps64 = torch.finfo(dtype64).eps
        K = self.num_steps

        beta_min = float(schedule.beta_min)
        beta_max = float(schedule.beta_max)
        tau_min = float(schedule.tau_min)

        if not (0.0 < tau_min < 1.0):
            raise ValueError(f"tau_min must be in (0, 1); got {tau_min}")
        if beta_max <= beta_min:
            raise ValueError(
                f"beta_max ({beta_max}) must be > beta_min ({beta_min})"
            )

        tau = torch.linspace(tau_min, 1.0, K, dtype=dtype64)
        beta = beta_min + (beta_max - beta_min) * tau
        # ∫_0^tau beta(s) ds = beta_min*tau + 0.5*(beta_max-beta_min)*tau**2
        int_beta = beta_min * tau + 0.5 * (beta_max - beta_min) * tau * tau
        alpha = torch.exp(-0.5 * int_beta)
        alpha2 = alpha * alpha
        one_minus_alpha2 = (1.0 - alpha2).clamp_min(eps64)
        sigma2 = one_minus_alpha2  # noise variance under VP forward kernel
        sigma_tilde = torch.sqrt(sigma2 / alpha2.clamp_min(eps64))

        # EDM constants (sigma_data = 1)
        c_skip = alpha2
        c_out = torch.sqrt(one_minus_alpha2)
        c_in = alpha
        c_noise = 0.25 * torch.log(sigma_tilde.clamp_min(eps64))

        # Per-tau ESM weight w(tau) = beta(tau) * alpha**2 / (1 - alpha**2)
        # (model-v2.org L506-509).  The per-step KL is the *integral*
        #     L_t = (1/2) * E_{tau ~ Unif[tau_min, 1]} E_q[ w(tau) * ||F - F*||**2 ]
        # so an unbiased single-MC-sample estimator with k ~ p_k requires
        # the per-grid weight  (1/2) * dtau * w(tau_k) / p_k[k], where
        # dtau = (1 - tau_min) / K is the Riemann measure that matches the
        # uniform PMF on K grid points.  We bake (1/2) * dtau into ``wtilde``
        # so callers get a correctly scaled KL in nats; ``w_per_tau`` is kept
        # available as a separate buffer for diagnostics / sampling.
        w_per_tau = beta * alpha2 / one_minus_alpha2
        dtau = (1.0 - tau_min) / float(K)
        wtilde = 0.5 * dtau * w_per_tau

        # Derivative d(sigma_tilde**2)/dtau, kept around for diagnostics
        # and as a reference for any future importance-sampling schedule
        # (the previous LSGM-style schedule that used this directly was
        # incorrect; see the ``k_sampling_mode == "importance"`` branch).
        #   sigma_tilde**2 = (1 - alpha**2) / alpha**2 = 1/alpha**2 - 1
        #   d(sigma_tilde**2)/dtau = beta / alpha**2
        dsigma2_tilde_dtau = beta / alpha2.clamp_min(eps64)

        self.register_buffer("alpha", alpha.to(torch.float32))
        self.register_buffer("sigma_tilde", sigma_tilde.to(torch.float32))
        self.register_buffer("w_per_tau", w_per_tau.to(torch.float32))
        self.register_buffer("wtilde", wtilde.to(torch.float32))
        self.register_buffer(
            "dsigma2_tilde_dtau", dsigma2_tilde_dtau.to(torch.float32)
        )
        self.register_buffer("c_skip", c_skip.to(torch.float32))
        self.register_buffer("c_out", c_out.to(torch.float32))
        self.register_buffer("c_in", c_in.to(torch.float32))
        self.register_buffer("c_noise", c_noise.to(torch.float32))
        self.register_buffer("beta", beta.to(torch.float32))
        self.register_buffer("tau", tau.to(torch.float32))

        # Sampling probabilities p_k for the tau index.
        #
        # ``"uniform"``     : p_k = 1/K (no IS).
        # ``"importance"``  : currently unsupported in V2.  The previous
        #                     LSGM-style derivation (p_k ∝ d(sigma_tilde**2)/dtau)
        #                     was incorrect, so we raise NotImplementedError
        #                     until the IS schedule is re-derived.  Use
        #                     "uniform" for now.
        self.gamma = float(schedule.pk_gamma)
        self.gfloor = float(schedule.pk_floor)
        ismode = schedule.k_sampling_mode
        if ismode == "importance":
            raise NotImplementedError(
                "DiffusionV2Transition: importance sampling is currently "
                "disabled because the previous LSGM-style schedule "
                "(p_k ∝ d(sigma_tilde**2)/dtau) was incorrectly derived. "
                "Use k_sampling_mode='uniform' until a corrected importance "
                "schedule is implemented."
            )
        elif ismode == "uniform":
            p_k = torch.full(
                (self.num_steps,),
                1.0 / self.num_steps,
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"Unknown k_sampling_mode={ismode!r}; expected 'uniform'"
            )
        self.register_buffer("p_k", p_k)
        self.k_sampling_mode = ismode
        self.schedule = schedule

    # ------------------------------------------------------------------ #
    # Disabled paths.  V2 bypasses log_prob / seq_log_prob entirely
    # because the ESM target needs encoder (mu, sigma**2), not just z_t.
    # ------------------------------------------------------------------ #
    def log_prob(
        self,
        z: torch.Tensor,
        z_hist: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "DiffusionV2Transition does not implement log_prob; "
            "transition_kl is overridden to use closed-form ESM directly."
        )

    def forward_kl_loss(self, z0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "DiffusionV2Transition does not implement the boundary KL term; "
            "see model-v2.org for the deferred derivation."
        )

    def log_likelihood(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError(
            "DiffusionV2Transition does not implement log_likelihood; "
            "see model-v2.org for the deferred derivation."
        )

    # ------------------------------------------------------------------ #
    # ESM transition KL — overrides BaseTransition default.
    # ------------------------------------------------------------------ #
    def transition_kl(
        self,
        enc_stats: GaussianStats,
        zs: torch.Tensor,  # (B, S, d, T)
        logq_paths: torch.Tensor,  # (B, S, T)
        time_embed: torch.Tensor,  # (B, T, E_t)
        covariates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Closed-form ESM regression loss IS the transition KL.

        Per the V2 derivation (see ``model-v2.org`` "ELBO modifications"
        / "Using Song Thm. 2"), the cross entropy decomposes as
        ``CE(q‖p) = H(q) + E_τ E_q[½ g²(τ) ‖s_q − s_p‖²]``, so the KL
        equals the ESM regression term directly — the encoder entropy
        is *not* subtracted from it.

        ``L_q`` is the closed-form Gaussian entropy when ``logvars`` is
        available, otherwise an MC estimate from ``logq_paths``; it is
        retained for logging only.  ``L_p`` (cross-entropy, also for
        logging) is recovered as ``kl + L_q`` so that the V1 invariant
        ``kl = L_p − L_q`` continues to hold across the dict contract
        ``{"kl", "L_p", "L_q"}``.
        """
        B, S, d, T = zs.shape
        j = self.j
        device = zs.device
        dtype = zs.dtype

        mus = enc_stats.get("mus") if enc_stats is not None else None
        logvars = enc_stats.get("logvars") if enc_stats is not None else None
        have_stats = mus is not None and logvars is not None

        # ----- KL : ESM loss summed over chunks ----- #
        # Per the V2 derivation, the closed-form ESM regression term
        # IS the transition KL (the encoder-entropy cancellation is
        # already baked into the CE decomposition).  We accumulate it
        # under the name ``kl_sum`` to make the semantics explicit.
        kl_sum = torch.zeros((), device=device, dtype=dtype)
        n_target_steps = max(0, T - j)

        if n_target_steps > 0:
            for (
                B_,
                S_,
                chunk_len,
                t_start,
                t_end,
                z_target_flat,  # (BS*chunk_len, d) — used only as DSM fallback
                z_hist_flat,    # (BS*chunk_len, d, j)
                ctx,
            ) in self._iter_window_chunks(
                zs, time_embed, covariates=covariates,
            ):
                if have_stats:
                    # mus, logvars: (B, S, d, T) -> (B, S, d, chunk_len)
                    mu_chunk = mus[..., t_start:t_end]
                    lv_chunk = logvars[..., t_start:t_end]
                    # match z_target_flat ordering: (B, S, chunk_len, d)
                    mu_t_flat = mu_chunk.permute(0, 1, 3, 2).reshape(-1, d)
                    sigma2_t_flat = (
                        lv_chunk.exp().permute(0, 1, 3, 2).reshape(-1, d)
                    )
                else:
                    # Encoder did not expose Gaussian stats: fall back to a
                    # degenerate posterior with mu_t = z_t and sigma_t**2 = 0,
                    # which collapses the ESM target to standard DSM around z_t.
                    mu_t_flat = z_target_flat
                    sigma2_t_flat = torch.zeros_like(z_target_flat)

                kl_sum = kl_sum + self._esm_chunk_loss(
                    mu_t=mu_t_flat,
                    sigma2_t=sigma2_t_flat,
                    z_hist=z_hist_flat,
                    ctx=ctx,
                )

        # mean over (B, S, n_target_steps): _esm_chunk_loss returns the
        # weighted-sum-of-squared-errors over the chunk's BS*chunk_len rows
        # (already averaged over S_k internally), so dividing by B*S*(T-j)
        # yields the per-(b, s, t) average matching V1's `.mean()` convention.
        denom = float(B * S * n_target_steps) if n_target_steps > 0 else 1.0
        kl = kl_sum / denom

        # ----- L_q : encoder entropy over t = j..T-1 ----- #
        if have_stats:
            if j >= T:
                L_q = torch.zeros((), device=device, dtype=dtype)
            else:
                lv = logvars[:, :, :, j:]  # (B, S, d, T-j)
                L_q = gaussian_entropy(lv).mean()
        else:
            L_q = _mc_entropy_from_logq(logq_paths, j)

        # ``L_p`` (cross-entropy, for logging only) recovered from the
        # identity ``CE = KL + H(q)`` so that ``kl == L_p - L_q`` matches
        # the V1 invariant.
        L_p = kl + L_q

        return {"kl": kl, "L_p": L_p, "L_q": L_q}

    # ------------------------------------------------------------------ #
    # Per-chunk ESM loss.
    # ------------------------------------------------------------------ #
    def _esm_chunk_loss(
        self,
        mu_t: torch.Tensor,        # (N, d)
        sigma2_t: torch.Tensor,    # (N, d)  >= 0
        z_hist: torch.Tensor,      # (N, d, j)
        ctx: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return scalar = sum over the N rows of (mean-over-S_k weighted sq.err)."""
        N, d = mu_t.shape
        device = mu_t.device
        dtype = mu_t.dtype

        # ---- side info window (history + 1 target) ---- #
        if "hist_time_emb" not in ctx or "target_time_emb" not in ctx:
            raise ValueError(
                "DiffusionV2Transition requires hist_time_emb and target_time_emb in ctx"
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
        )  # (N, side_dim, d, j+1)

        # ---- chunk over S_k ---- #
        k_chunk = max(1, min(int(self.schedule.k_chunk), int(self.S_k)))
        total_sqerr = torch.zeros(N, device=device, dtype=dtype)

        remaining_k = int(self.S_k)
        while remaining_k > 0:
            kc = min(k_chunk, remaining_k)
            remaining_k -= kc

            k_idx = torch.multinomial(self.p_k, N * kc, replacement=True).view(
                N, kc
            )  # (N, kc)
            eps_n = torch.randn(N, d, kc, device=device, dtype=dtype)

            z_in, F_target = self._vp_precondition(
                mu_t=mu_t, sigma2_t=sigma2_t, k_idx=k_idx, eps=eps_n
            )  # both (N, d, kc)

            # Latent window: concat [hist, z_in] -> (N, d, j+1, kc) -> (N*kc, d, j+1)
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
            weights = (
                self.wtilde[k_flat] / self.p_k[k_flat].clamp_min(1e-12)
            ).detach()  # (N*kc,)

            F_pred = self.diffmodel(latent_w, side_w, c_noise_flat)  # (N*kc, d, 1)
            F_pred = F_pred.squeeze(-1)  # (N*kc, d)
            F_tgt_flat = F_target.permute(0, 2, 1).reshape(N * kc, d)

            sqerr = (F_pred - F_tgt_flat).pow(2).sum(dim=1) * weights  # (N*kc,)
            total_sqerr = total_sqerr + sqerr.view(N, kc).sum(dim=1)

        # mean over S_k, then sum over N (caller divides by B*S*T_target)
        return (total_sqerr / float(self.S_k)).sum()

    def _vp_precondition(
        self,
        mu_t: torch.Tensor,        # (N, d)
        sigma2_t: torch.Tensor,    # (N, d)
        k_idx: torch.Tensor,       # (N, S_k)
        eps: torch.Tensor,         # (N, d, S_k)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (z_in, F_target) for the ESM regression.

        z_tilde = mu_t + sqrt(sigma2_t + sigma_tilde**2) * eps   (N, d, S_k)
        s_q     = -(z_tilde - mu_t) / (sigma2_t + sigma_tilde**2)
        D_star  = z_tilde + sigma_tilde**2 * s_q
        F_target= (D_star - c_skip * z_tilde) / c_out
        z_in    = c_in * z_tilde
        """
        eps_dtype = torch.finfo(mu_t.dtype).eps

        sigma_tilde = self.sigma_tilde[k_idx]      # (N, S_k)
        sigma_tilde2 = sigma_tilde * sigma_tilde   # (N, S_k)
        c_skip = self.c_skip[k_idx]                # (N, S_k)
        c_out = self.c_out[k_idx].clamp_min(eps_dtype)  # (N, S_k)
        c_in = self.c_in[k_idx]                    # (N, S_k)

        # Broadcast (N, S_k) -> (N, 1, S_k) for the latent dim.
        st2_ = sigma_tilde2.unsqueeze(1)           # (N, 1, S_k)
        cskip_ = c_skip.unsqueeze(1)
        cout_ = c_out.unsqueeze(1)
        cin_ = c_in.unsqueeze(1)

        sigma2_t_ = sigma2_t.unsqueeze(-1)         # (N, d, 1)
        mu_t_ = mu_t.unsqueeze(-1)                 # (N, d, 1)

        var_total = (sigma2_t_ + st2_).clamp_min(eps_dtype)  # (N, d, S_k)
        z_tilde = mu_t_ + var_total.sqrt() * eps             # (N, d, S_k)

        s_q = -(z_tilde - mu_t_) / var_total                 # (N, d, S_k)
        D_star = z_tilde + st2_ * s_q                        # (N, d, S_k)
        F_target = (D_star - cskip_ * z_tilde) / cout_       # (N, d, S_k)
        z_in = cin_ * z_tilde                                # (N, d, S_k)

        return z_in, F_target

    # ------------------------------------------------------------------ #
    # Sampling: VP probability-flow ODE (Euler-Maruyama in tau).
    # ------------------------------------------------------------------ #
    def sample(
        self,
        z_hist: torch.Tensor,
        S: int = 1,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Draw one sample of z_t conditioned on z_hist via reverse VP-SDE.

        Args:
            z_hist: (B, d, j)
            S: ignored (returns one sample per batch row, in axis 1).
            ctx: dict with ``hist_time_emb`` (B, j, E) and ``target_time_emb``
                (B, 1, E), or ``time_embed_window`` (B, j+1, E).

        Returns:
            (B, 1, d) sampled latent at the target step.
        """
        del S  # current sampler returns 1 trajectory; mirror V1 contract
        B, d, j_in = z_hist.shape
        if j_in != self.j:
            raise ValueError(f"Expected history j={self.j}, got {j_in}")
        device = z_hist.device
        dtype = z_hist.dtype

        # build time window (B, j+1, E+V)
        time_win: Optional[torch.Tensor] = None
        if ctx is not None:
            if "time_embed_window" in ctx:
                time_win = ctx["time_embed_window"]
            elif "hist_time_emb" in ctx and "target_time_emb" in ctx:
                hist_emb = ctx["hist_time_emb"]
                tgt_emb = ctx["target_time_emb"]
                if "hist_covariates" in ctx:
                    hist_emb = torch.cat([hist_emb, ctx["hist_covariates"]], dim=-1)
                if "target_covariates" in ctx:
                    tgt_emb = torch.cat([tgt_emb, ctx["target_covariates"]], dim=-1)
                time_win = torch.cat([hist_emb, tgt_emb], dim=1)
        if time_win is None:
            raise ValueError("DiffusionV2Transition.sample requires time embeddings in ctx")

        cond_mask = torch.ones(B, d, self.j + 1, device=device, dtype=dtype)
        cond_mask[..., -1] = 0.0
        side_win = get_side_info(
            data_dim=self.latent_dim,
            time_embed=time_win,
            embed_layer=self.embed_layer,
            cond_mask=cond_mask,
            device=device,
        )

        z_sample = self._vp_pf_sample(z_hist=z_hist, side_win=side_win)
        return z_sample.unsqueeze(1)  # (B, 1, d)

    @torch.no_grad()
    def _vp_pf_sample(
        self,
        z_hist: torch.Tensor,    # (B, d, j)
        side_win: torch.Tensor,  # (B, side_dim, d, j+1)
    ) -> torch.Tensor:
        """Reverse-time probability-flow Euler sampler over the precomputed tau grid."""
        B, d, _ = z_hist.shape
        device = z_hist.device
        dtype = z_hist.dtype
        eps_dtype = torch.finfo(dtype).eps

        # Initialise at the largest tau: under VP, the marginal variance is 1 for
        # standard data (sigma_data = 1), so we sample the prior as N(0, I).
        x = torch.randn(B, d, device=device, dtype=dtype)

        K = self.num_steps
        # Iterate from largest tau (index K-1) down to smallest (index 0).
        for i in range(K - 1, 0, -1):
            alpha_i = self.alpha[i]
            sigma_t_i = self.sigma_tilde[i]
            c_skip_i = self.c_skip[i].expand(B)
            c_out_i = self.c_out[i].clamp_min(eps_dtype).expand(B)
            c_in_i = self.c_in[i]
            c_noise_i = self.c_noise[i].expand(B)

            # Convert native z to VE-coords z_tilde (drives the network).
            z_tilde = x / alpha_i.clamp_min(eps_dtype)
            z_in = (c_in_i * z_tilde).unsqueeze(2)  # (B, d, 1)
            latent_w = torch.cat([z_hist, z_in], dim=2)  # (B, d, j+1)
            F_pred = self.diffmodel(latent_w, side_win, c_noise_i).squeeze(-1)
            D_pred = c_skip_i.view(B, 1) * z_tilde + c_out_i.view(B, 1) * F_pred
            # score in VE coords
            s_pred = (D_pred - z_tilde) / (sigma_t_i.clamp_min(eps_dtype) ** 2)
            # native-coord score: s_native = s_tilde / alpha
            s_native = s_pred / alpha_i.clamp_min(eps_dtype)

            # VP probability-flow ODE: dx = [-1/2 beta x - 1/2 beta s_native] dtau
            beta_i = self.beta[i]
            d_tau = (self.tau[i - 1] - self.tau[i]).to(dtype)  # < 0

            drift = -0.5 * beta_i * x - 0.5 * beta_i * s_native
            x = x + d_tau * drift

        return x


# ---------------------------------------------------------------------------
# Hydra-zen config
# ---------------------------------------------------------------------------

DiffusionV2TransitionConf = builds(
    DiffusionV2Transition,
    unet=CSDIUnetConf(),
    schedule=DiffusionV2ScheduleConfig(),
    populate_full_signature=True,
)
