"""Core DDSSM model: ELBO forward pass, encoder/decoder/transition dispatch, and forecast rollout."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, final

import torch
import torch.nn as nn

from .aux_posterior import AuxPosterior
from .centering.baselines import BaseBaseline
from .centering.regularizers import r_mu_p_loss, r_sigma_p_loss
from .centering.sigma_data import SigmaDataBuffer
from .decoder import BaseDecoder, GaussianDecoder
from .encoder import (
    BaseEncoder,
    GaussianEncoder,
)
from .losses import LossComponents
from .net_utils import (
    time_embedding,
)
from .transitions.transitions import BaseTransition, GaussianTransition


@dataclass
class ProbeBatch:
    """Detached latent-encoding payload reused by variance probes."""

    zs: torch.Tensor
    logq_paths: torch.Tensor
    enc_stats: dict
    time_embed: torch.Tensor
    covariates: torch.Tensor | None = None

    def as_kwargs(self) -> dict:
        return {
            "enc_stats": self.enc_stats,
            "zs": self.zs,
            "logq_paths": self.logq_paths,
            "time_embed": self.time_embed,
            "covariates": self.covariates,
        }


@final
class DDSSM_base(nn.Module):
    """Diffusion-Driven State Space Model (DDSSM).

    Implements the full variational model: encoder q_ϕ, decoder p_θ,
    initialisation prior p_η, and a pluggable transition p_ψ (Gaussian or
    diffusion-based).  The ``forward`` method returns the ELBO loss and its
    components; ``forecast`` autoregressively rolls out future latent states
    and decodes them.

    Args:
        encoder: Instantiated encoder module.
        decoder: Instantiated decoder module.
        transition: Instantiated transition module.
        j: Number of history steps used by each module.
        data_dim: Observed data dimension D.
        latent_dim: Latent dimension d.
        emb_time_dim: Time embedding dimension.
        covariate_dim: Dimension of time-varying covariates (0 = none).
        static_embed_dim: Per-feature categorical embedding size.
        num_classes_per_static: Vocabulary size per static categorical feature.
        use_observation_mask: Whether to use the observation mask in the encoder.
        mask_emb_dim: Mask embedding dimension (stored for reference).
        logvar_min: Min clamp for decoder/encoder log-variance.
        logvar_max: Max clamp for decoder/encoder log-variance.
        S: Number of Monte Carlo encoder samples.
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
        num_classes_per_static: List[int] | None = None,
        use_observation_mask: bool = True,
        mask_emb_dim: int = 8,
        logvar_min: float = -7.0,
        logvar_max: float = 7.0,
        S: int = 1,
        # --- VHP-via-diffusion + baseline-centering path (the only init path) ---
        aux_posterior: AuxPosterior | None = None,
        baseline: BaseBaseline | None = None,
        baseline_anchor: BaseBaseline | None = None,
        baseline_mode: str = "pinned",
        sigma_data: SigmaDataBuffer | None = None,
        stage1_transition: BaseTransition | None = None,
        report_sigma_data_diag: bool = True,
    ) -> None:
        super().__init__()

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
                self.static_embeddings.append(
                    nn.Embedding(num_classes, self.static_embed_dim)
                )
            self.total_static_dim = len(self.num_classes_per_static) * self.static_embed_dim
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
        if baseline_mode not in ("pinned", "learnable"):
            raise ValueError(
                f"baseline_mode must be 'pinned' or 'learnable'; got {baseline_mode!r}"
            )

        # Sub-modules (already instantiated)
        self.encoder: BaseEncoder = encoder
        self.decoder = decoder
        self.transition = transition

        # --- model-v2 slots ---
        self.aux_posterior: AuxPosterior | None = aux_posterior
        self.baseline: BaseBaseline | None = baseline
        self.baseline_anchor: BaseBaseline | None = baseline_anchor
        self.baseline_mode: str = baseline_mode
        self.sigma_data: SigmaDataBuffer | None = sigma_data
        self.stage1_transition: BaseTransition | None = stage1_transition
        self._report_sigma_data_diag: bool = bool(report_sigma_data_diag)

        # Orchestrator flips this between stages.
        self.stage_selector: str = "stage_2"

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
        covariates = batch.get("covariates", None)
        static_covariates = batch.get("static_covariates", None)

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
            k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in enc_stats.items()
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

        total_neg_logp = torch.zeros((), device=device, dtype=dtype)
        total_obs = torch.zeros((), device=device, dtype=dtype)

        # For calibration
        res2_sum = torch.zeros((), device=device, dtype=dtype)
        sigma2_sum = torch.zeros((), device=device, dtype=dtype)

        # Vectorize S dimension
        # zs: (B, S, d, T) -> (B*S, d, T)
        zs_flat = zs.reshape(B * S, d, T)

        # Expand data to match: (B, D, T) -> (B, S, D, T) -> (B*S, D, T)
        obs_flat = observed_data.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, D, T)
        mask_flat = (
            observation_mask.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, D, T)
        )

        # Expand time embeddings: (B, T, E) -> (B*S, T, E)
        time_flat = time_embed.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, T, -1)

        if covariates is not None:
            V = covariates.shape[1]
            covariates_flat = (
                covariates.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, V, T)
            )
        else:
            covariates_flat = None

        if static_embed is not None:
            # static_embed is (B, D, E_s) -> expand to (B*S, D, E_s)
            V_s = static_embed.shape[2]
            static_flat = (
                static_embed.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, D, V_s)
            )
        else:
            static_flat = None

        for t in range(T):
            # Slice current step for all B*S
            x_t = obs_flat[:, :, t]  # (BS, D)
            m_t = mask_flat[:, :, t]  # (BS, D)
            t_idx = torch.full((B * S,), t, device=device, dtype=torch.long)

            # Slice history for all B*S
            # Note: This preserves logic of passing 0..t+1
            z_hist = zs_flat[:, :, : t + 1]  # (BS, d, k)
            if z_hist.shape[-1] > self.j:
                z_hist = z_hist[..., -self.j :]

            # Single Decoder Call (Vectorized over S)
            logp_t, mu_x, logvar_x, obs_count_t = self.decoder.log_likelihood(
                x_t=x_t,
                z_hist=z_hist,
                time_embed=time_flat,
                time_idx=t_idx,
                observation_mask_t=m_t,
                covariates=covariates_flat,
                static_embed=static_flat,
            )

            # Reshape back to (B, S) for averaging
            # logp_t: (BS,) -> (B, S)
            logp_paths = logp_t.view(B, S)
            obs_paths = obs_count_t.view(B, S)

            # mu_x: (BS, D) -> (B, S, D) -> permute to (S, B, D) to match original logic
            mu_paths = mu_x.view(B, S, D).permute(1, 0, 2)  # (S, B, D)
            logvar_paths = logvar_x.view(B, S, D).permute(1, 0, 2)  # (S, B, D)

            mean_logp_b = logp_paths.mean(dim=1)  # (B,)
            mean_obs_b = obs_paths.mean(dim=1)  # (B,)

            total_neg_logp = total_neg_logp - mean_logp_b.sum()
            total_obs = total_obs + mean_obs_b.sum()

            # calibration accumulators over all S equally
            # x_t is (BS, D) -> view (B, S, D) -> permute (S, B, D)
            x_t_sb = x_t.view(B, S, D).permute(1, 0, 2)
            m_t_sb = m_t.view(B, S, D).permute(1, 0, 2)

            resid2 = (x_t_sb - mu_paths).pow(2)  # (S, B, D)
            sigma2 = logvar_paths.exp().clamp_min(1e-6)  # (S, B, D)

            res2_sum = res2_sum + (resid2 * m_t_sb).sum()
            sigma2_sum = sigma2_sum + (sigma2 * m_t_sb).sum()

        total_obs = total_obs.clamp_min(1.0)

        # Mean NLL per observed entry, then scale back to (T * D)
        nll_mean_per_entry = total_neg_logp / total_obs
        L_rec = nll_mean_per_entry * (D * T)

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
    ) -> Dict[str, torch.Tensor]:
        """Initialization loss for the first ``j`` latent steps.

        Pure pass-through to the active transition's hierarchical VHP init
        walk (:meth:`BaseTransition.transition_kl_init`). Per ADR-0006 the
        transition owns the per-step scoring, the encoder-entropy policy
        (``-H`` for closed-form, ``0`` for the ESM-cancelling diffusion
        path), and any σ_data update — the model does not gate on
        transition type. Returns ``{loss, entropy, vhp, kl_aux, loss_init}``.
        """
        del logq_paths  # the VHP path scores encoder moments, not sampled log-q
        return self._active_transition().transition_kl_init(
            enc_stats=enc_stats,
            zs=zs,
            aux_posterior=self.aux_posterior,
            time_embed=time_embed,
            sigma_data=self.sigma_data,
            covariates=covariates,
        )

    def _active_transition(self) -> nn.Module:
        """Return the transition picked by :attr:`stage_selector`."""
        if (
            self.stage_selector == "stage_1"
            and self.stage1_transition is not None
        ):
            return self.stage1_transition
        return self.transition

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
        """Compute transition KL term, dispatched on ``self.stage_selector``.

        When ``stage_selector == "stage_1"`` and ``stage1_transition`` is
        set, uses it; otherwise uses ``self.transition`` (the stage-2 /
        legacy slot).  The σ_data buffer is forwarded as a kwarg to
        whichever transition is called (legacy V2 ignores it; new V3 /
        BaselineGaussian consume it).

        Returns the transition's dict (at least ``"kl"``, plus any
        sub-components for logging).
        """
        active = self._active_transition()
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
        return active.transition_kl(**transition_kwargs)

    def _gather_z_hist_samples_for_regularizers(
        self, zs: torch.Tensor
    ) -> torch.Tensor:
        """Return ``(N, d, j)`` detached z_hist windows for the regularizers.

        Walks the transition-target range ``t = j … T-1`` (0-based code
        indexing) and stacks all (B, S, chunk_len=1) histories into a
        flat ``(N, d, j)`` tensor.  Detached so the regularizer
        gradient flows only into the baseline / anchor.
        """
        B, S, d, T = zs.shape
        j = self.j
        if T <= j:
            return torch.zeros(0, d, j, device=zs.device, dtype=zs.dtype)
        # (B, S, d, T - j + j - 1 + 1) -> unfold gives (B, S, d, T - j, j)
        unfolded = zs.unfold(dimension=-1, size=j, step=1)[..., :T - j, :]
        # Permute to (B, S, T-j, d, j) and flatten leading dims.
        return (
            unfolded.permute(0, 1, 3, 2, 4)
            .reshape(-1, d, j)
            .detach()
        )

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
          * Per-transition: :meth:`DiffusionV3Transition.log_prob` for
            ``t = j..T-1`` via the probability-flow ODE.
          * Decoder: :meth:`BaseDecoder.log_likelihood` summed across
            ``t = 0..T-1``.
          * Initial state: ``log p_ψ(z_{1:j})`` via
            :meth:`DiffusionV3Transition.log_prob_init` (VHP IS under
            ``q_Φ``) when an ``aux_posterior`` is present; otherwise 0.

        Args:
            observed_data, observation_mask, timepoints, covariates,
                static_covariates: same shapes/semantics as
                :meth:`forward`.
            K: number of trajectory samples (defaults to ``self.S``).
            rtol, atol, method, divergence_mode, generator: forwarded
                to the per-transition prob-flow ODE solver.

        Returns:
            ``(B,)`` per-sequence log-likelihood estimate.
        """
        from .likelihood import iwae_log_likelihood

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

        transition = self._active_transition()
        log_p_trans = torch.zeros(B, K_use, device=device, dtype=dtype)
        for t in range(j, T):
            if self.sigma_data is not None:
                sigma_d2 = self.sigma_data.read(t + 1).expand(B).to(
                    device=device, dtype=dtype
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
        report_scaled: bool = True,
    ):
        """Compute ELBO loss and its components for a batch.

        Args:
            observed_data: Observed time-series, shape ``(B, D, T)``.
            observation_mask: Binary mask (1 = observed, 0 = missing), shape ``(B, D, T)``.
            timepoints: Integer or real timestamps, shape ``(B, T)``.
            covariates: Optional time-varying covariates, shape ``(B, V, T)``.
            static_covariates: Optional static categorical features, shape ``(B, D, V_s)``.
            train: If ``False``, also returns posterior samples/stats in ``stats``.
            report_scaled: If ``True``, also emit dimension-normalised variants of
                each loss component under ``"<key>_scaled"`` keys in ``metrics``.

        Returns:
            components: ``LossComponents`` with unweighted per-term
                tensors (recon, init_kl, trans_kl, r_sigma_p, r_mu_p).
                The loss object applies its own weights and produces
                the scalar that gets backpropped.
            metrics: Dict of scalar tensors for logging (regularizer
                values here are still WEIGHTED for log continuity).
            stats: Empty dict during training; contains ``zs``, ``mus``,
                ``logvars`` when ``train=False``.
        """
        j = self.j

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
        L_vhp = init_terms.get(
            "vhp", torch.tensor(0.0, device=observed_data.device)
        )
        L_ent_init = init_terms.get(
            "entropy", torch.tensor(0.0, device=observed_data.device)
        )

        trans_terms = self._compute_transition_kl(
            zs=zs,
            logq_paths=logq_paths,
            enc_stats=enc_stats,
            time_embed=time_embed,
            covariates=covariates,
        )
        L_trans = trans_terms["kl"]
        trans_subterms = {k: v for k, v in trans_terms.items() if k != "kl"}

        # Model-v2 centering regularizers (free functions in
        # ``centering.regularizers``).  Returned UNWEIGHTED in
        # `LossComponents`; the loss object applies `lambda_sigma_p`
        # and `lambda_mu_p` (see ADR-0004). Stage gating stays here:
        # the regularizer is exactly zero outside its applicable stage,
        # regardless of the loss object's λ.
        r_sigma_p_raw = torch.tensor(0.0, device=observed_data.device)
        r_mu_p_raw = torch.tensor(0.0, device=observed_data.device)
        if self.baseline is not None:
            # Build a flat (N, d, j) batch of z_hist samples from the
            # encoder paths for evaluating the regularizers.  Detached
            # so gradient flows only into the baseline / anchor, not
            # back through the encoder.
            z_hist_samples = self._gather_z_hist_samples_for_regularizers(zs)
            if self.stage_selector == "stage_1":
                r_sigma_p_raw = r_sigma_p_loss(
                    baseline=self.baseline,
                    z_hist_samples=z_hist_samples,
                    lambda_sigma_p=1.0,
                )
            if (
                self.stage_selector == "stage_2"
                and self.baseline_mode == "learnable"
                and self.baseline_anchor is not None
            ):
                r_mu_p_raw = r_mu_p_loss(
                    baseline=self.baseline,
                    baseline_anchor=self.baseline_anchor,
                    z_hist_samples=z_hist_samples,
                    lambda_mu_p=1.0,
                )

        distortion = L_rec
        # `loss` and `rate` in the metrics dict below are UNWEIGHTED
        # post-ADR-0004 — the loss object owns weights now.
        rate = L_init + L_trans + r_sigma_p_raw + r_mu_p_raw
        loss = distortion + rate

        dev = L_rec.device
        dtype = L_rec.dtype
        d = int(self.latent_dim)
        t = int(timepoints.size(1))
        j = int(self.j)
        Tk = max(t - j, 1)
        D_dim = torch.tensor(float(d), device=dev, dtype=dtype)
        DT = torch.tensor(float(d * t), device=dev, dtype=dtype)
        DTk = torch.tensor(float(d * Tk), device=dev, dtype=dtype)

        metrics = {
            "loss/total": loss.detach(),
            "loss/distortion/rec": L_rec.detach(),
            "loss/rate/init/tot": L_init.detach(),
            "loss/rate/init/vhp": L_vhp.detach(),
            "loss/rate/init/entropy": L_ent_init.detach(),
            "loss/rate/trans/kl": L_trans.detach(),
            "loss/rate/total": rate.detach(),
            "calib/ratio_res2_to_sigma2": L_rec_calib.detach(),
            # Post-ADR-0004: report UNWEIGHTED regularizer values.
            # The loss object that's currently active owns the
            # weighting; downstream eval applies it when it cares.
            "loss/rate/trans/r_sigma_p": r_sigma_p_raw.detach(),
            "loss/rate/trans/r_mu_p": r_mu_p_raw.detach(),
        }
        # Surface model-v2 init-term sub-components when present.
        if self.aux_posterior is not None:
            if "kl_aux" in init_terms:
                metrics["loss/rate/init/kl_aux"] = init_terms["kl_aux"].detach()
            if "loss_init" in init_terms:
                metrics["loss/rate/init/loss_init"] = init_terms["loss_init"].detach()
        # Surface per-t σ_data²[t] buffer values when configured.  These
        # feed the post-hoc ``sigma_data_drift`` metric's trajectory plot
        # (init-experiment.org § Headline metrics, metric 6).  Logged once
        # per step so the trajectory is recoverable from metrics.csv alone.
        if self.sigma_data is not None and self._report_sigma_data_diag:
            buf = self.sigma_data.sigma_data2.detach()
            for slot, value in enumerate(buf):
                metrics[f"diag/sigma_data2/t={slot + 1}"] = value
        # Surface any optional transition sub-components (e.g. L_p, L_q) under
        # transition-driven keys.
        for key, val in trans_subterms.items():
            metrics[f"loss/rate/trans/{key}"] = val.detach()

        if report_scaled:
            rescale = lambda key, factor: metrics.update({
                f"{key}_scaled": metrics[key] / factor
            })

            rescale("loss/total", DT)
            rescale("loss/distortion/rec", DT)
            rescale("loss/rate/init/tot", D_dim * j)
            rescale("loss/rate/init/vhp", D_dim * j)
            rescale("loss/rate/init/entropy", D_dim * j)
            rescale("loss/rate/trans/kl", DTk)
            for key in trans_subterms:
                rescale(f"loss/rate/trans/{key}", DTk)
            rescale("loss/rate/total", DT)
            rescale("calib/ratio_res2_to_sigma2", 1.0)

        stats = {}  # return optional params
        if not train:  # Optionally return posterior stats/samples for analysis
            stats["zs"] = zs
            stats["mus"] = mus
            stats["logvars"] = logvars

        components = LossComponents(
            recon=L_rec,
            init_kl=L_init,
            trans_kl=L_trans,
            r_sigma_p=r_sigma_p_raw,
            r_mu_p=r_mu_p_raw,
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
    ) -> Dict[str, torch.Tensor]:
        """Encode history → Autoregresie transition → decode future.

        Returns:
          {'pred_mean': (B, K, L2), 'pred_samples': (B, S, K, L2)}
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
            if hist_covariates is not None:
                ctx["hist_covariates"] = hist_covariates.permute(
                    0, 2, 1
                )  # to (BS, j, V)

            if covariates_all_bs is not None:
                target_covariates = covariates_all_bs[:, :, t_end : t_end + 1]
                ctx["target_covariates"] = target_covariates.permute(
                    0, 2, 1
                )  # to (BS, 1, V)

            # Sample z_t
            # transition.sample returns (BS, 1, d) because we pass S=1 (we already flattened S)
            z_t = self.transition.sample(z_hist_flat, S=1, ctx=ctx)  # (BS, 1, d)
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
        weight_decay=1e-2,
        batch_size=16,
        grad_accum_steps=4,
        t_chunk=16,
        clip_grad_norm=None,
        enc_lr=5e-4,
        dec_lr=5e-4,
        trans_lr=5e-4,
        logvar_min=-7.0,
        logvar_max=7.0,
    )



# ---------------------------------------------------------------------------
# Hyperparameters and top-level DDSSMConf, co-located with DDSSM_base.
# ---------------------------------------------------------------------------


@dataclass
class DDSSMHyperParamsConf:
    """Training hyperparameters for DDSSM."""

    S: int = 1
    ema_decay: float = 0.999
    weight_decay: float = 1e-2
    batch_size: int = 16
    grad_accum_steps: int = 4
    t_chunk: int = 16
    clip_grad_norm: float | None = None

    enc_lr: float = 5e-4
    dec_lr: float = 5e-4
    trans_lr: float = 5e-4

    logvar_min: float = -7.0
    logvar_max: float = 7.0


