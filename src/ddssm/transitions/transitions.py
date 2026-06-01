"""Pluggable transition modules for DDSSM.

Defines the ``BaseTransition`` interface and the concrete ``GaussianTransition``
(non-linear diagonal Gaussian).

``BaseTransition`` interface:
    ``transition_kl(enc_stats, zs, logq_paths, time_embed, covariates=None) -> dict``
        Returns ``{"kl": L_trans, ...}`` containing the scalar KL term and any
        optional sub-components the implementation chooses to expose for
        logging (e.g. ``"L_p"``, ``"L_q"``).
    ``seq_log_prob(zs, time_embed, ...) -> Tensor``
        Per-batch ``sum_t E_q[log p_psi(z_t | z_{t-j:t-1})]``; lower-level
        building block used internally by some transitions.
    ``log_prob(z, z_hist, ctx=None) -> Tensor``
        Per-sample log p(z | z_hist, ctx).  Optional; raises NotImplementedError
        if not supported.
    ``sample(z_hist, S=1, ctx=None) -> Tensor``
        Draw S samples from p(z_t | z_hist).  Optional.
    ``prior_params(z_hist, ctx=None) -> (mu, logvar)``
        Return Gaussian prior parameters.  Optional diagnostic helper.
"""

import math
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Tuple, Optional

import torch
import torch.nn as nn


from ..encoder import GaussianHead, ContextProducer
from ..gaussians import GaussianStats, gaussian_entropy, gaussian_kl_divergence
from ..torch_compile import maybe_compile

if TYPE_CHECKING:
    from ..aux_posterior import AuxPosterior
    from ..centering.sigma_data import SigmaDataBuffer


def _mc_entropy_from_logq(
    logq_paths: torch.Tensor,  # (B, S, T)
    j: int,
) -> torch.Tensor:
    """Monte-Carlo encoder entropy over ``t = j..T-1``.

    Returns the scalar ``-E_q[ sum_{t=j}^{T-1} log q(z_t|·) ]`` averaged over
    sample paths and batch.  Mirrors ``BaseEncoder.mc_entropy_transition``
    semantics so transitions can compute it without holding an encoder ref.
    """
    B, S, T = logq_paths.shape
    if j >= T:
        return torch.zeros((), device=logq_paths.device, dtype=logq_paths.dtype)
    logs = logq_paths[:, :, j:]  # (B, S, T-j)
    neg_entropy = logs.mean(dim=1).sum(dim=1)  # (B,)
    return -neg_entropy.mean()


class BaseTransition(nn.Module):
    """Abstract transition interface."""

    def prior_params(
        self, z_hist: torch.Tensor, ctx: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (p_mu, p_logvar) conditioned on z_hist/context.

        Optional helper for diagnostics. Implementors may raise NotImplementedError.
        """
        raise NotImplementedError

    def _iter_window_chunks(
        self,
        zs: torch.Tensor,  # (B, S, d, T)
        time_embed: torch.Tensor,  # (B, T, E_t)
        time_chunk_num: Optional[int] = None,
        time_chunk_size: Optional[int] = None,
        covariates: Optional[torch.Tensor] = None,
    ) -> Iterator[Tuple[int, int, int, int, int, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """Yield per-chunk window tensors + context for the time loop.

        Yields tuples ``(B, S, current_chunk_len, t_start, t_end, z_target_flat,
        z_hist_flat, ctx)`` where ``z_target_flat`` has shape ``(N, d)`` and
        ``z_hist_flat`` has shape ``(N, d, j)`` with ``N = B*S*chunk_len``.
        """
        B, S, d, T = zs.shape
        j = self.j
        device = zs.device

        total_steps = T - j
        if total_steps <= 0:
            return

        if time_chunk_size is not None and time_chunk_size > 0:
            chunk_size = time_chunk_size
        elif time_chunk_num is not None and time_chunk_num > 0:
            chunk_size = math.ceil(total_steps / time_chunk_num)
        else:
            chunk_size = 1

        for start_rel in range(0, total_steps, chunk_size):
            end_rel = min(start_rel + chunk_size, total_steps)

            t_start = j + start_rel
            t_end = j + end_rel
            current_chunk_len = t_end - t_start

            z_target_chunk = zs[..., t_start:t_end]  # (B, S, d, chunk_len)
            BS_chunk = B * S * z_target_chunk.size(-1)

            zs_source = zs[..., t_start - j : t_end]  # (B, S, d, chunk_len + j)
            z_hist_chunk = zs_source.unfold(dimension=-1, size=j, step=1)
            z_hist_chunk = z_hist_chunk[..., :current_chunk_len, :]

            t_emb_source = time_embed[:, t_start - j : t_end, :]
            t_hist_chunk = t_emb_source.unfold(dimension=1, size=j, step=1)
            t_hist_chunk = t_hist_chunk[:, :current_chunk_len, :, :]

            z_target_flat = z_target_chunk.permute(0, 1, 3, 2).reshape(-1, d)
            z_hist_flat = z_hist_chunk.permute(0, 1, 3, 2, 4).reshape(-1, d, j)

            t_hist_chunk = t_hist_chunk.permute(0, 1, 3, 2)  # (B, chunk_len, j, E)
            t_hist_flat = (
                t_hist_chunk
                .unsqueeze(1)
                .expand(-1, S, -1, -1, -1)
                .reshape(BS_chunk, j, self.emb_time_dim)
            )

            if covariates is not None:
                c_emb_source = covariates[:, :, t_start - j : t_end]
                c_hist_chunk = c_emb_source.unfold(dimension=2, size=j, step=1)
                c_hist_chunk = c_hist_chunk[:, :, :current_chunk_len, :]
                c_hist_chunk = c_hist_chunk.permute(0, 2, 3, 1)  # (B, chunk_len, j, V)
                c_hist_flat = (
                    c_hist_chunk
                    .unsqueeze(1)
                    .expand(-1, S, -1, -1, -1)
                    .reshape(BS_chunk, j, covariates.size(1))
                )

                c_target_chunk = covariates[:, :, t_start:t_end].permute(0, 2, 1)
                c_target_flat = (
                    c_target_chunk
                    .unsqueeze(1)
                    .expand(-1, S, -1, -1)
                    .reshape(BS_chunk, 1, covariates.size(1))
                )
            else:
                c_hist_flat = None
                c_target_flat = None

            t_target_chunk = time_embed[:, t_start:t_end, :]
            t_target_flat = (
                t_target_chunk
                .unsqueeze(1)
                .expand(-1, S, -1, -1)
                .reshape(BS_chunk, 1, self.emb_time_dim)
            )

            ctx: Dict[str, torch.Tensor] = {
                "hist_time_emb": t_hist_flat,
                "target_time_emb": t_target_flat,
            }
            if c_hist_flat is not None:
                ctx["hist_covariates"] = c_hist_flat
            if c_target_flat is not None:
                ctx["target_covariates"] = c_target_flat

            yield (
                B,
                S,
                current_chunk_len,
                t_start,
                t_end,
                z_target_flat,
                z_hist_flat,
                ctx,
            )

    def seq_log_prob(
        self,
        zs: torch.Tensor,  # (B, S, d, T)
        time_embed: torch.Tensor,  # (B, T, E_t)
        time_chunk_num: Optional[int] = None,
        time_chunk_size: Optional[int] = None,
        covariates: Optional[torch.Tensor] = None,
        mc_override: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute ``sum_{t=j}^{T-1} E_q[ log p_psi(z_t | z_{t-j:t-1}) ]``
        (or a bound thereof) over chunks to bound peak memory.

        Args:
            zs: ``(B, S, d, T)`` latent samples from encoder.
            time_embed: ``(B, T, E_t)`` absolute time embeddings.
            time_chunk_num: Number of chunks to split the time dimension into.
            time_chunk_size: Size of each chunk along the time dimension.
                Overrides ``time_chunk_num`` when provided.
            covariates: Optional ``(B, V, T)`` time-varying covariates.
            mc_override: Optional forced MC draws forwarded to
                :meth:`log_prob` (used by the variance probe).

        Returns:
            ``(B,)`` total log-probability per batch element.
        """
        B, S, d, T = zs.shape
        j = self.j
        device = zs.device

        total_steps = T - j
        if total_steps <= 0:
            return torch.zeros(B, device=device)

        total_nll = torch.zeros(B, device=device)

        for (
            B_,
            S_,
            current_chunk_len,
            _t_start,
            _t_end,
            z_target_flat,
            z_hist_flat,
            ctx,
        ) in self._iter_window_chunks(
            zs,
            time_embed,
            time_chunk_num=time_chunk_num,
            time_chunk_size=time_chunk_size,
            covariates=covariates,
        ):
            log_p_flat = self.log_prob(z=z_target_flat, z_hist=z_hist_flat, ctx=ctx, mc_override=mc_override)
            log_p = log_p_flat.view(B_, S_, current_chunk_len)

            # Sum over time in chunk, Mean over S
            chunk_nll = log_p.sum(dim=2).mean(dim=1)
            total_nll += chunk_nll

        return total_nll

    def transition_kl(
        self,
        enc_stats: GaussianStats,
        zs: torch.Tensor,  # (B, S, d, T)
        logq_paths: torch.Tensor,  # (B, S, T)
        time_embed: torch.Tensor,  # (B, T, E_t)
        covariates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the transition KL contribution to the ELBO rate.

        Implementations must return a dict containing at least ``"kl"`` (the
        scalar ``L_trans`` term, summed over ``t = j..T-1`` and averaged over
        ``B`` and ``S``).  Implementations may additionally include sub-component
        scalars (e.g. ``"L_p"`` for the prior log-density / regression loss and
        ``"L_q"`` for the encoder entropy / log-density contribution) which the
        caller will surface as logging metrics.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Hierarchical VHP init term (t = 1 … j) — shared walk + hooks.
    # ------------------------------------------------------------------
    def transition_kl_init(
        self,
        enc_stats: GaussianStats,
        zs: torch.Tensor,                 # (B, S, d, T)
        aux_posterior: "AuxPosterior",
        time_embed: torch.Tensor,         # (B, T, E_t)
        sigma_data: "Optional[SigmaDataBuffer]" = None,
        covariates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Hierarchical VHP init term over ``t = 1 … j`` (shared walk).

        Samples the auxiliary previous states ``z_aux ~ q_Φ(·|z_{1:j})``,
        walks ``t = 1 … j`` with the mixed ``z_aux → real`` history, and
        accumulates the per-step score via the polymorphic
        :meth:`_score_init_step` hook. Returns the **complete** init
        decomposition ``{loss, entropy, vhp, kl_aux, loss_init}``; the
        entropy policy (:meth:`_init_entropy_term`) and any σ_data update
        (inside :meth:`_score_init_step`) are owned by the subclass, so
        :class:`DDSSM_base` never gates on transition type.
        """
        del covariates  # init does not condition on covariates
        if enc_stats.get("mus") is None or enc_stats.get("logvars") is None:
            raise ValueError(
                "transition_kl_init requires Gaussian encoder stats (mus, logvars)."
            )
        B, S, d, T = zs.shape
        j = self.j
        if d != self.latent_dim:
            raise ValueError(f"zs latent dim {d} != self.latent_dim {self.latent_dim}")
        if T < j:
            raise ValueError(f"zs has T={T} < j={j}")
        device, dtype = zs.device, zs.dtype
        BS = B * S

        # Aux latents conditioned on the S-averaged encoder samples.
        z_init = zs[..., :j].mean(dim=1)  # (B, d, j)
        z_aux, aux_mu, aux_logvar = aux_posterior.sample(z_init)
        z_hist = (
            z_aux.unsqueeze(1).expand(B, S, d, j).reshape(BS, d, j).clone()
        )

        total = torch.zeros((), device=device, dtype=dtype)
        for step in range(j):
            z_t = zs[:, :, :, step].reshape(BS, d)
            total = total + self._score_init_step(
                step=step, z_t=z_t, z_hist=z_hist, enc_stats=enc_stats,
                time_embed=time_embed, sigma_data=sigma_data, B=B, S=S, T=T,
            )
            # Shift history: drop oldest, append the real z_t.
            if j > 1:
                z_hist = torch.cat([z_hist[:, :, 1:], z_t.unsqueeze(-1)], dim=-1)
            else:
                z_hist = z_t.unsqueeze(-1)

        loss_init = total / float(BS)
        kl_aux = aux_posterior.kl_against_standard_normal(aux_mu, aux_logvar)
        entropy = self._init_entropy_term(enc_stats)
        vhp = loss_init + kl_aux
        return {
            "loss": entropy + vhp,
            "entropy": entropy,
            "vhp": vhp,
            "kl_aux": kl_aux,
            "loss_init": loss_init,
        }

    def _score_init_step(
        self,
        *,
        step: int,
        z_t: torch.Tensor,            # (BS, d) encoder sample at this init step
        z_hist: torch.Tensor,        # (BS, d, j) mixed aux→real history
        enc_stats: GaussianStats,
        time_embed: torch.Tensor,    # (B, T, E_t)
        sigma_data: "Optional[SigmaDataBuffer]",
        B: int,
        S: int,
        T: int,
    ) -> torch.Tensor:
        """Per-step init score, **summed over B·S**. Override per transition.

        Implementations also own any σ_data update at this step.
        """
        raise NotImplementedError

    def _init_entropy_term(self, enc_stats: GaussianStats) -> torch.Tensor:
        """Encoder-entropy contribution to the init loss.

        Default: ``-H(q_φ(z_{1:j}|x))`` (closed-form Gaussian transitions
        need the full entropy). The diffusion path overrides this to ``0``
        because its ESM expansion already cancels the encoder entropy.
        """
        lv_init = enc_stats["logvars"][..., : self.j]
        return -gaussian_entropy(lv_init).mean()

    def _init_step_time_ctx(
        self, step: int, time_embed: torch.Tensor, B: int, S: int, T: int,
    ) -> Dict[str, torch.Tensor]:
        """Build the ``(BS, j, E_t)`` history + ``(BS, 1, E_t)`` target time windows.

        Shared by the transitions whose init scoring is time-conditioned
        (closed-form Gaussian, diffusion). History slots map to abstract
        timesteps ``t-j … t-1`` clamped into ``[0, T-1]``.
        """
        j = self.j
        E = self.emb_time_dim
        device = time_embed.device
        hist_idx = torch.arange(
            step - j, step, device=device, dtype=torch.long
        ).clamp(min=0, max=T - 1)
        hist_te = time_embed.index_select(1, hist_idx)        # (B, j, E)
        tgt_te = time_embed[:, step : step + 1, :]            # (B, 1, E)
        win = torch.cat([hist_te, tgt_te], dim=1)             # (B, j+1, E)
        win = win.unsqueeze(1).expand(B, S, j + 1, E).reshape(B * S, j + 1, E)
        return {
            "hist_time_emb": win[:, :j, :],
            "target_time_emb": win[:, j : j + 1, :],
        }

    def log_prob(
        self,
        z: torch.Tensor,
        z_hist: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None,
        mc_override: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Log p(z | z_hist, ctx).

        Optional helper for diagnostics. Implementors may raise NotImplementedError.
        """
        raise NotImplementedError

    def sample(
        self,
        z_hist: torch.Tensor,
        S: int = 1,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Draw samples from p(z_t | z_hist, ctx).

        Optional helper for diagnostics. Implementors may raise NotImplementedError.
        """
        raise NotImplementedError

    def sample_latent_trajectory(
        self,
        z_hist: torch.Tensor,
        steps: int,
        S: int = 1,
        ctx: Optional[Dict[str, Any]] = None,
    ):
        """Draw autoregressive sample trajectories from the transition prior.

        Args:
            z_hist: Initial latent history, shape ``(B, d, j)`` (or equivalent).
            steps: Number of future steps to generate.
            S: Number of independent trajectories to sample.
            ctx: Optional context dict.  Recognised keys:
                - ``"time_embed"``: ``(B, j+steps, E_t)`` absolute time embeddings.
                - ``"covariates"``: ``(B, V, j+steps)`` time-varying covariates.

        Returns:
            Sampled latent trajectories, shape ``(B, S, d, steps)``.
        """

        z_hist = self._ensure_seq(z_hist)  # (B, d, j)
        B, d, j = z_hist.shape
        assert j == self.j
        device, dtype = z_hist.device, z_hist.dtype

        base_ctx = ctx or {}
        time_windows = None
        cov_windows = None
        if "time_embed" in base_ctx:
            time_embed = base_ctx["time_embed"]
            if time_embed.shape[0] != B or time_embed.shape[2] != self.emb_time_dim:
                raise ValueError(
                    f"time_embed must have shape (B, j+steps, E_t); got {tuple(time_embed.shape)}"
                )
            if time_embed.shape[1] < self.j + steps:
                raise ValueError(
                    f"time_embed needs length >= j+steps ({self.j + steps}); got {time_embed.shape[1]}"
                )

            time_windows = time_embed.unfold(
                dimension=1, size=self.j, step=1
            )  # (B, steps+1, j, E)

        if "covariates" in base_ctx:
            covs = base_ctx["covariates"]  # (B, V, j+steps)
            covs_tiled = covs.permute(0, 2, 1)  # (B, j+steps, V)
            cov_windows = covs_tiled.unfold(
                dimension=1, size=self.j, step=1
            )  # (B, steps+1, j, V)

        # copy z_hist for each trajectory
        hist = z_hist.unsqueeze(1).expand(-1, S, -1, -1).clone()  # (B, S, d, j)

        traj = torch.zeros(B, S, d, steps, device=device, dtype=dtype)

        if hist_valid_len is not None:
            hist_valid_len = hist_valid_len.to(device=device)
            if hist_valid_len.shape != (B,):
                raise ValueError(
                    f"hist_valid_len must be (B,); got {tuple(hist_valid_len.shape)}"
                )
            mask = (  # Todo fix this. incorrect
                torch.arange(self.j, device=device).view(1, 1, 1, self.j)
                >= hist_valid_len.view(B, 1, 1, 1).clamp(max=self.j)
            )
            hist = hist.masked_fill(mask, 0)
        valid_len = hist_valid_len
        for t in range(steps):
            hist_flat = hist.reshape(B * S, d, self.j)

            step_ctx = {
                k: v
                for k, v in base_ctx.items()
                if k not in ["time_embed", "covariates"]
            }
            if time_windows is not None:
                hist_time = time_windows[:, t, :, :]  # (B, E, j)
                if valid_len is not None:
                    tmask = torch.arange(self.j, device=device).view(
                        1, 1, self.j
                    ) >= valid_len.view(B, 1, 1).clamp(max=self.j)
                    hist_time = hist_time.masked_fill(tmask, 0)
                hist_time = (
                    hist_time
                    .permute(0, 2, 1)  # (B, j, E)
                    .unsqueeze(1)
                    .expand(-1, S, -1, -1)
                    .reshape(B * S, self.j, self.emb_time_dim)
                )
                step_ctx["hist_time_emb"] = hist_time

            if cov_windows is not None:
                hist_cov = cov_windows[:, t, :, :]  # (B, V, j)
                if valid_len is not None:
                    tmask = torch.arange(self.j, device=device).view(
                        1, 1, self.j
                    ) >= valid_len.view(B, 1, 1).clamp(max=self.j)
                    hist_cov = hist_cov.masked_fill(tmask, 0)
                hist_cov = (
                    hist_cov
                    .permute(0, 2, 1)  # (B, j, V)
                    .unsqueeze(1)
                    .expand(-1, S, -1, -1)
                    .reshape(B * S, self.j, self.covariate_dim)
                )
                step_ctx["hist_covariates"] = hist_cov

            if not step_ctx:
                step_ctx = None

            mu, logvar = self.prior_params(hist_flat, ctx=step_ctx)
            sigma = (0.5 * logvar).exp()
            eps = torch.randn_like(mu)
            z_next_flat = mu + sigma * eps  # (B*S, d)
            z_next = z_next_flat.view(B, S, d)
            traj[:, :, :, t] = z_next

            if self.j > 1:
                hist = torch.cat([hist[:, :, :, 1:], z_next.unsqueeze(-1)], dim=-1)
            else:
                hist = z_next.unsqueeze(-1)
            if valid_len is not None:
                valid_len = torch.clamp(valid_len + 1, max=self.j)

        return traj


class GaussianTransition(BaseTransition):
    """Non-linear diagonal Gaussian transition ``p(z_t | z_{t-j:t-1}, ...)``.

    The latent history is projected, summarised by a
    :class:`~ddssm.diffnets.ContextProducer` (no mask features), and mapped to
    ``(μ, log σ²)`` by a :class:`~ddssm.encoder.GaussianHead`.
    """

    def __init__(
        self,
        latent_dim: int,
        j: int,
        emb_time_dim: int,
        covariate_dim: int = 0,
        hidden_dim: int = 64,
        context: Callable[..., ContextProducer] | None = None,
        gaussian_head: Callable[..., GaussianHead] | None = None,
    ) -> None:
        super().__init__()
        if context is None:
            context = partial(ContextProducer, channels=8, num_layers=2)
        if gaussian_head is None:
            gaussian_head = GaussianHead

        self.latent_dim = int(latent_dim)
        self.j = int(j)
        self.emb_time_dim = int(emb_time_dim)
        self.covariate_dim = int(covariate_dim)

        self.hidden_dim = hidden_dim  # H

        # Project z history: d -> H
        self.z_hist_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        # ContextProducer over length j, with no explicit mask features
        self.context_producer = context(
            combined_dim=self.hidden_dim,
            mask_tot_dim=0,
            emb_time_dim=self.emb_time_dim + self.covariate_dim,
            combined_len=self.j,
        )

        # Gaussian head over flattened context
        head_in_dim = self.context_producer.channels * self.hidden_dim

        self.context_producer = maybe_compile(self.context_producer, dynamic=True)

        self.gaussian_head = gaussian_head(
            in_features=int(head_in_dim),
            out_features=self.latent_dim,
        )

    # --------- helpers ----------

    def _ensure_seq(self, z: torch.Tensor) -> torch.Tensor:
        """Ensure z has shape (B, d, j).

        Accepts:
          - (B, d) for j==1
          - (B, d, j)
          - (B, j, d)
        """
        if z.dim() == 2:
            B, d = z.shape
            assert self.j == 1, "z with no history dimension requires j==1"
            assert d == self.latent_dim
            return z.unsqueeze(-1)  # (B, d, 1)
        if z.dim() == 3:
            B, a, b = z.shape
            # try to infer if shape is (B,d,j) or (B,j,d)
            if a == self.latent_dim and b == self.j:
                return z  # (B,d,j)
            if a == self.j and b == self.latent_dim:
                return z.permute(0, 2, 1)  # (B,d,j)
        raise ValueError(
            f"z_hist must be (B,d) or (B,d,j) or (B,j,d); got {tuple(z.shape)}"
        )

    def prior_params(
        self,
        z_hist: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute p_mu, p_logvar for p(z_t | z_hist, ctx).

        Expects:
            z_hist: (B, d, j) or equivalent
            ctx (optional):
                - "hist_time_emb": (B, j, E_t) absolute time embeddings
                  If not provided, zeros are used.
        """
        z_hist = self._ensure_seq(z_hist)  # (B, d, j)
        B, d, j = z_hist.shape
        assert d == self.latent_dim and j == self.j

        device = z_hist.device
        dtype = z_hist.dtype

        # project z history
        # (B, d, j) -> (B, j, d) -> (B, j, H) -> (B, H, j)
        z_seq = z_hist.permute(0, 2, 1)  # (B, j, d)
        z_proj = self.z_hist_proj(z_seq)  # (B, j, H)
        combined = z_proj.permute(0, 2, 1)  # (B, H, j)

        # time embeddings for history: (B, j, E_t) -> (B, E_t, j)
        if ctx is not None and "hist_time_emb" in ctx:
            hist_time_emb = ctx["hist_time_emb"]
            assert hist_time_emb.shape == (B, self.j, self.emb_time_dim)
        else:
            # fallback: zeros if no explicit time conditioning given
            hist_time_emb = torch.zeros(
                B, self.j, self.emb_time_dim, device=device, dtype=dtype
            )

        if ctx is not None and "hist_covariates" in ctx:
            hist_covs = ctx["hist_covariates"]  # (B, j, V)
            assert hist_covs.shape == (B, self.j, self.covariate_dim)
            hist_time_emb = torch.cat([hist_time_emb, hist_covs], dim=-1)

        hist_time_emb = hist_time_emb.permute(0, 2, 1)  # (B, E_t or E_t+V, j)

        # dummy mask embedding: none needed, but ContextProducer expects a tensor
        # with shape (B, 0, j) when mask_tot_dim=0.
        mask_embedded = torch.zeros(
            B, 0, self.j, device=device, dtype=dtype
        )  # (B, 0, j)

        # context token (via __call__ so the in-place torch.compile fires)
        x = self.context_producer(
            combined=combined,
            mask_embedded=mask_embedded,
            hist_time_emb=hist_time_emb,
        )  # (B, C * (H + E_t))

        # Gaussian parameters
        mu, logvar = self.gaussian_head(x)  # both (B, d)

        return mu, logvar

    def log_prob(
        self,
        z: torch.Tensor,
        z_hist: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None,
        mc_override: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Log p(z | z_hist, ctx) summed over latent dims.

        Shapes:
            z:
                - (B, d)      -> returns (B,)
                - (B, S, d)   -> returns (B, S)
            z_hist: (B, d, j) or equivalent (see _ensure_seq)
        """
        B = z_hist.shape[0]
        p_mu, p_logvar = self.prior_params(z_hist, ctx=ctx)  # (B, d)
        assert p_mu.shape == p_logvar.shape == (B, self.latent_dim)

        if z.dim() == 2:
            # (B, d) -> add sample axis
            assert z.shape == (B, self.latent_dim)
            z_exp = z.unsqueeze(1)  # (B, 1, d)
            p_mu_exp = p_mu.unsqueeze(1)  # (B, 1, d)
            p_logvar_exp = p_logvar.unsqueeze(1)  # (B, 1, d)
            squeeze_result = True
        elif z.dim() == 3:
            # (B, S, d)
            Bz, S, d = z.shape
            assert Bz == B and d == self.latent_dim
            z_exp = z
            p_mu_exp = p_mu.unsqueeze(1).expand(B, S, d)
            p_logvar_exp = p_logvar.unsqueeze(1).expand(B, S, d)
            squeeze_result = False
        else:
            raise ValueError(f"z must be (B,d) or (B,S,d); got shape {tuple(z.shape)}")

        var = p_logvar_exp.exp()
        diff = z_exp - p_mu_exp
        lp_per_dim = -0.5 * (diff * diff / var + p_logvar_exp + math.log(2 * math.pi))
        log_p = lp_per_dim.sum(dim=-1)  # (B, 1) or (B, S)

        if squeeze_result:
            return log_p.squeeze(1)  # (B,)
        return log_p  # (B, S)

    def _score_init_step(
        self,
        *,
        step: int,
        z_t: torch.Tensor,            # (BS, d)
        z_hist: torch.Tensor,         # (BS, d, j)
        enc_stats: GaussianStats,
        time_embed: torch.Tensor,     # (B, T, E_t)
        sigma_data: "Optional[SigmaDataBuffer]",
        B: int,
        S: int,
        T: int,
    ) -> torch.Tensor:
        """Closed-form ``-log p_ψ(z_t | z_hist)`` for one init step (summed B·S).

        Scores the encoder sample ``z_t`` under the Gaussian prior
        ``p_ψ(·|z_hist)`` (same head as the ``t>j`` term). No σ_data — the
        plain Gaussian transition does not use EDM preconditioning; the
        base ``_init_entropy_term`` default adds ``-H(q_φ)``.
        """
        del sigma_data  # plain Gaussian transition does not track σ_data
        ctx_step = self._init_step_time_ctx(step, time_embed, B, S, T)
        log_p = self.log_prob(z_t, z_hist, ctx=ctx_step)  # (BS,)
        return (-log_p).sum()

    def transition_kl(
        self,
        enc_stats: GaussianStats,
        zs: torch.Tensor,  # (B, S, d, T)
        logq_paths: torch.Tensor,  # (B, S, T)
        time_embed: torch.Tensor,  # (B, T, E_t)
        sigma_data: "Optional[SigmaDataBuffer]" = None,  # accepted for the uniform interface; unused
        covariates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Transition KL term for the Gaussian prior.

        Two paths:
          * Closed-form (when ``enc_stats`` provides Gaussian ``mus``/``logvars``):
            compute ``KL(q(z_t|x) || p_psi(z_t|z_{t-j:t-1}))`` per time step
            using :func:`gaussian_kl_divergence`.  Returned dict contains only
            ``"kl"`` since the closed-form quantity does not naturally split
            into separately loggable ``L_p``/``L_q`` parts.
          * Monte-Carlo fallback: KL = ``-E_q[log p_psi] - H[q]`` estimated from
            ``zs`` and ``logq_paths``.  Returned dict contains ``"kl"``,
            ``"L_p"`` (negative prior log-prob) and ``"L_q"`` (encoder MC entropy).
        """
        B, S, d, T = zs.shape
        j = self.j
        device = zs.device
        dtype = zs.dtype

        if (
            "mus" in enc_stats
            and "logvars" in enc_stats
            and enc_stats["mus"] is not None
            and enc_stats["logvars"] is not None
        ):
            mus = enc_stats["mus"]  # (B, S, d, T)
            logvars = enc_stats["logvars"]  # (B, S, d, T)
            assert mus.shape == logvars.shape == (B, S, d, T)

            total_kl = torch.zeros(B, device=device, dtype=dtype)
            total_steps = T - j
            if total_steps > 0:
                for (
                    B_,
                    S_,
                    current_chunk_len,
                    t_start,
                    t_end,
                    _z_target_flat,
                    z_hist_flat,
                    ctx,
                ) in self._iter_window_chunks(
                    zs, time_embed, covariates=covariates,
                ):
                    # prior params for the chunk's targets, shape (N, d) where
                    # N = B*S*chunk_len
                    p_mu, p_logvar = self.prior_params(z_hist_flat, ctx=ctx)

                    # encoder stats slices for the same targets:
                    # (B, S, d, chunk_len) -> (B, S, chunk_len, d) -> (N, d)
                    q_mu = (
                        mus[..., t_start:t_end]
                        .permute(0, 1, 3, 2)
                        .reshape(-1, d)
                    )
                    q_logvar = (
                        logvars[..., t_start:t_end]
                        .permute(0, 1, 3, 2)
                        .reshape(-1, d)
                    )

                    kl_flat = gaussian_kl_divergence(
                        q_mu, q_logvar, p_mu, p_logvar
                    )  # (N,)
                    kl = kl_flat.view(B_, S_, current_chunk_len)
                    chunk_kl = kl.sum(dim=2).mean(dim=1)  # (B,)
                    total_kl = total_kl + chunk_kl

            return {"kl": total_kl.mean()}

        # MC fallback
        L_p = -self.seq_log_prob(
            zs=zs, time_embed=time_embed, covariates=covariates
        ).mean()
        L_q = _mc_entropy_from_logq(logq_paths, j)
        return {"kl": L_p - L_q, "L_p": L_p, "L_q": L_q}

    def sample(
        self,
        z_hist: torch.Tensor,
        S: int = 1,
        ctx: Optional[Dict[str, Any]] = None,
        hist_valid_len: Optional[torch.Tensor] = None,  # ignored, kept for API
    ) -> torch.Tensor:
        """Draw from p(z_t | z_hist). Returns (B, S, d)."""
        mu, logvar = self.prior_params(z_hist, ctx=ctx)
        sigma = (0.5 * logvar).exp()
        B, d = mu.shape
        eps = torch.randn(B, S, d, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(1) + sigma.unsqueeze(1) * eps


