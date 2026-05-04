"""Define different transition functions for use in DDSSM models."""

"""
Pluggable transition modules.

Interface:
- BaseTransition.loss(z_samples, z_hist, ctx=None, hist_valid_len=None, reduction='mean') -> Tensor
    z_samples: (B, S, d) or (B, d)  -- samples drawn from encoder (S optional)
    z_hist:    (B, d, j) or (B, j, d) or (B, d) if j==1
    ctx:       optional dict for extra conditioning (e.g. {'hist_time_emb': tensor})
    hist_valid_len: (B,) optional ints for masking history when t < j
    reduction: 'mean'|'sum'|'none'
- BaseTransition.prior_params and .log_prob are optional helpers for diagnostics.

Concrete: GaussianTransition (non-linear diagonal Gaussian).
"""

import math
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn


from ..config import TransitionConfig
from ..encoder import GaussianHead, ContextProducer


class BaseTransition(nn.Module):
    """Abstract transition interface."""

    def prior_params(
        self, z_hist: torch.Tensor, ctx: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (p_mu, p_logvar) conditioned on z_hist/context.

        Optional helper for diagnostics. Implementors may raise NotImplementedError.
        """
        raise NotImplementedError

    def seq_log_prob(
        self,
        zs: torch.Tensor,  # (B, S, d, T)
        time_embed: torch.Tensor,  # (B, T, E_t)
        time_chunk_num: Optional[int] = None,  #
        time_chunk_size: Optional[int] = None,
        covariates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute : sum_{t=j}^{T-1} E_q[ log p_psi(z_t | z_{t-j:t-1}) ]
        ( or a bound)

        TODO: just compute over the full sequence, caller should specify the
        time indices, ideally

        Args:
            zs: (B, S, d, T) latent samples from encoder
            time_embed: (B, T, E_t) absolute time embeddings
            time_chunk_num: number of chunks to split time dimension into
            time_chunk_size: size of each chunk along time dimension. Overrides time_chunk_num if given.


        Returns: (B,) total log prob per batch element
        """
        B, S, d, T = zs.shape
        j = self.j
        device = zs.device

        total_steps = T - j
        if total_steps <= 0:
            return torch.zeros(B, device=device)

        # determine chunk size
        if time_chunk_size is not None and time_chunk_size > 0:
            chunk_size = time_chunk_size
        elif time_chunk_num is not None and time_chunk_num > 0:
            chunk_size = math.ceil(total_steps / time_chunk_num)
        else:
            chunk_size = 1

        total_nll = torch.zeros(B, device=zs.device)

        # Iterate over chunks
        for start_rel in range(0, total_steps, chunk_size):
            end_rel = min(start_rel + chunk_size, total_steps)

            # Absolute time indices for targets
            t_start = j + start_rel
            t_end = j + end_rel
            current_chunk_len = t_end - t_start

            # (B, S, d, chunk_len)
            z_target_chunk = zs[..., t_start:t_end]

            BS_chunk = B * S * z_target_chunk.size(-1)

            # We need a sliding window view of zs.
            # Source range needed: [t_start - j, t_end - 1]
            # Length needed: (t_end - 1) - (t_start - j) + 1 = t_end - t_start + j = chunk_len + j

            zs_source = zs[..., t_start - j : t_end]  # (B, S, d, chunk_len + j)

            # Unfold to get sliding windows of size j
            # (B, S, d, num_windows, j)
            # num_windows should be chunk_len + 1, we take the first chunk_len
            z_hist_chunk = zs_source.unfold(dimension=-1, size=j, step=1)
            z_hist_chunk = z_hist_chunk[..., :current_chunk_len, :]

            # Source range: [t_start - j, t_end - 1]
            t_emb_source = time_embed[
                :, t_start - j : t_end, :
            ]  # (B, chunk_len + j, E)
            # need (B, num_windows, E, j)
            t_hist_chunk = t_emb_source.unfold(dimension=1, size=j, step=1)
            t_hist_chunk = t_hist_chunk[:, :current_chunk_len, :, :]

            # treat (B * S * chunk_len) as the batch dimension

            # z_target: (B, S, d, chunk_len) -> (B, S, chunk_len, d) -> (N, d)
            z_target_flat = z_target_chunk.permute(0, 1, 3, 2).reshape(-1, d)

            # z_hist: (B, S, d, chunk_len, j) -> (B, S, chunk_len, d, j) -> (N, d, j)
            z_hist_flat = z_hist_chunk.permute(0, 1, 3, 2, 4).reshape(-1, d, j)

            # t_hist: (B, chunk_len, E, j) -> (B, chunk_len, j, E) -> expand S -> (N, j, E)
            # Note: BaseTransition.prior_params expects ctx["hist_time_emb"] as (N, j, E)
            t_hist_chunk = t_hist_chunk.permute(0, 1, 3, 2)  # (B, chunk_len, j, E)
            t_hist_flat = (
                t_hist_chunk
                .unsqueeze(1)
                .expand(-1, S, -1, -1, -1)
                .reshape(BS_chunk, j, self.emb_time_dim)
            )

            if covariates is not None:
                c_emb_source = covariates[
                    :, :, t_start - j : t_end
                ]  # (B, V, chunk_len + j)
                c_hist_chunk = c_emb_source.unfold(dimension=2, size=j, step=1)
                c_hist_chunk = c_hist_chunk[:, :, :current_chunk_len, :]
                c_hist_chunk = c_hist_chunk.permute(0, 2, 3, 1)  # (B, chunk_len, j, V)
                c_hist_flat = (
                    c_hist_chunk
                    .unsqueeze(1)
                    .expand(-1, S, -1, -1, -1)
                    .reshape(BS_chunk, j, covariates.size(1))
                )

                c_target_chunk = covariates[:, :, t_start:t_end]  # (B, V, chunk_len)
                c_target_chunk = c_target_chunk.permute(0, 2, 1)  # (B, chunk_len, V)
                c_target_flat = (
                    c_target_chunk
                    .unsqueeze(1)
                    .expand(-1, S, -1, -1)
                    .reshape(BS_chunk, 1, covariates.size(1))
                )
            else:
                c_hist_flat = None
                c_target_flat = None

            t_target_chunk = time_embed[:, t_start:t_end, :]  # (B, chunk_len, E)
            t_target_flat = (
                t_target_chunk
                .unsqueeze(1)
                .expand(-1, S, -1, -1)
                .reshape(BS_chunk, 1, self.emb_time_dim)
            )

            ctx = {"hist_time_emb": t_hist_flat, "target_time_emb": t_target_flat}
            if c_hist_flat is not None:
                ctx["hist_covariates"] = c_hist_flat
            if c_target_flat is not None:
                ctx["target_covariates"] = c_target_flat

            log_p_flat = self.log_prob(z=z_target_flat, z_hist=z_hist_flat, ctx=ctx)
            # Reshape to (B, S, chunk_len)
            log_p = log_p_flat.view(B, S, current_chunk_len)

            # Sum over time in chunk, Mean over S
            # (B,)
            chunk_nll = log_p.sum(dim=2).mean(dim=1)
            total_nll += chunk_nll

        return total_nll

    def loss(
        self,
        zs: torch.Tensor,  # (B, S, d, T)
        time_embed: torch.Tensor,  # (B, T, E_t)
        reduction: str = "mean",
        covariates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute transition loss over sequence:
        E_q[ sum_{t=j}^{T-1} -log p_psi(z_t | z_{t-j:t-1}) ]
        Returns: scalar (mean over B, mean over S, sum over T)
        """
        return -self.seq_log_prob(
            zs=zs, time_embed=time_embed, covariates=covariates
        ).mean()

    def log_prob(
        self,
        z: torch.Tensor,
        z_hist: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None,
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
        S: int = 1,  # num trajs
        ctx: Optional[Dict[str, Any]] = None,
    ):
        """Draw samples from p(z_t:t+steps 1 | z_hist, ctx)

        Args:
         S: number of trajectories to sample
         steps: length of sampled trajectory

        Return:

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
    """Non-linear diagonal Gaussian transition p(z_t | z_{t-j:t-1}, ...).

    Refactored to use the same internal architecture as InitPrior/Decoder:
      - z history -> ContextProducer -> GaussianHead
    No padding mask and no collect_samples.
    """

    def __init__(
        self,
        transition_config: TransitionConfig,
        latent_dim: int,
        j: int,
        emb_time_dim: int,
        covariate_dim: int = 0,
    ) -> None:
        """Args:
        latent_dim: latent dimension d
        j: history length
        transition_config: TransitionConfig to set hidden_dim/context/gaussian_head
        emb_time_dim: time embedding dimension E_t
        covariate_dim: covariate dimension V
        """
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.j = int(j)
        self.emb_time_dim = int(emb_time_dim)
        self.covariate_dim = int(covariate_dim)

        self.hidden_dim = transition_config.hidden_dim  # H

        # We reuse the same Gaussian head config defined for the transition
        self.gaussian_head_config = transition_config.gaussian_head

        # Project z history: d -> H
        self.z_hist_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        self.config = transition_config

        # ContextProducer over length j, with no explicit mask features
        # (mask_tot_dim=0, but ContextProducer still expects some tensor)
        self.context_producer = ContextProducer(
            config=transition_config.context,
            combined_dim=self.hidden_dim,
            mask_tot_dim=0,
            emb_time_dim=self.emb_time_dim + self.covariate_dim,
            combined_len=self.j,
        )

        self.context_producer = torch.compile(self.context_producer, dynamic=True)

        # Gaussian head over flattened context
        # tot_dim = H + E_t + 0
        head_in_dim = self.config.context.channels * self.hidden_dim

        self.gaussian_head = GaussianHead(
            in_features=int(head_in_dim),
            out_features=self.latent_dim,
            config=self.gaussian_head_config,
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

        # context token
        x = self.context_producer.forward(
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
