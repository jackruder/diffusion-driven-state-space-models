import math
from typing import Any, Dict, Optional, final

import torch
import torch.nn as nn

from ..windows import WindowBuilder

from ..config import (
    DDSSMConfig,
    DiffusionScheduleConfig,
    DiffusionTransitionConfig,
)
from ..diffnets import CSDIUnet
from ..net_utils import (
    get_side_info,
)
from .transitions import BaseTransition



@final
class DiffusionTransition(BaseTransition):
    """Diffusion-based transition model p(z_t | z_{t-j:t-1}).

    Wraps CSDIUnet and implements BaseTransition interface.
    """

    def __init__(
        self,
        transition_config: DiffusionTransitionConfig,
        latent_dim: int,
        j: int,
        emb_time_dim: int,
        device: torch.device,
        covariate_dim: int = 0,
    ) -> None:
        super().__init__()
        self.config = transition_config

        self.j = j
        self.latent_dim = latent_dim

        self.emb_time_dim = emb_time_dim
        self.covariate_dim = covariate_dim

        # TODO : change if we remove feature embeddings
        # TODO: unsure about feature embedding at the moment, so
        # quick hardcode to avoid another parameter
        self.emb_feature_dim = emb_time_dim
        self.side_dim = (
            self.emb_time_dim + self.covariate_dim + self.emb_feature_dim + 1
        )

        self.schedule = transition_config.schedule

        self.diffmodel = CSDIUnet(
            self.config.unet,
            1,  # predict 1 latent step
            self.schedule.num_steps,
            self.latent_dim,
            self.j,
            self.side_dim,
        )

        self.diffmodel = torch.compile(self.diffmodel)

        self.embed_layer = nn.Embedding(
            num_embeddings=self.latent_dim, embedding_dim=self.emb_feature_dim
        )

        self.S_k = self.schedule.S_k  # number of k-draws per z_t monte-carlo sample
        self.num_steps = self.schedule.num_steps

        dtype64 = torch.float64
        eps64 = torch.finfo(dtype64).eps
        K = self.num_steps
        i = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype64)
        inv_rho = 1.0 / float(self.schedule.rho)
        sigma = (
            self.schedule.sigma_min**inv_rho
            + i * (self.schedule.sigma_max**inv_rho - self.schedule.sigma_min**inv_rho)
        ) ** float(self.schedule.rho)

        sigma = sigma.to(torch.float64)
        sigma2 = sigma * sigma
        # define previous sigma^2, with sigma_{-1} = 0
        sigma2_prev = torch.cat([
            torch.zeros(1, dtype=sigma2.dtype, device=sigma2.device),
            sigma2[:-1],
        ])

        #  tilded ELBO weight purely from sigma
        wtilde = (sigma2 - sigma2_prev) / (
            2.0 * sigma2.clamp_min(eps64) * (1.0 + sigma2_prev)
        )

        alpha_bar = 1.0 / (1.0 + sigma**2)  # (K,)

        c_skip = 1 / (sigma**2 + 1)
        c_out = sigma / torch.sqrt(sigma**2 + 1)
        c_in = 1 / torch.sqrt(sigma**2 + 1)
        c_noise = 0.25 * torch.log(torch.clamp(sigma, min=eps64))

        self.register_buffer("alpha_bar", alpha_bar.to(torch.float32))
        self.register_buffer("wtilde", wtilde.to(torch.float32))
        self.register_buffer("sigma", sigma.to(torch.float32))
        self.register_buffer("c_skip", c_skip.to(torch.float32))
        self.register_buffer("c_out", c_out.to(torch.float32))
        self.register_buffer("c_in", c_in.to(torch.float32))
        self.register_buffer("c_noise", c_noise.to(torch.float32))

        # sampling probabilities for k
        self.gamma = self.schedule.pk_gamma
        self.gfloor = self.schedule.pk_floor
        ismode = self.schedule.k_sampling_mode
        if ismode == "importance":
            p_k = self.wtilde.detach().to(
                device=self.wtilde.device, dtype=torch.float32
            )
            p_k = p_k.clamp_min(1e-12).pow(self.gamma)  # safe base for pow
            p_k = p_k.clamp_min(self.gfloor)  # pre-normalization floor
            p_k = p_k / p_k.sum()
        else:
            p_k = torch.full(
                (self.num_steps,),
                1.0 / self.num_steps,
                device=self.wtilde.device,
                dtype=torch.float32,
            )
        self.register_buffer("p_k", p_k.to(torch.float32))
        self.k_sampling_mode = ismode

    def log_prob(
        self,
        z: torch.Tensor,
        z_hist: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute negative EDM loss for the batch.

        We chunk over S_k to avoid increasing memory
        disproportionately in the diffusion model compared
        to the rest of the SSM.

        Args:
            z: (N, d) target latents
            z_hist: (N, d, j) history latents
            ctx: dict with 'hist_time_emb' (N, j, E) and 'target_time_emb' (N, 1, E)

        Returns:
            (N,) negative loss per item (averaged over S_k draws)
        """
        N, d = z.shape
        device = z.device
        dtype = z.dtype

        # Reconstruct time window for side info: [t-j, ..., t]
        # hist_time_emb: (N, j, E)
        # target_time_emb: (N, 1, E)
        if ctx is None or "hist_time_emb" not in ctx or "target_time_emb" not in ctx:
            raise ValueError(
                "DiffusionTransition.log_prob requires time embeddings in ctx"
            )

        # # --- DIAGNOSTIC: Monitor Latent Scale ---
        # if not hasattr(self, "_diag_counter"):
        #     self._diag_counter = 0
        # self._diag_counter += 1
        #
        # if self._diag_counter % 20 == 0:
        #     z_std = z.std()
        #     z_mean = z.mean()
        #     z_max = z.abs().max()
        #     print(f"\n[DIAGNOSTIC Step {self._diag_counter}]")
        #     print(
        #         f"  Latent z: mean={z_mean:.3f}, std={z_std:.3f}, max_abs={z_max:.3f}"
        #     )
        #     print(f"  Schedule: [{self.schedule.sigma_min}, {self.schedule.sigma_max}]")
        #     if z_std > 2.0:
        #         print(
        #             "  [WARNING] Latent space is expanding. Diffusion schedule expects std~1.0."
        #         )
        # # ----------------------------------------

        hist_time = ctx["hist_time_emb"]
        tgt_time = ctx["target_time_emb"]

        if "hist_covariates" in ctx:
            hist_time = torch.cat([hist_time, ctx["hist_covariates"]], dim=-1)
        if "target_covariates" in ctx:
            tgt_time = torch.cat([tgt_time, ctx["target_covariates"]], dim=-1)

        time_win = torch.cat([hist_time, tgt_time], dim=1)  # (N, j+1, E+V)

        # Create conditioning mask: 1 for history (j), 0 for target (1)
        # Shape: (N, d, j+1)
        cond_mask = torch.ones(N, d, self.j + 1, device=device, dtype=dtype)
        cond_mask[..., -1] = 0.0

        # Build side info: (N, C_side, d, j+1)
        side_win = get_side_info(
            data_dim=self.latent_dim,
            time_embed=time_win,
            embed_layer=self.embed_layer,
            cond_mask=cond_mask,
            device=device,
        )

        # Config for chunking S_k
        k_chunk = self.schedule.k_chunk
        k_chunk = max(1, min(int(k_chunk), int(self.S_k)))

        # setting k_chunk=1 will avoid chunking and use the full S_k at once,
        # setting k_chunk=S_k will do all S_k draws in one go (if memory allows)

        total_sqerr = torch.zeros(N, device=device, dtype=dtype)

        # Iterate over S_k in chunks
        remaining_k = int(self.S_k)
        while remaining_k > 0:
            kc = min(k_chunk, remaining_k)
            remaining_k -= kc

            # Sample k indices and noise
            k_idx = torch.multinomial(self.p_k, N * kc, replacement=True).view(
                N, kc
            )  # (N, kc)
            eps = torch.randn(N, d, kc, device=device, dtype=dtype)

            # EDM preconditioning
            _, z_in, y_target = self._edm_precondition(z, k_idx, eps)  # (N, d, kc) each

            # Assemble latent window: concat [hist, current] along time axis
            # z_hist: (N, d, j) -> (N, d, j, kc)
            z_hist_rep = z_hist.unsqueeze(-1).expand(N, d, self.j, kc)
            # z_in: (N, d, kc) -> (N, d, 1, kc)
            z_in_exp = z_in.unsqueeze(2)

            latent_w = torch.cat([z_hist_rep, z_in_exp], dim=2)  # (N, d, j+1, kc)
            # Flatten for batch processing: (N*kc, d, j+1)
            latent_w = (
                latent_w.permute(0, 3, 1, 2).reshape(N * kc, d, self.j + 1).contiguous()
            )

            # Side window: tile over kc
            # side_win: (N, C_side, d, j+1) -> (N*kc, C_side, d, j+1)
            side_w = (
                side_win
                .unsqueeze(1)
                .expand(N, kc, -1, -1, -1)
                .reshape(N * kc, self.side_dim, d, self.j + 1)
                .contiguous()
            )

            # c_noise and weights
            k_flat = k_idx.reshape(N * kc)
            c_noise_flat = self.c_noise[k_flat]  # (N*kc,)
            weights = (
                self.wtilde[k_flat] / self.p_k[k_flat].clamp_min(1e-12)
            ).detach()  # (N*kc,)

            # Forward pass
            y_pred = self.diffmodel(latent_w, side_w, c_noise_flat)  # (N*kc, d, 1)
            y_pred = y_pred.squeeze(-1)  # (N*kc, d)
            y_flat = y_target.permute(0, 2, 1).reshape(N * kc, d)  # (N*kc, d)

            # Squared error per item (sum over d)
            sqerr = (y_pred - y_flat).pow(2).sum(dim=1) * weights  # (N*kc,)

            # Accumulate back to (N,)
            sqerr_n = sqerr.view(N, kc).sum(dim=1)
            total_sqerr += sqerr_n

        # Average over S_k
        avg_loss = total_sqerr / self.S_k

        fwd_kl = self.forward_kl_loss(z)  # (N,)

        # Return negative loss (since seq_log_prob sums log probs)
        return -(avg_loss + fwd_kl)

    def sample(
        self,
        z_hist: torch.Tensor,
        S: int = 1,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Draw samples from p(z_t | z_hist).

        Args:
            z_hist: (B, d, j)
            S: number of samples (ignored, returns 1 sample per batch item currently)
            ctx: dict containing 'hist_time_emb' (B, j, E) and optionally 'target_time_emb' (B, 1, E)
                 or 'time_embed_window' (B, j+1, E).

        Returns:
            (B, S, d)
        """
        B, d, j = z_hist.shape
        device = z_hist.device

        # Construct side_win: (B, C_side, d, j+1)
        # We need time embeddings for [t-j, ..., t]
        time_win = None
        if ctx is not None:
            if "time_embed_window" in ctx:
                time_win = ctx["time_embed_window"]  # (B, j+1, E)
            elif "hist_time_emb" in ctx:
                hist_emb = ctx["hist_time_emb"]  # (B, j, E)
                if "hist_covariates" in ctx:
                    hist_emb = torch.cat([hist_emb, ctx["hist_covariates"]], dim=-1)

                if "target_time_emb" in ctx:
                    tgt_emb = ctx["target_time_emb"]  # (B, 1, E)
                    if "target_covariates" in ctx:
                        tgt_emb = torch.cat([tgt_emb, ctx["target_covariates"]], dim=-1)
                    time_win = torch.cat([hist_emb, tgt_emb], dim=1)
                else:
                    raise ValueError(
                        "DiffusionTransition.sample requires target time embedding in ctx"
                    )

        if time_win is None:
            raise ValueError("DiffusionTransition.sample requires time embeddings")

        # Create conditioning mask: 1 for history, 0 for target
        cond_mask = torch.ones(B, d, self.j + 1, device=device, dtype=z_hist.dtype)
        cond_mask[..., -1] = 0.0
        # Build side info
        side_win = get_side_info(
            data_dim=self.latent_dim,
            time_embed=time_win,
            embed_layer=self.embed_layer,
            cond_mask=cond_mask,
            device=device,
        )  # (B, C_side, d, j+1)

        # Sample
        # edm_sample_latent returns (B, d)
        z_sample = self.edm_sample_latent(
            z_hist=z_hist,
            side_win=side_win,
            # Use default sampling params or from config if available
        )
        return z_sample.unsqueeze(1)  # (B, 1, d)

    def _edm_precondition(
        self,
        z_t: torch.Tensor,  # (N, d)          clean z_t^0 flattened over N=B·S (or any N)
        k_idx: torch.Tensor,  # (N, S_k)        diffusion step indices per row
        eps: torch.Tensor,  # (N, d, S_k)     the same ε used to make the noisy samples
    ):
        r"""EDM preconditioning for diffusion training with flattened inputs.

        Shapes
        ------
        Let N be any batch size (e.g., N = B·S), d the latent dimension, and S_k the
        number of k-draws per example.

          • z_t       ∈ ℝ^{N×d}              : clean latent z_t^0 per row
          • k_idx     ∈ ℕ^{N×S_k}            : indices into the K-step schedule
          • nps       ∈ ℝ^{N×d×S_k}          : standard Gaussian noise used for noising


        we construct
            ẑ^{(k)}   = z_t + σ_k · ε                        ∈ ℝ^{N×d×S_k}
            z_in      = c_in(σ_k) · ẑ^{(k)}                  ∈ ℝ^{N×d×S_k}
            y_target  = (z_t − c_skip(σ_k)·ẑ^{(k)}) / c_out(σ_k)  ∈ ℝ^{N×d×S_k}

        Returns:
        -------
        z_hat    : torch.Tensor, shape (N, d, S_k)
            The preconditioned noisy inputs ẑ^{(k)}.

        z_in     : torch.Tensor, shape (N, d, S_k)
            The model input after EDM scaling, c_in(σ_k) · ẑ^{(k)}.

        y_target : torch.Tensor, shape (N, d, S_k)
            The regression target for F.

        """
        # Gather per-(n,k) coefficients from schedule buffers
        sigma = self.sigma[k_idx]  # (N, S_k)
        c_skip = self.c_skip[k_idx]  # (N, S_k)
        c_out = self.c_out[k_idx]  # (N, S_k)
        c_in = self.c_in[k_idx]  # (N, S_k)

        # ẑ^{(k)} = z_t + σ_k ε
        z_hat = z_t.unsqueeze(-1) + sigma.unsqueeze(1) * eps  # (N, d, S_k)

        # z_in = c_in(σ_k) · ẑ^{(k)}
        z_in = c_in.unsqueeze(1) * z_hat  # (N, d, S_k)

        # y = (z_t − c_skip(σ_k)·ẑ^{(k)}) / c_out(σ_k)
        denom = c_out.clamp_min(1e-12).unsqueeze(1)  # (N, 1, S_k)
        y_target = (
            z_t.unsqueeze(-1) - c_skip.unsqueeze(1) * z_hat
        ) / denom  # (N, d, S_k)

        return z_hat, z_in, y_target

    def _build_side_chunk_flat(
        self,
        time_embed: torch.Tensor,  # (B, T, E_t)
        t0: int,
        t1: int,
        *,
        B: int,
        d: int,
    ) -> torch.Tensor:
        """Side info for forecasting timesteps [t0..t1) (t0,t1 measured in the T-j domain).
        Returns (B*(t1-t0), C_side, d, j+1) without expanding over S.
        """
        j = self.j
        device = time_embed.device
        # per-time side info (no S yet): (B, C_side, d, T)

        cond_mask = torch.ones(B, d, self.j + 1, device=device)
        cond_mask[..., -1] = 0.0
        side_per_t = get_side_info(
            data_dim=self.latent_dim,
            time_embed=time_embed,
            embed_layer=self.embed_layer,
            cond_mask=cond_mask,
            device=self.device,
        )  # (B, C_side, d, T)

        # get all the (j+1)-length windows ending at absolute times t_abs = j + t_local
        wins = []
        for t_local in range(t0, t1):
            t_end = j + t_local  # absolute index in [0..T-1]
            wins.append(side_per_t[..., t_end - j : t_end + 1])  # (B, C_side, d, j+1)
        side_btcdl = torch.stack(wins, dim=1)  # (B, t_len, C_side, d, j+1)
        t_len = side_btcdl.shape[1]
        side_flat = side_btcdl.permute(0, 1, 2, 3, 4).reshape(
            B * t_len, *side_btcdl.shape[2:]
        )
        # (B*t_len, C_side, d, j+1)
        return side_flat.contiguous()

    # def diffusion_forward_and_loss(
    #     self,
    #     zs: torch.Tensor,  # (B, S, d, T)
    #     time_embed: torch.Tensor,  # (B, T, E_t)
    # ):
    #     """Run the diffusion U-Net over all forecasting steps (t = j..T-1) with S_k draws
    #     and compute the EDM regression loss against the target y.
    #
    #     This chunks over the data.
    #
    #     Notation & Shapes
    #     -----------------
    #     Let N = B·S·(T−j) and M = N·S_k. For all valid t (j..T-1):
    #       • z_in_all     ∈ ℝ^{N×d×S_k}     : EDM-preconditioned current latent (c_in·ẑ)
    #       • z_hist_all   ∈ ℝ^{N×d×j×S_k}   : left-padded latent history z_{t-j:t-1}, repeated over S_k
    #       • y_target_all ∈ ℝ^{N×d×S_k}     : EDM regression target
    #       • k_idx        ∈ ℕ^{N×S_k}       : diffusion step indices
    #       • side_all     ∈ ℝ^{N×C_side×d×(j+1)} : side info window over [t-j, …, t]
    #
    #     We assemble:
    #       • latent_window ∈ ℝ^{M×d×(j+1)}       : concat [z_hist, z_in] on the window axis
    #       • side_window   ∈ ℝ^{M×C_side×d×(j+1)}
    #       • c_noise_flat  ∈ ℝ^{M}               : EDM scalar per sample (from schedule)
    #
    #     Returns:
    #     -------
    #     loss : scalar tensor
    #     sqerr_per_item : (M,) tensor, squared error summed over latent dim d
    #     """
    #     # --- config for chunking (use config if present, else fallbacks) ---
    #     t_chunk = self.config.hyperparams.t_chunk
    #     k_chunk = self.config.hyperparams.k_chunk
    #     k_chunk = max(1, min(int(k_chunk), int(self.S_k)))
    #
    #     B, S, d, T = zs.shape
    #     j = self.j
    #     Tk = T - j
    #     assert Tk > 0, f"No forecasting steps: T={T}, j={j}"
    #
    #     device = zs.device
    #     dtype = zs.dtype
    #
    #     # set default full-span chunk if not provided
    #     if not t_chunk or t_chunk <= 0:
    #         t_chunk = Tk
    #
    #     # running numerator for: loss = (1/(B*S*S_k)) * sum_{b,s,k} sum_{t} sqerr(b,s,t,k)
    #     loss_numer = torch.zeros((), device=device, dtype=dtype)
    #
    #     # iterate over time chunks
    #     for t0 in range(0, Tk, t_chunk):
    #         t1 = min(Tk, t0 + t_chunk)
    #         t_len = t1 - t0
    #
    #         # side info for this time slice (no S here): (B*t_len, C_side, d, j+1)
    #         side_bt_flat = self._build_side_chunk_flat(time_embed, t0, t1, B=B, d=d)
    #         C_side = side_bt_flat.shape[1]
    #
    #         # loop over S to avoid expanding across S at once. S usually is 1 so no biggie
    #         for s_idx in range(S):
    #             # slice latents for times t=j+t0..j+t1-1
    #             z_t_bdt = zs[:, s_idx, :, j + t0 : j + t1]  # (B, d, t_len)
    #             z_hist_bdj_t = wb.z_win[
    #                 :, s_idx, :, :, (j - 1) + t0 : (j - 1) + t1
    #             ]  # (B, d, j, t_len)
    #
    #             # flatten (B, t_len) -> (B*t_len) for this S-path
    #             z_t_flat = z_t_bdt.permute(0, 2, 1).reshape(B * t_len, d)  # (N_bs, d)
    #             z_hist_flat = z_hist_bdj_t.permute(0, 3, 1, 2).reshape(
    #                 B * t_len, d, j
    #             )  # (N_bs, d, j)
    #             side_flat = side_bt_flat  # (B*t_len, C_side, d, j+1), already built
    #
    #             N_bs = z_t_flat.shape[0]
    #
    #             # iterate over S_k in smaller groups
    #             remaining_k = int(self.S_k)
    #             while remaining_k > 0:
    #                 kc = min(k_chunk, remaining_k)
    #                 remaining_k -= kc
    #
    #                 # sample k indices and noise for this small group
    #                 k_idx = torch.multinomial(
    #                     self.p_k, N_bs * kc, replacement=True
    #                 ).view(N_bs, kc)  # (N_bs, kc)
    #                 eps = torch.randn(N_bs, d, kc, device=device, dtype=dtype)
    #
    #                 # EDM preconditioning (vectorized over this chunk)
    #                 _, z_in, y_target = self._edm_precondition(
    #                     z_t_flat, k_idx, eps
    #                 )  # (N_bs, d, kc) each
    #
    #                 # assemble latent window: concat [hist, current] along time axis
    #                 z_hist_rep = z_hist_flat.unsqueeze(-1).expand(
    #                     N_bs, d, j, kc
    #                 )  # (N_bs, d, j, kc)
    #                 z_in_exp = z_in.unsqueeze(2)  # (N_bs, d, 1, kc)
    #                 latent_w = torch.cat(
    #                     [z_hist_rep, z_in_exp], dim=2
    #                 )  # (N_bs, d, j+1, kc)
    #                 # (N_bs*kc, d, j+1)
    #                 latent_w = (
    #                     latent_w.permute(0, 3, 1, 2)
    #                     .reshape(N_bs * kc, d, j + 1)
    #                     .contiguous()
    #                 )
    #
    #                 # side window: tile over kc draws for this chunk
    #                 side_w = (
    #                     side_flat.unsqueeze(1)
    #                     .expand(N_bs, kc, C_side, d, j + 1)
    #                     .reshape(N_bs * kc, C_side, d, j + 1)
    #                     .contiguous()
    #                 )
    #
    #                 # c_noise and weights per sample
    #                 k_flat = k_idx.reshape(N_bs * kc)
    #                 c_noise_flat = self.c_noise[k_flat]  # (N_bs*kc,)
    #                 weights = (
    #                     self.wtilde[k_flat] / self.p_k[k_flat].clamp_min(1e-12)
    #                 ).detach()  # (N_bs*kc,)
    #
    #                 # forward
    #                 y_pred = self.diffmodel(
    #                     latent_w, side_w, c_noise_flat
    #                 )  # (N_bs*kc, d, 1)
    #                 y_pred = y_pred.squeeze(-1)  # (N_bs*kc, d)
    #                 y_flat = y_target.permute(0, 2, 1).reshape(
    #                     N_bs * kc, d
    #                 )  # (N_bs*kc, d)
    #
    #                 # per-item squared error * weight
    #                 sqerr = (y_pred - y_flat).pow(2).sum(dim=1) * weights  # (N_bs*kc,)
    #
    #                 # map back to (B, t_len, kc), sum over time in this chunk
    #                 sqerr_btk = sqerr.view(B, t_len, kc)
    #                 loss_numer = (
    #                     loss_numer + sqerr_btk.sum(dim=1).sum()
    #                 )  # sum_t, then sum_{b,k}
    #                 # accumulate across s_idx loop and k-chunks
    #
    #     # final average over B, S, S_k (time already summed)
    #     denom = float(B * S * self.S_k)
    #     loss = loss_numer / denom
    #     return loss

    def forward_kl_loss(self, z0: torch.Tensor) -> torch.Tensor:
        r"""Forward KL term:
            L_fwd = KL(q(z_t^K | z_t^0) || p(z_t^K))

        For the standard forward noising
            q(z_t^K | z_t^0) = N( sqrt(ᾱ_K)·z_t^0, (1-ᾱ_K) I ),
            p(z_t^K) = N(0, I),
        the KL is:
            0.5 * [ ᾱ_K ||z_t^0||^2 - d ᾱ_K - d log(1-ᾱ_K) ].

        Args:
           z0: (N, d) Clean target latents z_t^0

        Returns:
           (N,) KL divergence per sample
        """
        d = z0.shape[-1]
        z_sq = z0.pow(2).sum(dim=-1)  # (N,) ||z_t^0||^2

        # Schedule scalar ᾱ_K (the terminal cumulative alpha)
        alpha_bar_K = self.alpha_bar[-1].to(z0.dtype)  # scalar
        # numerical guard for log(1 - ᾱ_K)
        one_minus = (1.0 - alpha_bar_K).clamp_min(torch.finfo(z0.dtype).eps)

        # Constant term
        const = 0.5 * (-d * alpha_bar_K - d * torch.log(one_minus))

        # KL
        kl = 0.5 * alpha_bar_K * z_sq + const
        return kl

    def _build_time_seq(self, device) -> torch.Tensor:
        """Build σ sequence for EDM sampling, length num_steps, on self.device.

        We compute t_i = (σ_max ^ 1/ρ + i/(K - 1) (σ_min ^ 1/ρ - σ_max ^ 1/ρ)) ^ ρ
        """
        dtype64 = torch.float64
        eps64 = torch.finfo(dtype64).eps
        sigma_max = float(self.schedule.sigma_max)
        sigma_min = float(self.schedule.sigma_min)
        rho = float(self.schedule.rho)
        K = self.num_steps
        i = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype64)
        inv_rho = 1.0 / rho
        t = (
            sigma_max**inv_rho + i * (sigma_min**inv_rho - sigma_max**inv_rho)
        ) ** float(rho)
        return t.to(torch.float32)

    @torch.no_grad()
    def _edm_denoise(
        self,
        x_noisy: torch.Tensor,  # (B, d)           current ẑ at σ
        z_hist: torch.Tensor,  # (B, d, j)        clean latents history
        side_win: torch.Tensor,  # (B, C_side, d, j+1)
        sigma: torch.Tensor,  # () or (B,)       current σ
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One EDM denoise eval at a given σ, this computes D_θ
        returns (x_denoised, dσ) where
          x_denoised ≈ z_t^0                (B, d)
          dσ = (x_noisy - x_denoised)/σ     (B, d)   ODE drift in σ-space
        """
        B, d = x_noisy.shape
        if sigma.dim() == 0:
            sigma = sigma.expand(B)
        # continuous preconditioning (same formulas as training)
        sigma2 = sigma * sigma
        c_skip = 1.0 / (1.0 + sigma2)  # (B,)
        c_out = sigma / torch.sqrt(1.0 + sigma2)  # (B,)
        c_in = 1.0 / torch.sqrt(1.0 + sigma2)  # (B,)
        c_noise = 0.25 * torch.log(
            torch.clamp(sigma, min=torch.finfo(sigma.dtype).tiny)
        )

        # latent window: concat [history, current] along time axis
        x_in = (c_in.view(B, 1) * x_noisy).unsqueeze(2)  # (B, d, 1)
        latent_w = torch.cat([z_hist, x_in], dim=2)  # (B, d, j+1)

        # U-Net forward
        y_pred = self.diffmodel(latent_w, side_win, c_noise)  # (B, d, 1)
        y_pred = y_pred.squeeze(-1)  # (B, d)

        # denoised estimate under target: z ≈ c_skip·ẑ + c_out·Fθ
        x_denoised = c_skip.view(B, 1) * x_noisy + c_out.view(B, 1) * y_pred
        d_sigma = (x_noisy - x_denoised) / sigma.view(B, 1)  # ODE drift
        return x_denoised, d_sigma

    @torch.no_grad()
    def edm_sample_latent(
        self,
        z_hist: torch.Tensor,  # (B, d, j)
        side_win: torch.Tensor,  # (B, C_side, d, j+1)
        s_churn: float = 30,
        s_tmin: float = 0.05,
        s_tmax: float = 30,
        s_noise: float = 1.003,
        return_last_denoised: bool = True,
    ) -> torch.Tensor:
        """EDM Karras sampler (Euler or Heun) for one time index t, conditioned on its history.
        Returns z_t^0 (clean) by default; set return_last_denoised=False to return the final noisy state.
        """
        device = z_hist.device
        B, d, j = z_hist.shape
        assert j == self.j, f"expected history j={self.j}, got {j}"
        assert side_win.shape == (B, self.side_dim, d, j + 1)

        # build t
        time_seq = self._build_time_seq(device)  # (Ks,) on self.device

        # # default σ trajectory = training schedule (reversed: max→min)
        # if sigma_seq is None:
        #     sigma_seq = self.sigma.flip(0).to(device=device, dtype=z_hist.dtype)  # (Ks,)
        #
        # init noisy state at the first (largest) σ
        x = time_seq[0] * torch.randn(B, d, device=device, dtype=z_hist.dtype)

        max_gamma = math.sqrt(2.0) - 1.0
        for i in range(len(time_seq) - 1):  # line 3
            time_i = time_seq[i]
            time_ip1 = time_seq[i + 1]
            # optional churn
            gamma = 0.0
            if s_tmin <= float(time_i) <= s_tmax:
                gamma = min(s_churn / (len(time_seq) - 1), max_gamma)
            time_i_hat = time_i * (1.0 + gamma)  # line 5.

            noise = torch.randn_like(x)  # line 4
            x = (
                x + (time_i_hat**2 - time_i**2).clamp_min(0).sqrt() * s_noise * noise
            )  # line 4/6

            x_denoised, d_i = self._edm_denoise(
                x, z_hist, side_win, time_i_hat
            )  # line 7

            # Euler step to t_{i+1}
            dt = time_ip1 - time_i_hat
            xp1 = x + dt * d_i

            # evaluate derivative at the end of the interval
            _, d_iprime = self._edm_denoise(xp1, z_hist, side_win, time_ip1)
            xp1 = x + dt * 0.5 * (d_i + d_iprime)

            x = xp1

        # final clean estimate
        if return_last_denoised:
            x_denoised, _ = self._edm_denoise(x, z_hist, side_win, time_seq[-1])
            return x_denoised
        return x
