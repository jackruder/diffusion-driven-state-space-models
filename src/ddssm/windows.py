from typing import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ReconViews:
    """Flattened, vectorized views for reconstruction over all (b, s, t).

    Let:
      - B = batch size, S = #MC posterior paths, T = #time steps,
      - d = latent dim, D = data dim (features), j = emission order (window length),
      - E_t = time-embedding dim.

    This object contains:

      • z_arg ∈ ℝ^{N×d}   if j = 1,
              ∈ ℝ^{N×d×j} if j > 1,
        where N = B·S·T and each row corresponds to the latent window
        z_{t-j+1:t} for a particular (b,s,t), left-padded on the time axis.

      • time_win ∈ ℝ^{N×j×E_t},
        the absolute time-embedding window aligned with z_{t-j+1:t};
        the u-th slice (u=0..j−1) is the embedding at time index max(0, t−j+1+u).

      • hist_valid ∈ ℕ^{N},
        the valid (non-padded) length ℓ_t = min(j, t+1) for each (b,s,t).

      • x_target ∈ ℝ^{N×D},
        the corresponding observed data x_t.

      • x_mask ∈ {0,1}^{N×D},
        mask of observed entries at (t, feature). If no mask was provided,
        this is an all-ones tensor.

    Use `unflatten(x, B, S, T)` to reshape any (N, …) tensor back to (B, S, T, …).
    """

    z_arg: torch.Tensor  # (N, d) if j==1 else (N, d, j)
    time_win: torch.Tensor  # (N, j, E_t)
    hist_valid: torch.Tensor  # (N,)
    x_target: torch.Tensor  # (N, D)
    x_mask: torch.Tensor  # (N, D)

    # Helper to reshape any (N, ...) back to (B, S, T, ...)
    def unflatten(self, x: torch.Tensor, B: int, S: int, T: int) -> torch.Tensor:
        lead = x.shape[1:]
        return x.view(B, S, T, *lead)


@dataclass
class StateInitWin:
    """Per-time window for state-initialization KL at step t (0-based, t ≥ 1).

    Mathematical contents (all flattened over B·S):

      • t: the global time index (0-based).

      • z_prev_bs ∈ ℝ^{(B·S)×d×k}, with k ≤ j,
        the last k = min(j, t) latents z_{t−k:t−1} (left-padded if needed).
        This is exactly the latent history the init model p_η will condition on.

      • time_win_bs ∈ ℝ^{(B·S)×j×E_t},
        absolute time-embedding window aligned to the CURRENT step t:
        indices [t−j+1, …, t] clamped to [0, T−1].
        (Right edge is t because p_η predicts z_t.)

      • x_win_bs ∈ ℝ^{(B·S)×D×j},
        data window aligned to the HISTORY (ends at t−1):
        indices [t−j, …, t−1] clamped to [0, T−1], left-padded.

      • mask_win_bs ∈ {0,1}^{(B·S)×D×j} or None,
        observation mask aligned with x_win_bs (all-ones if no mask provided).

      • side_info_bs: prebuilt side info for the window
        (e.g., concatenated time/feature embeddings), shaped as required
        by p_η/encoder ((B·S)×C_side×D×j).
    """

    t: int  # 0-based time index
    z_prev_bs: torch.Tensor  # (B*S, d, k<=j)
    time_win_bs: torch.Tensor  # (B*S, j, E_t)  (ends at t, last token is time t)
    x_win_bs: torch.Tensor  # (B*S, D, j)    (ends at t-1; left-padded)
    mask_win_bs: torch.Tensor | None  # (B*S, D, j) or None
    side_info_bs: torch.Tensor | None = None  # (B*S, C_side, D, j)


class WindowBuilder:
    r"""Build and cache left-padded length-j windows for latents and time,
    then expose vectorized views for reconstruction and per-t windows for
    state-initialization KL terms.

    Inputs
    ------
    observed_data : torch.Tensor, shape (B, D, T)
        The observed time series 𝐗 with entries x_{b,d,t}.

    observation_mask : Optional[torch.Tensor], shape (B, D, T)
        Binary mask 𝕄 with 1=observed, 0=missing. If None, an all-ones mask
        is used internally.

    time_embed : torch.Tensor, shape (B, T, E_t)
        Absolute time embeddings e_time(b,t) ∈ ℝ^{E_t}.

    zs : torch.Tensor, shape (B, S, d, T)
        Posterior samples of latents 𝐙; zs[b,s,:,t] = z_t for MC path s.

    j : int
        Emission order / window length.

    Cached tensors
    --------------
    z_win : torch.Tensor, shape (B, S, d, j, T)
        For each (b,s,t) the window [z_{t−j+1}, …, z_t], left-padded with zeros
        on the time axis.

    hist_valid : torch.Tensor, shape (B, S, T), integer
        ℓ_t = min(j, t+1) (the number of valid, non-padded elements within z_win at t).

    time_win : torch.Tensor, shape (B, S, j, T, E_t)
        Absolute time-embedding window aligned with z_win; for slice u (0..j−1),
        time index is max(0, t−j+1+u).

    Notes:
    -----
    • Left-padding semantics are consistent for z and time windows.
    • For j=1, windows degenerate to the current time step (no padding).
    • observation_mask=None is handled gracefully (treated as all ones).
    """

    def __init__(
        self,
        *,
        observed_data: torch.Tensor,
        observation_mask: torch.Tensor,
        time_embed: torch.Tensor,
        zs: torch.Tensor,
        j: int,
    ) -> None:
        # Bind & basic shapes
        self.x = observed_data
        self.m = observation_mask
        self.te = time_embed
        self.zs = zs
        self.j = int(j)

        B, D, T = observed_data.shape
        _, S, d, Tz = zs.shape
        assert Tz == T, "observed_data and zs must have same T"
        assert time_embed.shape[:2] == (B, T), "time_embed must be (B,T,E_t)"
        self.B, self.D, self.T = B, D, T
        self.S, self.d = S, d
        self.E_t = time_embed.shape[2]

        # Build caches
        self.z_win = self._build_z_windows()  # (B,S,d,j,T)
        self.hist_valid = self._build_hist_valid()  # (B,S,T)
        self.time_win = self._build_time_windows()  # (B,S,j,T,E_t)

    # ----------------- core builders -----------------

    def _build_z_windows(self) -> torch.Tensor:
        """Construct Z-windows Zwin[b,s,:, :, t] = [z_{t−j+1}, …, z_t] with left-padding.

        Returns:
        -------
        torch.Tensor, shape (B, S, d, j, T)

        Fast paths:
          • j == 1: simply zs.unsqueeze(3) to add the j-axis.
          • j ∈ {2,3}: use pad + small stack of j shifted slices (no unfold).
          • general j: same pad + stack approach (scales linearly in j).

        Padding
        -------
        Time-axis left-padding uses zeros for the missing positions.
        (The encoder/decoder handle true missing context via learned padding.)
        """
        B, S, d, T, j = self.B, self.S, self.d, self.T, self.j

        if j == 1:
            # trivial: each "window" is the current latent only
            return self.zs.unsqueeze(3)  # (B,S,d,1,T)

        # Left-pad in time with (j-1) zeros
        z_pad = F.pad(self.zs, (j - 1, 0))  # (B,S,d,T + j - 1)

        if j in (2, 3):
            # Stack j shifted views: each slice is z_{t-j+1+offset}
            slices = [z_pad[..., i : i + T] for i in range(j)]  # list of (B,S,d,T)
            z_win = torch.stack(slices, dim=3)  # (B,S,d,j,T)
            return z_win

        # General path (also fine for small j, but above is a hair cheaper)
        # Using the same pad+stack approach scales well; no need for unfold.
        slices = [z_pad[..., i : i + T] for i in range(j)]
        z_win = torch.stack(slices, dim=3)  # (B,S,d,j,T)
        return z_win

    def _build_time_windows(self) -> torch.Tensor:
        """Returns time_win: (B, S, j, T, E_t), left-padded by repeating the first token.
        Optimized (no gather): pad time by repeating te[:,0,:], then stack shifted slices.
        """
        B, S, T, E, j = self.B, self.S, self.T, self.E_t, self.j
        te = self.te  # (B, T, E)
        if j == 1:
            # Correct: (B,1,1,T,E) -> expand to (B,S,1,T,E)
            return te.unsqueeze(1).unsqueeze(2).expand(B, S, 1, T, E)

        # j >= 2 : left-clamped absolute windows of length j, ending at t
        pad_left = te[:, :1, :].expand(B, j - 1, E)  # (B, j-1, E)
        te_pad = torch.cat([pad_left, te], dim=1)  # (B, T + j - 1, E)
        # collect j shifted T-length slices
        slices = [te_pad[:, i : i + T, :] for i in range(j)]  # list of (B, T, E)
        time_win = torch.stack(slices, dim=1)  # (B, j, T, E)
        return time_win.unsqueeze(1).expand(B, S, j, T, E)  # (B, S, j, T, E)

    def _build_hist_valid(self) -> torch.Tensor:
        """Compute valid history lengths ℓ_t = min(j, t+1).

        Returns:
        -------
        torch.Tensor, shape (B, S, T), integer
        """
        B, S, T, j = self.B, self.S, self.T, self.j
        t_arange = torch.arange(T, device=self.x.device) + 1  # 1..T
        hist_valid = t_arange.clamp_max(j).view(1, 1, T).expand(B, S, T)
        return hist_valid

    # ----------------- reconstruction APIs -----------------

    def recon_flat_views(self) -> ReconViews:
        """Prepare vectorized inputs/targets for reconstruction across all (b,s,t).

        Returns:
        -------
        ReconViews with:
          • z_arg ∈ ℝ^{N×d} or ℝ^{N×d×j}
          • time_win ∈ ℝ^{N×j×E_t}
          • hist_valid ∈ ℕ^{N}
          • x_target ∈ ℝ^{N×D}
          • x_mask ∈ {0,1}^{N×D}

        where N = B·S·T.
        """
        B, S, T, D, d, j, E = self.B, self.S, self.T, self.D, self.d, self.j, self.E_t

        # z windows -> (N, d) if j==1 else (N, d, j)
        N = B * S * T
        z_win_flat = self.z_win.permute(0, 1, 4, 2, 3).reshape(
            N, d, j
        )  # (B,S,T,d,j)->(N,d,j)
        z_arg = z_win_flat.squeeze(-1) if j == 1 else z_win_flat

        # time windows -> (N, j, E_t)
        time_win_flat = self.time_win.permute(0, 1, 3, 2, 4).reshape(
            N, j, E
        )  # (B,S,T,j,E)->(N,j,E)

        # hist_valid -> (N,)
        hist_valid_flat = self.hist_valid.reshape(N)

        # targets and mask -> (N, D)
        x_flat = (
            self.x.unsqueeze(1).expand(B, S, D, T).permute(0, 1, 3, 2).reshape(N, D)
        )
        m_flat = (
            self.m.unsqueeze(1)
            .expand(B, S, D, T)
            .permute(0, 1, 3, 2)
            .reshape(N, D)
            .to(self.x.dtype)
        )

        return ReconViews(
            z_arg=z_arg,
            time_win=time_win_flat,
            hist_valid=hist_valid_flat,
            x_target=x_flat,
            x_mask=m_flat,
        )

    def recon_chunks(self, chunk_size: int) -> Iterator[ReconViews]:
        """Iterate over reconstruction views in chunks to limit memory.

        Yields:
        ------
        ReconViews for disjoint slices of size at most `chunk_size` along N=B·S·T.
        Shapes are the same as in `recon_flat_views`, but restricted to the slice.
        """
        rv = self.recon_flat_views()
        N = rv.x_target.shape[0]
        for i in range(0, N, chunk_size):
            sl = slice(i, min(i + chunk_size, N))
            yield ReconViews(
                z_arg=rv.z_arg[sl],
                time_win=rv.time_win[sl],
                hist_valid=rv.hist_valid[sl],
                x_target=rv.x_target[sl],
                x_mask=rv.x_mask[sl],
            )

    # ----------------- state-init KL windows -----------------

    def iter_state_init_windows(
        self,
        *,
        build_side_info: Callable[
            [int, torch.Tensor, torch.nn.Embedding, torch.Tensor | None, str],
            torch.Tensor,
        ]
        | None = None,
        embed_layer: torch.nn.Embedding | None = None,
        use_cond_mask: bool = True,
        device: str | None = None,
    ) -> Iterator[StateInitWin]:
        """Iterate windows for the state-initialization KLs at t = 1..min(j−1, T−1) (0-based).

        For each t (0-based, t ≥ 1), yields a StateInitWin containing:

          • z_prev_bs ∈ ℝ^{(B·S)×d×k}, k = min(j, t):
              z_{t−k:t−1} (left-padded) — the latent history used by p_η and q_ϕ.

          • time_win_bs ∈ ℝ^{(B·S)×j×E_t}:
              time window aligned to step t: indices [t−j+1, …, t] (right edge = t).

          • x_win_bs ∈ ℝ^{(B·S)×D×j}:
              data history window aligned to (t−1): indices [t−j, …, t−1].

          • mask_win_bs ∈ {0,1}^{(B·S)×D×j} or None:
              observation mask window aligned with x_win_bs; all-ones if no mask used.

          • mu_q_t_bs, logv_q_t_bs ∈ ℝ^{(B·S)×d}:
              encoder’s parameters for q_ϕ(z_t | …), broadcast across S.

          • side_info_bs:
              if `build_side_info` is provided a tensor
              shaped for the p_η/encoder (commonly (B·S)×C_side×D×j).

        Notes:
        -----
        • We use the cached `z_win` at t−1 to get z_{t−j: t−1} (length ≤ j).
        • We pass the time window ending at t because p_η predicts z_t.
        • We pass the x/mask window ending at t−1 to avoid peeking at x_t.
        """
        B, S, D, T, d, j, E = self.B, self.S, self.D, self.T, self.d, self.j, self.E_t
        device = device or self.x.device
        Tmax = min(j - 1, T - 1)  # t = 1..Tmax (0-based)

        if Tmax <= 0:
            return  # nothing to yield

        for t in range(1, Tmax + 1):
            # z history: take the z-window that ends at t, drop last slot -> ends at t-1
            z_hist_paths = self.z_win[..., t]  # (B,S,d,j)
            z_hist_paths = z_hist_paths[..., :-1]  # (B,S,d,j-1)  (k<=j)
            if z_hist_paths.numel() == 0:
                z_prev_bs = torch.zeros(B * S, d, 0, device=device)
            else:
                z_prev_bs = z_hist_paths.reshape(B * S, d, -1)  # (B*S,d,k<=j)

            # time window for step t: use time_win[..., t] (ends at t)
            time_win_bs = self.time_win[..., t, :].reshape(B * S, j, E)  # (B*S,j,E)

            # x window (length j) ending at t-1; left-padded
            x_win_t = self.x.new_zeros(B, D, j)  # (B,D,j)
            L = min(j, t)  # valid slots before t
            if L > 0:
                x_win_t[..., j - L :] = self.x[:, :, t - L : t]  # exclude x_t
            x_win_bs = x_win_t.unsqueeze(1).expand(B, S, D, j).reshape(B * S, D, j)

            # mask window aligned with x
            mask_win_t = None
            if use_cond_mask:
                m_win_t = self.m.new_zeros(B, D, j)
                if L > 0:
                    m_win_t[..., j - L :] = self.m[:, :, t - L : t]
                mask_win_t = (
                    m_win_t.unsqueeze(1)  # (B,1,D,j)
                    .expand(B, S, D, j)  # (B,S,D,j)
                    .reshape(B * S, D, j)  # (B·S,D,j)
                )

            # q stats for z_t, broadcast across S

            win = StateInitWin(
                t=t,
                z_prev_bs=z_prev_bs,
                time_win_bs=time_win_bs,
                x_win_bs=x_win_bs,
                mask_win_bs=mask_win_t,
            )

            # build side info for this window
            if build_side_info is not None:
                assert embed_layer is not None, (
                    "embed_layer required to build side info"
                )
                win.side_info_bs = build_side_info(
                    self.D,  # data_dim
                    win.time_win_bs,  # (B*S, j, E_t)
                    embed_layer,  # nn.Embedding(D, E_f)
                    mask_win_t if use_cond_mask else None,  # (B*S, D, j) or None
                    device=device,
                )

            yield win
