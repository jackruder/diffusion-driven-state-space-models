"""Future-summary modules (F_ϕ) that summarise an observed sequence.

Each module projects to a per-step context vector via a pluggable mixing
backbone (GRU, Conv, Identity, Transformer). With ``reverse_time=True`` (the
default) it runs in reversed time so each step ``t`` summarises its FUTURE
``x_{t:T}`` — the backward data message ``b_t``. With ``reverse_time=False`` the
same backbone runs forward-causally, so each step summarises its PAST
``x_{1:t}`` — the forward data message ``f_t``. Missing-data masking is handled
inline. The summary is consumed by the encoder to produce q_ϕ(z_t | ·).
"""

import torch
import torch.nn as nn

from ddssm.nn.net_utils import TransformerEncoder


class FutureSummary(nn.Module):
    """Base class for F_ϕ: future-summary.

    Inputs (batched):
        observed_data : (B, T, D)
        observed_mask : (B, T, D)
        timepoints    : (B, T)
        static_embed  : (B, D, E_s)

    Output:
        h : (B, T, summary_dim)
    """

    def __init__(
        self,
        data_dim: int,  # D
        emb_time_dim: int,  # E_t
        use_mask: bool,
        static_embed_dim: int = 0,
        summary_dim: int = 64,
        num_layers: int = 2,
        reverse_time: bool = True,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.emb_time_dim = emb_time_dim
        self.static_embed_dim = static_embed_dim

        self.summary_dim = summary_dim
        self.num_layers = num_layers
        self.use_mask = use_mask
        # True: backward summary b_t (reversed time → step t sees x_{t:T}).
        # False: forward-causal message f_t (step t sees x_{1:t}).
        self.reverse_time = reverse_time

        self.input_dim = data_dim + emb_time_dim + (data_dim if use_mask else 0)

        # Add the flattened categorical embeddings to the input dim
        if self.static_embed_dim > 0:
            self.input_dim += data_dim * static_embed_dim

        self.input_proj = nn.Linear(self.input_dim, self.summary_dim)

    def _forward_mixer(self, x: torch.Tensor) -> torch.Tensor:
        """Process the time-reversed sequence in the summary space.

        Args:
            x: (B, T, summary_dim), already time-reversed.

        Returns:
            x: (B, T, summary_dim).
        """
        raise NotImplementedError

    def forward(
        self,
        observed_data: torch.Tensor,  # (B, T, D)
        observed_mask: torch.Tensor | None,  # (B, T, D)
        t_emb: torch.Tensor,  # (B, T, E_t)
        static_embed: torch.Tensor | None = None,  # (B, D, E_s)
    ) -> torch.Tensor:
        _B, _T, D = observed_data.shape
        assert self.data_dim == D

        # Gate the time concat on the Python int (compile-time constant) rather
        # than on tensor shape — keeps Inductor specialization clean when the
        # absolute-time path is off (``emb_time_dim == 0``).
        if self.emb_time_dim > 0:
            x_aug = torch.cat([observed_data, t_emb], dim=-1)  # (B, T, D + E_t)
        else:
            x_aug = observed_data
        if observed_mask is not None:
            x_aug = torch.cat([x_aug, observed_mask], dim=-1)  # (B, T, D + E_t + D)
        if self.static_embed_dim > 0 and static_embed is not None:
            # Flatten: (B, D, E_s) -> (B, D * E_s)
            se_flat = static_embed.reshape(_B, -1)
            # Expand over time: (B, D * E_s) -> (B, T, D * E_s)
            se_expanded = se_flat.unsqueeze(1).expand(-1, _T, -1)
            x_aug = torch.cat(
                [x_aug, se_expanded], dim=-1
            )  # (B, T, D + E_t + D + D * E_s)

        h_in = self.input_proj(x_aug)
        if self.reverse_time:
            # Backward summary: mix in reversed time so a causal backbone gives
            # each step its FUTURE, then flip back to original order.
            h_in = torch.flip(h_in, dims=[1])  # (B, T, summary_dim)
            h_mix = self._forward_mixer(h_in)
            return torch.flip(h_mix, dims=[1])  # (B, T, summary_dim)
        # Forward-causal message f_t: mix in original order (causal backbone →
        # step t sees only x_{1:t}); no flip.
        return self._forward_mixer(h_in)


class GRUFutureSummary(FutureSummary):
    """Future-summary with a GRU time-mixing backbone."""

    def __init__(
        self,
        data_dim: int,
        emb_time_dim: int,
        use_mask: bool,
        static_embed_dim: int = 0,
        summary_dim: int = 64,
        num_layers: int = 2,
        gru_layers: int = 1,
        reverse_time: bool = True,
    ):
        super().__init__(
            data_dim=data_dim,
            emb_time_dim=emb_time_dim,
            use_mask=use_mask,
            static_embed_dim=static_embed_dim,
            summary_dim=summary_dim,
            num_layers=num_layers,
            reverse_time=reverse_time,
        )
        self.rnn = nn.GRU(
            input_size=self.summary_dim,
            hidden_size=self.summary_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

    def _forward_mixer(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        x, _ = self.rnn(x)
        return x


class TransformerFutureSummary(FutureSummary):
    """Future-summary with a causal Transformer-encoder time-mixing backbone.

    Causality is applied in reversed-time order (each step attends only to its
    own future in the original sequence). Uses the project's RMSNorm + SwiGLU
    + SDPA TransformerEncoder for bf16-autocast stability — the stock
    nn.TransformerEncoder's MHA softmax-backward path produces NaNs in bf16
    when attention scores get peaky (see test_bf16_attention).
    """

    def __init__(
        self,
        data_dim: int,
        emb_time_dim: int,
        use_mask: bool,
        static_embed_dim: int = 0,
        summary_dim: int = 64,
        num_layers: int = 2,
        nheads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        transformer_layers: int = 1,
        reverse_time: bool = True,
    ):
        super().__init__(
            data_dim=data_dim,
            emb_time_dim=emb_time_dim,
            use_mask=use_mask,
            static_embed_dim=static_embed_dim,
            summary_dim=summary_dim,
            num_layers=num_layers,
            reverse_time=reverse_time,
        )
        d_model = self.summary_dim
        if d_model % nheads != 0:
            raise ValueError(
                f"FutureSummary summary_dim ({d_model}) must be divisible by "
                f"transformer nheads ({nheads})"
            )

        ff_dim = max(d_model, ff_mult * d_model)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nheads=nheads,
            num_layers=transformer_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            causal=True,
            rope=True,
        )

    def _forward_mixer(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H) in reversed time (backward b_t) or original time (forward
        # f_t), set by reverse_time. The causal mask (SDPA is_causal=True) gives
        # each step its future (reversed) or its past (forward); no manual triu.
        return self.encoder(x)


class IdentityFutureSummary(FutureSummary):
    """Local per-step summary: ``h_t = Linear(x_t)``; no time mixing.

    The identity backbone gives the encoder a FILTERING posterior
    ``q(z_t | x_t, z_hist)`` (each step sees only its own observation) instead of
    the smoothing ``q(z_t | x_{t:T}, z_hist)`` of the GRU/Transformer summaries.
    For a per-timestep lift ``x_t = lift(z_t)`` this is the matched inverse: the
    encoder can learn a clean near-identity frame at ``latent_dim == data_dim``
    without blurring information across time. ``reverse_time`` is a no-op here
    (identity commutes with the time flip).
    """

    def _forward_mixer(self, x: torch.Tensor) -> torch.Tensor:
        return x
