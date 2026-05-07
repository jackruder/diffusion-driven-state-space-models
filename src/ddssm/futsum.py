"""Future-summary modules (F_ϕ) that summarise an observed sequence into a fixed-dimension context vector.

Supports pluggable time-mixing backbones (Mamba, GRU, Conv, Identity, Transformer)
and handles missing-data masking.  The summary is consumed by the encoder to
produce latent distributions q_ϕ(z_t | ·).
"""

import torch
import torch.nn as nn

from dataclasses import dataclass, field

from hydra_zen import builds

from .diffnets import TimeLayer, ConvTimeLayer, IdentityLayer  # , MambaTimeLayer


class FutureSummary(nn.Module):
    """Base class for F_ϕ: future-summary.

    Inputs (batched):
        observed_data : (B, T, D)
        observed_mask : (B, T, D)
        timepoints    : (B, T)
        static_embed  : (B, D, E_s)

    Output:
        h : (B, T, hidden_dim)
    """

    def __init__(
        self,
        data_dim: int,  # D
        emb_time_dim: int,  # E_t
        use_mask: bool,
        static_embed_dim: int = 0,
        summary_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.emb_time_dim = emb_time_dim
        self.static_embed_dim = static_embed_dim

        self.summary_dim = summary_dim
        self.num_layers = num_layers
        self.use_mask = use_mask

        self.input_dim = data_dim + emb_time_dim + (data_dim if use_mask else 0)

        # Add the flattened categorical embeddings to the input dim
        if self.static_embed_dim > 0:
            self.input_dim += data_dim * static_embed_dim

        self.input_proj = nn.Linear(self.input_dim, self.summary_dim)

    def _forward_mixer(self, x: torch.Tensor) -> torch.Tensor:
        """Process the sequence in the hidden space.
        Args:
            x: (B, T, hidden_dim) - already reversed
        Returns:
            x: (B, T, hidden_dim)
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

        x_aug = torch.cat([observed_data, t_emb], dim=-1)  # (B, T, D + E_t)
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
        h_rev = torch.flip(h_in, dims=[1])  # reverse time (B, T, summary_dim)

        h_rev = self._forward_mixer(h_rev)

        h = torch.flip(h_rev, dims=[1])  # (B, T, summary_dim)
        return h


class GRUFutureSummary(FutureSummary):
    def __init__(
        self,
        data_dim: int,
        emb_time_dim: int,
        use_mask: bool,
        static_embed_dim: int = 0,
        summary_dim: int = 64,
        num_layers: int = 2,
        gru_layers: int = 1,
    ):
        super().__init__(
            data_dim=data_dim,
            emb_time_dim=emb_time_dim,
            use_mask=use_mask,
            static_embed_dim=static_embed_dim,
            summary_dim=summary_dim,
            num_layers=num_layers,
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
    ):
        super().__init__(
            data_dim=data_dim,
            emb_time_dim=emb_time_dim,
            use_mask=use_mask,
            static_embed_dim=static_embed_dim,
            summary_dim=summary_dim,
            num_layers=num_layers,
        )
        d_model = self.summary_dim
        if d_model % nheads != 0:
            raise ValueError(
                f"FutureSummary summary_dim ({d_model}) must be divisible by "
                f"transformer nheads ({nheads})"
            )

        ff_dim = max(d_model, ff_mult * d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)

    def _forward_mixer(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H) in reversed time order.
        # Causal mask prevents looking "ahead" in reversed order.
        B, T, _ = x.shape
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        return self.encoder(x, mask=causal_mask)


# ---------------------------------------------------------------------------
# Nested type-specific configs
# ---------------------------------------------------------------------------

@dataclass
class GRUFutureSummaryConfig:
    """GRU-specific config for ``GRUFutureSummary``."""

    gru_layers: int = 1


@dataclass
class TransformerFutureSummaryConfig:
    """Transformer-specific config for ``TransformerFutureSummary``."""

    nheads: int = 4
    ff_mult: int = 4
    dropout: float = 0.0
    transformer_layers: int = 1


@dataclass
class FutureSummaryConfig:
    """Architectural config for the future-summary module.

    ``type`` selects the variant (``'gru'`` or ``'transformer'``).
    Type-specific parameters live in the ``gru`` / ``transformer`` sub-configs,
    so the parent module does not need to know which variant is active.

    Excludes shape params (``data_dim``, ``emb_time_dim``, ``use_mask``,
    ``static_embed_dim``) which are provided by the enclosing ``GaussianEncoder``.
    """

    type: str = "gru"
    summary_dim: int = 64
    num_layers: int = 2
    gru: GRUFutureSummaryConfig = field(default_factory=GRUFutureSummaryConfig)
    transformer: TransformerFutureSummaryConfig = field(
        default_factory=TransformerFutureSummaryConfig
    )


def build_future_summary(
    config: FutureSummaryConfig,
    data_dim: int,
    emb_time_dim: int,
    use_mask: bool,
    static_embed_dim: int = 0,
) -> FutureSummary:
    """Factory: create the appropriate FutureSummary from a ``FutureSummaryConfig``."""
    if config.type == "gru":
        return GRUFutureSummary(
            data_dim=data_dim,
            emb_time_dim=emb_time_dim,
            use_mask=use_mask,
            static_embed_dim=static_embed_dim,
            summary_dim=config.summary_dim,
            num_layers=config.num_layers,
            gru_layers=config.gru.gru_layers,
        )
    if config.type == "transformer":
        return TransformerFutureSummary(
            data_dim=data_dim,
            emb_time_dim=emb_time_dim,
            use_mask=use_mask,
            static_embed_dim=static_embed_dim,
            summary_dim=config.summary_dim,
            num_layers=config.num_layers,
            nheads=config.transformer.nheads,
            ff_mult=config.transformer.ff_mult,
            dropout=config.transformer.dropout,
            transformer_layers=config.transformer.transformer_layers,
        )
    raise NotImplementedError(
        f"FutureSummary type {config.type!r} not implemented. Choose 'gru' or 'transformer'."
    )


# ---------------------------------------------------------------------------
# Hydra-zen configs – co-located with the classes they describe
# ---------------------------------------------------------------------------

GRUFutureSummaryConf = builds(GRUFutureSummary, populate_full_signature=True)
TransformerFutureSummaryConf = builds(TransformerFutureSummary, populate_full_signature=True)
