"""History aggregators for the encoder (and reusable by decoder/init-prior/transition).

A :class:`BaseHistoryAggregator` turns a length-``j`` latent history
``z_{t-j:t-1}`` together with the corresponding time embeddings and pad mask
into a single feature vector. The encoder then fuses that feature with the
future summary ``h_fut`` (via :mod:`ddssm.fusions`); modules that have no
future summary (decoder / init prior / transition) can consume the aggregator
output directly.

Five backbones are provided:

* :class:`IdentityAggregator` — ``j=1`` only; no temporal mixing (the history
  is a single vector). Still projects ``z_{t-1}`` to ``hidden_dim`` so the
  downstream fusion stage sees a comparable feature.
* :class:`GRUAggregator` — GRU over the ``j``-step history (the DKS shape).
* :class:`MLPAggregator` — flatten the projected history + MLP.
* :class:`AttentionAggregator` — Transformer-encoder self-attention pool.
* :class:`ContextProducerAggregator` — wraps :class:`ContextProducer` (the
  residual block stack used in the encoder today).
"""

from __future__ import annotations

import abc
from typing import Optional

import torch
import torch.nn as nn

from .diffnets import ContextProducer, ResidualBlockConfig


class BaseHistoryAggregator(nn.Module, metaclass=abc.ABCMeta):
    """Maps a ``j``-step latent history to a single feature vector.

    Subclasses implement :meth:`forward` and expose :attr:`out_features`.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        j: int,
        hidden_dim: int,
        emb_time_dim: int,
        pad_mask_emb_dim: int = 8,
        static_emb_dim: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.j = int(j)
        self.hidden_dim = int(hidden_dim)
        self.emb_time_dim = int(emb_time_dim)
        self.pad_mask_emb_dim = int(pad_mask_emb_dim)
        self.static_emb_dim = int(static_emb_dim)

    @property
    @abc.abstractmethod
    def out_features(self) -> int:
        """Feature dim consumed by the fusion / distribution head."""
        ...

    @abc.abstractmethod
    def forward(
        self,
        *,
        z_hist: torch.Tensor,  # (B, d, j) — left-padded by caller
        hist_time_emb: torch.Tensor,  # (B, j, E_t)
        pad_mask: torch.Tensor,  # (B, j) — 1 if real history, 0 if padded
        static_context: Optional[torch.Tensor] = None,  # (B, E_s, H)
    ) -> torch.Tensor:  # (B, out_features)
        ...


class IdentityAggregator(BaseHistoryAggregator):
    """``j=1`` aggregator: no history mixing (there is no history to mix).

    Projects ``z_{t-1}`` to ``hidden_dim``; optionally folds in the per-step
    time embedding and pad-mask scalar.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        j: int,
        hidden_dim: int,
        emb_time_dim: int,
        pad_mask_emb_dim: int = 8,
        static_emb_dim: int = 0,
    ) -> None:
        assert j == 1, "IdentityAggregator requires j == 1"
        assert static_emb_dim == 0, (
            "IdentityAggregator does not consume static_context; "
            "use ContextProducerAggregator for static covariates"
        )
        super().__init__(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim,
            pad_mask_emb_dim=pad_mask_emb_dim,
            static_emb_dim=static_emb_dim,
        )
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_proj = nn.Linear(emb_time_dim, hidden_dim) if emb_time_dim > 0 else None
        self.pad_mask_proj = nn.Linear(1, hidden_dim)
        self._out_features = hidden_dim

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(
        self,
        *,
        z_hist: torch.Tensor,
        hist_time_emb: torch.Tensor,
        pad_mask: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # z_hist: (B, d, 1); time: (B, 1, E_t); mask: (B, 1)
        B = z_hist.shape[0]
        z = z_hist.squeeze(-1)  # (B, d)
        out = self.z_proj(z)  # (B, H)
        if self.time_proj is not None:
            t_emb = hist_time_emb.squeeze(1)  # (B, E_t)
            out = out + self.time_proj(t_emb)
        out = out + self.pad_mask_proj(pad_mask)  # (B, H)
        return out


class _PerStepProj(nn.Module):
    """Shared per-step projection used by GRU/MLP/Attention aggregators.

    Projects ``z`` + time + pad-mask into a length-``j`` sequence of ``H``-dim
    feature vectors.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        emb_time_dim: int,
    ) -> None:
        super().__init__()
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_proj = nn.Linear(emb_time_dim, hidden_dim) if emb_time_dim > 0 else None
        self.pad_mask_proj = nn.Linear(1, hidden_dim)

    def forward(
        self,
        *,
        z_hist: torch.Tensor,  # (B, d, j)
        hist_time_emb: torch.Tensor,  # (B, j, E_t)
        pad_mask: torch.Tensor,  # (B, j)
    ) -> torch.Tensor:  # (B, j, H)
        z = z_hist.permute(0, 2, 1)  # (B, j, d)
        out = self.z_proj(z)  # (B, j, H)
        if self.time_proj is not None:
            out = out + self.time_proj(hist_time_emb)
        out = out + self.pad_mask_proj(pad_mask.unsqueeze(-1))
        return out


class GRUAggregator(BaseHistoryAggregator):
    """GRU rolling over the ``j``-step history (DKS-style temporal mixer).

    The last GRU output is the aggregator feature.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        j: int,
        hidden_dim: int,
        emb_time_dim: int,
        pad_mask_emb_dim: int = 8,
        static_emb_dim: int = 0,
        num_gru_layers: int = 1,
    ) -> None:
        assert static_emb_dim == 0, (
            "GRUAggregator does not consume static_context"
        )
        super().__init__(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim,
            pad_mask_emb_dim=pad_mask_emb_dim,
            static_emb_dim=static_emb_dim,
        )
        self.per_step = _PerStepProj(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim,
        )
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=int(num_gru_layers),
            batch_first=True,
        )
        self._out_features = hidden_dim

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(
        self,
        *,
        z_hist: torch.Tensor,
        hist_time_emb: torch.Tensor,
        pad_mask: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.per_step(z_hist=z_hist, hist_time_emb=hist_time_emb, pad_mask=pad_mask)
        out, _ = self.rnn(x)  # (B, j, H)
        return out[:, -1, :]  # newest-step output


class MLPAggregator(BaseHistoryAggregator):
    """Flatten the projected history + MLP."""

    def __init__(
        self,
        *,
        latent_dim: int,
        j: int,
        hidden_dim: int,
        emb_time_dim: int,
        pad_mask_emb_dim: int = 8,
        static_emb_dim: int = 0,
        num_layers: int = 2,
    ) -> None:
        assert static_emb_dim == 0, (
            "MLPAggregator does not consume static_context"
        )
        super().__init__(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim,
            pad_mask_emb_dim=pad_mask_emb_dim,
            static_emb_dim=static_emb_dim,
        )
        self.per_step = _PerStepProj(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim,
        )
        in_dim = j * hidden_dim
        depth = max(int(num_layers), 1)
        layers: list[nn.Module] = []
        if depth == 1:
            layers.append(nn.Linear(in_dim, hidden_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            for _ in range(depth - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)
        self._out_features = hidden_dim

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(
        self,
        *,
        z_hist: torch.Tensor,
        hist_time_emb: torch.Tensor,
        pad_mask: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.per_step(z_hist=z_hist, hist_time_emb=hist_time_emb, pad_mask=pad_mask)
        B = x.shape[0]
        return self.mlp(x.reshape(B, -1))


class AttentionAggregator(BaseHistoryAggregator):
    """Transformer-encoder self-attention pool over the ``j``-step history.

    Masked-mean-pools the resulting length-``j`` sequence to a single feature:
    padded history slots are excluded both from the self-attention (via a
    ``src_key_padding_mask``) and from the pooling denominator.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        j: int,
        hidden_dim: int,
        emb_time_dim: int,
        pad_mask_emb_dim: int = 8,
        static_emb_dim: int = 0,
        nheads: int = 4,
        num_attn_layers: int = 1,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        assert static_emb_dim == 0, (
            "AttentionAggregator does not consume static_context"
        )
        assert hidden_dim % nheads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by nheads ({nheads})"
        )
        super().__init__(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim,
            pad_mask_emb_dim=pad_mask_emb_dim,
            static_emb_dim=static_emb_dim,
        )
        self.per_step = _PerStepProj(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(nheads),
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=int(num_attn_layers))
        self._out_features = hidden_dim

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(
        self,
        *,
        z_hist: torch.Tensor,
        hist_time_emb: torch.Tensor,
        pad_mask: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.per_step(z_hist=z_hist, hist_time_emb=hist_time_emb, pad_mask=pad_mask)

        # Hard key-padding mask so real query positions don't attend to padded
        # history keys (for early t<j left-padding and GluonTS past_is_pad).
        # ``nn.TransformerEncoder`` wants True == "ignore this key"; pad_mask is
        # 1=real / 0=pad. This is the repo's only hard attention-mask (the rest
        # use mask-as-feature) — intentional, to keep the attention aggregator
        # correct under j>1.
        key_pad = pad_mask == 0  # (B, j) bool, True at padded slots
        # A fully-padded row (no real history) would make every key -inf and
        # produce a NaN softmax. Let those rows attend uniformly (well-defined);
        # the masked mean below zeroes their contribution anyway.
        all_pad = key_pad.all(dim=1)
        if bool(all_pad.any()):
            key_pad = key_pad.clone()
            key_pad[all_pad] = False

        x = self.attn(x, src_key_padding_mask=key_pad)  # (B, j, H)

        # Masked mean over valid (non-padded) positions only — divide by the
        # real-step count, not j, so padding doesn't dilute the pooled feature.
        pm = pad_mask.to(x.dtype).unsqueeze(-1)  # (B, j, 1)
        denom = pm.sum(dim=1).clamp_min(1.0)  # (B, 1)
        return (x * pm).sum(dim=1) / denom


class ContextProducerAggregator(BaseHistoryAggregator):
    """Wraps :class:`ContextProducer` over a length-``j`` history.

    This preserves the residual-block stack the encoder uses today (the
    history side only; ``h_fut`` is now folded in at the fusion stage).
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        j: int,
        hidden_dim: int,
        emb_time_dim: int,
        pad_mask_emb_dim: int = 8,
        static_emb_dim: int = 0,
        channels: int = 8,
        num_layers: int = 2,
        residual_block: Optional[ResidualBlockConfig] = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim,
            pad_mask_emb_dim=pad_mask_emb_dim,
            static_emb_dim=static_emb_dim,
        )
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.pad_mask_embed = nn.Linear(1, pad_mask_emb_dim)
        self.context_producer = ContextProducer(
            channels=int(channels),
            num_layers=int(num_layers),
            combined_dim=hidden_dim,
            mask_tot_dim=pad_mask_emb_dim,
            emb_time_dim=emb_time_dim,
            combined_len=j,
            residual_block=residual_block,
            static_emb_dim=static_emb_dim,
        )
        self._out_features = int(channels) * hidden_dim

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(
        self,
        *,
        z_hist: torch.Tensor,
        hist_time_emb: torch.Tensor,
        pad_mask: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # z_hist: (B, d, j), hist_time_emb: (B, j, E_t), pad_mask: (B, j)
        z = z_hist.permute(0, 2, 1)  # (B, j, d)
        z_proj = self.z_proj(z)  # (B, j, H)
        combined = z_proj.permute(0, 2, 1)  # (B, H, j)
        hist_time_emb_t = hist_time_emb.permute(0, 2, 1)  # (B, E_t, j)
        pad_mask_emb = self.pad_mask_embed(pad_mask.unsqueeze(-1))  # (B, j, E_pad)
        pad_mask_emb = pad_mask_emb.permute(0, 2, 1)  # (B, E_pad, j)
        return self.context_producer.forward(
            combined=combined,
            mask_embedded=pad_mask_emb,
            hist_time_emb=hist_time_emb_t,
            static_embedded=static_context,
        )
