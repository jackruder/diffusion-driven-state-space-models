"""Encoder-level combiners: fuse ``h_fut`` with the latent history ``z_{t-j:t-1}``
into a single feature vector for the distribution head.

The combiner composes a :class:`~ddssm.nn.aggregators.BaseHistoryAggregator`
(which mixes the ``j``-step history into one feature) with a
:class:`~ddssm.nn.fusions.BaseEncoderFusion` (which combines that with
``h_fut``). Use :class:`~ddssm.nn.aggregators.IdentityAggregator` when
``j=1`` — there is no history to mix; the only mixing left to do is the
``(h_fut, z_{t-1})`` fusion.
"""

from __future__ import annotations

import abc
from collections.abc import Callable

import torch
import torch.nn as nn

from ddssm.nn.fusions import BaseEncoderFusion
from ddssm.nn.aggregators import BaseHistoryAggregator


class BaseEncoderCombiner(nn.Module, metaclass=abc.ABCMeta):
    """Maps ``(h_fut, z_hist, ...)`` to a feature vector for the distribution head."""

    def __init__(
        self,
        *,
        latent_dim: int,
        j: int,
        summary_dim: int,
        hidden_dim: int,
        emb_time_dim: int,
        pad_mask_emb_dim: int = 8,
        fut_mask_emb_dim: int = 8,
        static_emb_dim: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.j = int(j)
        self.summary_dim = int(summary_dim)
        self.hidden_dim = int(hidden_dim)
        self.emb_time_dim = int(emb_time_dim)
        self.pad_mask_emb_dim = int(pad_mask_emb_dim)
        self.fut_mask_emb_dim = int(fut_mask_emb_dim)
        self.static_emb_dim = int(static_emb_dim)

    @property
    @abc.abstractmethod
    def out_features(self) -> int: ...

    @abc.abstractmethod
    def forward(
        self,
        *,
        h_fut: torch.Tensor,  # (B, summary_dim)
        z_hist: torch.Tensor,  # (B, d, j) — left-padded
        hist_time_emb: torch.Tensor,  # (B, j, emb_time_dim) — history steps only
        pad_mask_hist: torch.Tensor,  # (B, j) — 1 if real history, 0 if padded
        static_context: torch.Tensor | None = None,  # (B, E_s, hidden_dim)
    ) -> torch.Tensor:  # (B, out_features)
        ...


class CompoundCombiner(BaseEncoderCombiner):
    """Aggregator-then-fusion combiner.

    The aggregator processes the ``j``-step history into one feature; the
    fusion mixes that feature with ``h_fut``. Pair with
    :class:`~ddssm.nn.aggregators.IdentityAggregator` for ``j=1`` (no history
    to mix); pair with ``GRU/MLP/Attention/ContextProducer`` aggregators
    for ``j>1``.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        j: int,
        summary_dim: int,
        hidden_dim: int,
        emb_time_dim: int,
        pad_mask_emb_dim: int = 8,
        fut_mask_emb_dim: int = 8,  # accepted for interface uniformity; unused
        static_emb_dim: int = 0,
        aggregator: Callable[..., BaseHistoryAggregator],
        fusion: Callable[..., BaseEncoderFusion],
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            j=j,
            summary_dim=summary_dim,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim,
            pad_mask_emb_dim=pad_mask_emb_dim,
            fut_mask_emb_dim=fut_mask_emb_dim,
            static_emb_dim=static_emb_dim,
        )
        self.aggregator = aggregator(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=hidden_dim,
            emb_time_dim=emb_time_dim,
            pad_mask_emb_dim=pad_mask_emb_dim,
            static_emb_dim=static_emb_dim,
        )
        self.fusion = fusion(
            hist_features=self.aggregator.out_features,
            summary_dim=summary_dim,
            hidden_dim=hidden_dim,
        )
        self._out_features = self.fusion.out_features

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(
        self,
        *,
        h_fut: torch.Tensor,
        z_hist: torch.Tensor,
        hist_time_emb: torch.Tensor,
        pad_mask_hist: torch.Tensor,
        static_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hist_time_emb is length j (no h_fut slot under the new contract).
        z_hist_feat = self.aggregator(
            z_hist=z_hist,
            hist_time_emb=hist_time_emb,
            pad_mask=pad_mask_hist,
            static_context=static_context,
        )
        return self.fusion(h_fut=h_fut, z_hist_feat=z_hist_feat)
