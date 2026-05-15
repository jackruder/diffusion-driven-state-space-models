"""Encoder fusions: combine the future summary ``h_fut`` with the
aggregator's z-history feature into a single feature vector for the
distribution head.

The fusion is encoder-specific (decoder/init-prior/transition have no
``h_fut`` to fuse with). Three flavors:

* :class:`ConcatLinearFusion` — project both inputs and concatenate; the
  simple baseline.
* :class:`DKSFusion` — the Krishnan-et-al combiner shape:
  ``0.5 * (tanh(W·z) + W'·h)``.
* :class:`GatedFusion` — sigmoid-gated mix of the two projections.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseEncoderFusion(nn.Module, metaclass=abc.ABCMeta):
    """Maps ``(h_fut, z_hist_feat)`` to a single feature vector."""

    def __init__(
        self,
        *,
        hist_features: int,
        summary_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.hist_features = int(hist_features)
        self.summary_dim = int(summary_dim)
        self.hidden_dim = int(hidden_dim)

    @property
    @abc.abstractmethod
    def out_features(self) -> int:
        ...

    @abc.abstractmethod
    def forward(
        self,
        *,
        h_fut: torch.Tensor,  # (B, summary_dim)
        z_hist_feat: torch.Tensor,  # (B, hist_features)
    ) -> torch.Tensor:  # (B, out_features)
        ...


class ConcatLinearFusion(BaseEncoderFusion):
    """Project ``h_fut`` and ``z_hist_feat`` to ``hidden_dim`` each, concatenate."""

    def __init__(
        self,
        *,
        hist_features: int,
        summary_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__(
            hist_features=hist_features,
            summary_dim=summary_dim,
            hidden_dim=hidden_dim,
        )
        self.h_proj = nn.Linear(self.summary_dim, self.hidden_dim)
        self.z_proj = nn.Linear(self.hist_features, self.hidden_dim)
        self._out_features = 2 * self.hidden_dim

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(
        self,
        *,
        h_fut: torch.Tensor,
        z_hist_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.h_proj(h_fut)
        z = self.z_proj(z_hist_feat)
        return torch.cat([h, z], dim=-1)


class DKSFusion(BaseEncoderFusion):
    """DKS combiner: ``0.5 * (W_h·h_fut + tanh(W_z·z_hist_feat))``."""

    def __init__(
        self,
        *,
        hist_features: int,
        summary_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__(
            hist_features=hist_features,
            summary_dim=summary_dim,
            hidden_dim=hidden_dim,
        )
        self.h_proj = nn.Linear(self.summary_dim, self.hidden_dim)
        self.z_proj = nn.Linear(self.hist_features, self.hidden_dim)
        self._out_features = self.hidden_dim

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(
        self,
        *,
        h_fut: torch.Tensor,
        z_hist_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.h_proj(h_fut)
        z = torch.tanh(self.z_proj(z_hist_feat))
        return 0.5 * (h + z)


class GatedFusion(BaseEncoderFusion):
    """Sigmoid-gated mix: ``g * z_proj + (1 - g) * h_proj``."""

    def __init__(
        self,
        *,
        hist_features: int,
        summary_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__(
            hist_features=hist_features,
            summary_dim=summary_dim,
            hidden_dim=hidden_dim,
        )
        self.h_proj = nn.Linear(self.summary_dim, self.hidden_dim)
        self.z_proj = nn.Linear(self.hist_features, self.hidden_dim)
        self.gate = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self._out_features = self.hidden_dim

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(
        self,
        *,
        h_fut: torch.Tensor,
        z_hist_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.h_proj(h_fut)
        z = self.z_proj(z_hist_feat)
        g = torch.sigmoid(self.gate(torch.cat([h, z], dim=-1)))
        return g * z + (1.0 - g) * h
