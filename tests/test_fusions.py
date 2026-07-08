"""Smoke tests for :mod:`ddssm.fusions`."""

from __future__ import annotations

import torch
import pytest

from ddssm.nn.fusions import DKSFusion, GatedFusion, ConcatLinearFusion

B = 3
H = 8  # hidden_dim
HIST = 12  # aggregator out_features
SUMMARY = 16  # h_fut dim


@pytest.mark.parametrize(
    "cls,out_factor",
    [
        (ConcatLinearFusion, 2),
        (DKSFusion, 1),
        (GatedFusion, 1),
    ],
)
def test_fusion_shape(cls, out_factor):
    fusion = cls(hist_features=HIST, summary_dim=SUMMARY, hidden_dim=H)
    h_fut = torch.randn(B, SUMMARY)
    z_hist_feat = torch.randn(B, HIST)
    out = fusion(h_fut=h_fut, z_hist_feat=z_hist_feat)
    assert out.shape == (B, fusion.out_features)
    assert fusion.out_features == out_factor * H


@pytest.mark.parametrize("cls", [ConcatLinearFusion, DKSFusion, GatedFusion])
def test_fusion_backprop(cls):
    fusion = cls(hist_features=HIST, summary_dim=SUMMARY, hidden_dim=H)
    h_fut = torch.randn(B, SUMMARY, requires_grad=True)
    z_hist_feat = torch.randn(B, HIST, requires_grad=True)
    out = fusion(h_fut=h_fut, z_hist_feat=z_hist_feat)
    out.sum().backward()
    assert h_fut.grad is not None and torch.isfinite(h_fut.grad).all()
    assert z_hist_feat.grad is not None and torch.isfinite(z_hist_feat.grad).all()


def test_dks_fusion_average_property():
    """DKSFusion outputs the average of two LN'd projections (LN on z pre-tanh)."""
    fusion = DKSFusion(hist_features=HIST, summary_dim=SUMMARY, hidden_dim=H)
    h_fut = torch.randn(B, SUMMARY)
    z = torch.randn(B, HIST)
    out = fusion(h_fut=h_fut, z_hist_feat=z)
    h_proj = fusion.h_ln(fusion.h_proj(h_fut))
    z_proj = torch.tanh(fusion.z_ln(fusion.z_proj(z)))
    expected = 0.5 * (h_proj + z_proj)
    assert torch.allclose(out, expected)
