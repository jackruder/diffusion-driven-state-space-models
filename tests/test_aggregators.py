"""Smoke tests for :mod:`ddssm.aggregators`.

Verifies that each aggregator subclass produces the expected feature shape
and that the ``j=1`` constraint on :class:`IdentityAggregator` is enforced.
"""

from __future__ import annotations

import torch
import pytest

from ddssm.diffnets import FeatureMixerConfig, ResidualBlockConfig
from ddssm.aggregators import (
    GRUAggregator,
    MLPAggregator,
    IdentityAggregator,
    AttentionAggregator,
    ContextProducerAggregator,
)

B = 3
D = 4
H = 8  # hidden_dim
E_T = 6
NHEADS = 4  # divides H=8


def _inputs(*, B: int, d: int, j: int, e_t: int):
    z_hist = torch.randn(B, d, j)
    hist_time_emb = torch.randn(B, j, e_t)
    pad_mask = torch.ones(B, j)
    return z_hist, hist_time_emb, pad_mask


@pytest.mark.parametrize("j", [1])
def test_identity_aggregator_shape(j):
    agg = IdentityAggregator(
        latent_dim=D, j=j, hidden_dim=H, emb_time_dim=E_T,
    )
    z, t, m = _inputs(B=B, d=D, j=j, e_t=E_T)
    out = agg(z_hist=z, hist_time_emb=t, pad_mask=m)
    assert out.shape == (B, agg.out_features)
    assert agg.out_features == H


def test_identity_aggregator_rejects_j_gt_1():
    with pytest.raises(AssertionError):
        IdentityAggregator(latent_dim=D, j=2, hidden_dim=H, emb_time_dim=E_T)


@pytest.mark.parametrize("j", [1, 2, 4])
def test_gru_aggregator_shape(j):
    agg = GRUAggregator(
        latent_dim=D, j=j, hidden_dim=H, emb_time_dim=E_T, num_gru_layers=1,
    )
    z, t, m = _inputs(B=B, d=D, j=j, e_t=E_T)
    out = agg(z_hist=z, hist_time_emb=t, pad_mask=m)
    assert out.shape == (B, agg.out_features)
    assert agg.out_features == H


@pytest.mark.parametrize("j", [1, 2, 4])
def test_mlp_aggregator_shape(j):
    agg = MLPAggregator(
        latent_dim=D, j=j, hidden_dim=H, emb_time_dim=E_T, num_layers=2,
    )
    z, t, m = _inputs(B=B, d=D, j=j, e_t=E_T)
    out = agg(z_hist=z, hist_time_emb=t, pad_mask=m)
    assert out.shape == (B, agg.out_features)
    assert agg.out_features == H


@pytest.mark.parametrize("j", [1, 2, 4])
def test_attention_aggregator_shape(j):
    agg = AttentionAggregator(
        latent_dim=D, j=j, hidden_dim=H, emb_time_dim=E_T,
        nheads=NHEADS, num_attn_layers=1,
    )
    z, t, m = _inputs(B=B, d=D, j=j, e_t=E_T)
    out = agg(z_hist=z, hist_time_emb=t, pad_mask=m)
    assert out.shape == (B, agg.out_features)
    assert agg.out_features == H


def test_attention_aggregator_ignores_padded_positions():
    """Output is invariant to the *values* at padded slots.

    With the key-padding mask, real query positions attend only to real keys,
    and the masked mean pools only real positions — so perturbing padded-slot
    z/time values must not change the pooled feature.
    """
    j = 4
    agg = AttentionAggregator(
        latent_dim=D, j=j, hidden_dim=H, emb_time_dim=E_T,
        nheads=NHEADS, num_attn_layers=1,
    )
    agg.eval()
    torch.manual_seed(0)
    z = torch.randn(B, D, j)
    t = torch.randn(B, j, E_T)
    pad_mask = torch.ones(B, j)
    pad_mask[:, :2] = 0.0  # first two slots padded (mimics t<j left-padding)

    out1 = agg(z_hist=z, hist_time_emb=t, pad_mask=pad_mask)
    z2 = z.clone()
    z2[:, :, :2] += 5.0 * torch.randn(B, D, 2)
    t2 = t.clone()
    t2[:, :2, :] += 5.0 * torch.randn(B, 2, E_T)
    out2 = agg(z_hist=z2, hist_time_emb=t2, pad_mask=pad_mask)

    assert torch.allclose(out1, out2, atol=1e-5)


def test_attention_aggregator_handles_fully_padded_row():
    """A fully-padded row yields a finite, zero pooled feature (no NaN)."""
    j = 4
    agg = AttentionAggregator(
        latent_dim=D, j=j, hidden_dim=H, emb_time_dim=E_T,
        nheads=NHEADS, num_attn_layers=1,
    )
    agg.eval()
    torch.manual_seed(0)
    z = torch.randn(B, D, j)
    t = torch.randn(B, j, E_T)
    pad_mask = torch.ones(B, j)
    pad_mask[0, :] = 0.0  # row 0 has no real history

    out = agg(z_hist=z, hist_time_emb=t, pad_mask=pad_mask)
    assert torch.isfinite(out).all()
    # masked mean over zero valid positions → zero contribution (denom clamped).
    assert torch.allclose(out[0], torch.zeros(H))


@pytest.mark.parametrize("j", [1, 2, 4])
def test_context_producer_aggregator_shape(j):
    # ResidualBlockConfig default nheads=8; use a small nheads divisor of H=8.
    rb = ResidualBlockConfig(feature=FeatureMixerConfig(nheads=4, n_layers=1))
    agg = ContextProducerAggregator(
        latent_dim=D, j=j, hidden_dim=H, emb_time_dim=E_T,
        channels=4, num_layers=1, residual_block=rb,
    )
    z, t, m = _inputs(B=B, d=D, j=j, e_t=E_T)
    out = agg(z_hist=z, hist_time_emb=t, pad_mask=m)
    assert out.shape == (B, agg.out_features)
    assert agg.out_features == 4 * H


@pytest.mark.parametrize("agg_cls,extra", [
    (GRUAggregator, {"num_gru_layers": 1}),
    (MLPAggregator, {"num_layers": 2}),
    (AttentionAggregator, {"nheads": NHEADS, "num_attn_layers": 1}),
])
def test_simple_aggregators_reject_static_emb(agg_cls, extra):
    with pytest.raises(AssertionError):
        agg_cls(
            latent_dim=D, j=2, hidden_dim=H, emb_time_dim=E_T,
            static_emb_dim=2, **extra,
        )


def test_aggregator_backprop():
    """Gradients flow through each aggregator backbone."""
    j = 2
    rb = ResidualBlockConfig(feature=FeatureMixerConfig(nheads=4, n_layers=1))
    for agg_cls, extra in [
        (GRUAggregator, {"num_gru_layers": 1}),
        (MLPAggregator, {"num_layers": 2}),
        (AttentionAggregator, {"nheads": NHEADS, "num_attn_layers": 1}),
        (ContextProducerAggregator, {"channels": 4, "num_layers": 1, "residual_block": rb}),
    ]:
        agg = agg_cls(
            latent_dim=D, j=j, hidden_dim=H, emb_time_dim=E_T, **extra,
        )
        z = torch.randn(B, D, j, requires_grad=True)
        t = torch.randn(B, j, E_T)
        m = torch.ones(B, j)
        out = agg(z_hist=z, hist_time_emb=t, pad_mask=m)
        out.sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()
