"""Verify transformer attention layers produce finite gradients under bf16 autocast.

Stress tests the pre-norm + SDPA path with peaky attention scores (one Q·K dot
product much larger than others) to guard against the softmax-backward NaN that
occurs when unnormalized (post-norm) bf16 attention saturates.
"""

import pytest
import torch

from ddssm.nn.net_utils import TransformerEncoder, get_torch_trans
from ddssm.nn.diffnets import (
    CausalTransformerTimeLayer,
    TransformerFeatureLayer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKIP_NO_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _peaky_input(B: int, T: int, C: int, scale: float = 50.0) -> torch.Tensor:
    """Random input with one position at extreme magnitude to drive peaky attention."""
    x = torch.randn(B, T, C, device=DEVICE)
    x[:, 0, :] *= scale
    return x


@SKIP_NO_CUDA
@pytest.mark.parametrize("channels,nheads", [(64, 8), (48, 2), (128, 4)])
def test_get_torch_trans_bf16_gradients(channels, nheads):
    model = get_torch_trans(
        heads=nheads, layers=1, channels=channels, dropout=0.0
    ).to(DEVICE)
    x = _peaky_input(4, 16, channels, scale=100.0).requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y = model(x)
        loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "NaN/Inf in input gradients"
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf in {n}"


@SKIP_NO_CUDA
def test_rope_encoder_bf16_causal():
    model = TransformerEncoder(
        d_model=64, nheads=8, num_layers=2, causal=True, rope=True, dropout=0.0
    ).to(DEVICE)
    x = _peaky_input(4, 32, 64, scale=80.0).requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y = model(x)
        loss = y.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf in {n}"


@SKIP_NO_CUDA
def test_rope_encoder_causality():
    """Output at position t must not depend on positions > t."""
    model = TransformerEncoder(
        d_model=16, nheads=2, num_layers=1, causal=True, rope=True, dropout=0.0
    ).to(DEVICE)
    model.eval()
    x = torch.randn(1, 8, 16, device=DEVICE)
    with torch.no_grad():
        y_full = model(x)
        y_prefix = model(x[:, :4, :])
    torch.testing.assert_close(y_full[:, :4, :], y_prefix, atol=1e-5, rtol=1e-5)


@SKIP_NO_CUDA
@pytest.mark.parametrize("channels,nheads", [(64, 8), (48, 2)])
def test_feature_layer_bf16_gradients(channels, nheads):
    layer = TransformerFeatureLayer(
        channels=channels, nheads=nheads, layers=1, dropout=0.0
    ).to(DEVICE)
    B, C, d, L = 4, channels, 8, 16
    x_flat = _peaky_input(B, d * L, C, scale=80.0).reshape(B, C, d * L)
    x_flat.requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x_flat, (B, C, d, L))
        loss = y.sum()
    loss.backward()
    assert torch.isfinite(x_flat.grad).all()


@SKIP_NO_CUDA
@pytest.mark.parametrize("channels,nheads", [(64, 8), (48, 2)])
def test_causal_time_layer_bf16_gradients(channels, nheads):
    layer = CausalTransformerTimeLayer(
        channels=channels, nheads=nheads, layers=1, dropout=0.0
    ).to(DEVICE)
    B, C, d, L = 4, channels, 8, 16
    x_flat = _peaky_input(B, d * L, C, scale=80.0).reshape(B, C, d * L)
    x_flat.requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x_flat, (B, C, d, L))
        loss = y.sum()
    loss.backward()
    assert torch.isfinite(x_flat.grad).all()


def test_head_dim_validation():
    with pytest.raises(ValueError, match="multiple of 8"):
        get_torch_trans(heads=4, channels=48)
    with pytest.raises(ValueError, match="multiple of 8"):
        TransformerEncoder(d_model=48, nheads=4)
