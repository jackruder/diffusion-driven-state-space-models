"""Tests for fail-fast validation in the RoPE implementation.

Covers:
- Odd head_dim raises at table construction time (via _precompute_rope_freqs).
- seq_len > max_len raises at forward time with a message naming both limits.
"""

from __future__ import annotations

import pytest
import torch

from ddssm.nn.net_utils import TransformerEncoder, _apply_rope, _precompute_rope_freqs


def test_rope_odd_head_dim_raises_at_construction() -> None:
    """_precompute_rope_freqs raises ValueError for odd head_dim."""
    with pytest.raises(ValueError, match="must be even"):
        _precompute_rope_freqs(head_dim=3, max_len=16)


def test_rope_even_head_dim_ok() -> None:
    """_precompute_rope_freqs succeeds for even head_dim."""
    freqs = _precompute_rope_freqs(head_dim=8, max_len=16)
    assert freqs.shape == (16, 4)


def test_rope_seq_len_exceeds_max_len_raises() -> None:
    """_apply_rope raises ValueError when T > max_len, naming both values."""
    max_len = 8
    head_dim = 8
    freqs = _precompute_rope_freqs(head_dim=head_dim, max_len=max_len)
    # Build a query tensor with T = max_len + 1
    T = max_len + 1
    x = torch.randn(1, T, 2, head_dim)
    with pytest.raises(ValueError, match=r"T=\d+") as exc_info:
        _apply_rope(x, freqs)
    msg = str(exc_info.value)
    assert f"T={T}" in msg
    assert f"max_len={max_len}" in msg


def test_rope_seq_len_at_max_len_ok() -> None:
    """_apply_rope succeeds when T == max_len."""
    max_len = 8
    head_dim = 8
    freqs = _precompute_rope_freqs(head_dim=head_dim, max_len=max_len)
    x = torch.randn(1, max_len, 2, head_dim)
    out = _apply_rope(x, freqs)
    assert out.shape == x.shape


def test_transformer_encoder_rope_seq_exceeds_max_len_raises() -> None:
    """TransformerEncoder with rope=True raises when forward T > max_len."""
    # d_model=16, nheads=2 -> head_dim=8 (multiple of 8, even)
    max_len = 4
    enc = TransformerEncoder(d_model=16, nheads=2, rope=True, max_len=max_len)
    x = torch.randn(1, max_len + 1, 16)
    with pytest.raises(ValueError, match="max_len"):
        enc(x)
