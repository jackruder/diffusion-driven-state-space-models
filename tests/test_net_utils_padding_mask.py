"""Unit tests for the padding-mask channel in :func:`ddssm.net_utils.get_side_info`.

Per ``init-experiment.org`` § Implementation precursors and
``model-v2.org`` § Padding mask in the diffusion side-info tensor.
"""

from __future__ import annotations

import torch
import pytest
import torch.nn as nn

from ddssm.nn.net_utils import get_side_info

B = 2
T = 5
E_t = 8
E_f = 4
D = 3


def _embed_layer() -> nn.Embedding:
    return nn.Embedding(D, E_f)


def test_get_side_info_no_padding_mask_unchanged() -> None:
    """Without the new kwarg the shape and content are unchanged."""
    torch.manual_seed(0)
    time_embed = torch.randn(B, T, E_t)
    emb = _embed_layer()
    out_no_kwarg = get_side_info(data_dim=D, time_embed=time_embed, embed_layer=emb)
    out_with_none = get_side_info(
        data_dim=D,
        time_embed=time_embed,
        embed_layer=emb,
        padding_mask=None,
    )
    assert out_no_kwarg.shape == out_with_none.shape
    assert torch.equal(out_no_kwarg, out_with_none)
    # Expected channel count: E_t + E_f (no cond_mask, no padding_mask).
    assert out_no_kwarg.shape == (B, E_t + E_f, D, T)


def test_get_side_info_padding_mask_adds_one_channel() -> None:
    """A non-None ``padding_mask`` increases the channel count by one."""
    torch.manual_seed(0)
    time_embed = torch.randn(B, T, E_t)
    emb = _embed_layer()
    padding_mask = torch.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    side = get_side_info(
        data_dim=D,
        time_embed=time_embed,
        embed_layer=emb,
        padding_mask=padding_mask,
    )
    # Expected channel count grows by 1.
    assert side.shape == (B, E_t + E_f + 1, D, T)


def test_get_side_info_padding_mask_broadcasts_across_d() -> None:
    """The padding channel's value at (b, t) is the same for every d."""
    torch.manual_seed(0)
    time_embed = torch.randn(B, T, E_t)
    emb = _embed_layer()
    padding_mask = torch.tensor([
        [1.0, 0.5, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.5],
    ])
    side = get_side_info(
        data_dim=D,
        time_embed=time_embed,
        embed_layer=emb,
        padding_mask=padding_mask,
    )
    # The last channel should equal padding_mask broadcast across D.
    last_channel = side[:, -1, :, :]  # (B, D, T)
    expected = padding_mask.unsqueeze(1).expand(B, D, T)
    assert torch.equal(last_channel, expected)


def test_get_side_info_padding_mask_and_cond_mask_coexist() -> None:
    """Both optional masks can be supplied; the order in channels is
    [time, feat, cond, padding].
    """
    torch.manual_seed(0)
    time_embed = torch.randn(B, T, E_t)
    emb = _embed_layer()
    cond_mask = torch.randint(0, 2, (B, D, T)).float()
    padding_mask = torch.randint(0, 2, (B, T)).float()
    side = get_side_info(
        data_dim=D,
        time_embed=time_embed,
        embed_layer=emb,
        cond_mask=cond_mask,
        padding_mask=padding_mask,
    )
    assert side.shape == (B, E_t + E_f + 2, D, T)
    # Penultimate channel is cond_mask.
    assert torch.equal(side[:, -2, :, :], cond_mask)
    # Last channel is padding_mask broadcast across D.
    assert torch.equal(side[:, -1, :, :], padding_mask.unsqueeze(1).expand(B, D, T))


def test_get_side_info_padding_mask_wrong_shape_raises() -> None:
    """Padding masks with the wrong shape are rejected with ``ValueError``."""
    time_embed = torch.randn(B, T, E_t)
    emb = _embed_layer()
    with pytest.raises(ValueError):
        get_side_info(
            data_dim=D,
            time_embed=time_embed,
            embed_layer=emb,
            padding_mask=torch.zeros(B, T + 1),
        )
    with pytest.raises(ValueError):
        get_side_info(
            data_dim=D,
            time_embed=time_embed,
            embed_layer=emb,
            padding_mask=torch.zeros(B, D, T),  # 3D instead of 2D
        )
