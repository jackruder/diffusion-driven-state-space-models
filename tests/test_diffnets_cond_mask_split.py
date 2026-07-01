"""Regression tests for the CSDI U-Net history/noise split.

The denoiser splits its input window into a clean-history stream and a noised
stream using the *conditioning* mask (``cond_mask == 1`` on history slots,
``0`` on the noised target slot). The side-info channel layout is
``[time(+cov), feat, cond_mask, padding_mask]``, so ``cond_mask`` is the
second-to-last channel.

A prior bug read ``side_info[:, -1]`` — the *padding* mask — which on the main
(t≥2) training path is all zeros. That silently routed the entire window into
the noisy stream and zeroed the clean-history stream on every forward pass.
These tests pin the split to the conditioning mask via its explicit index.
"""

from __future__ import annotations

from functools import partial

import torch

from ddssm.nn.diffnets import (
    CSDIUnet,
    FeatureMixerConfig,
    DiffResidualBlockConfig,
)
from ddssm.model.centering.baselines import MLPBaseline
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)


def _unet(side_dim: int, *, latent_dim: int = 2, history_len: int = 2) -> CSDIUnet:
    return CSDIUnet(
        output_len=1,
        diffusion_steps=10,
        latent_dim=latent_dim,
        latent_history_len=history_len,
        side_dim=side_dim,
        channels=16,
        n_layers=1,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=2, n_layers=1)
        ),
    )


def test_cond_mask_channel_defaults_to_second_to_last() -> None:
    """Default index is ``side_dim - 2`` (cond_mask sits before padding_mask)."""
    net = _unet(side_dim=5)
    assert net.cond_mask_channel == 3  # 5 - 2


def test_split_follows_cond_mask_with_all_zero_padding() -> None:
    """The exact regression: padding mask all-zeros (the t≥2 case).

    Under the old ``side_info[:, -1]`` bug the clean-history stream would be
    identically zero here; the split must instead follow the cond mask.
    """
    B, d, L = 2, 2, 3  # history slots [0, 1], target slot [2]
    side_dim = 5
    net = _unet(side_dim=side_dim, latent_dim=d, history_len=2)

    x = torch.arange(1, B * d * L + 1, dtype=torch.float32).reshape(B, d, L)

    cond = torch.tensor([1.0, 1.0, 0.0]).view(1, 1, L).expand(B, d, L)
    side_info = torch.randn(B, side_dim, d, L)
    side_info[:, net.cond_mask_channel] = cond  # cond_mask channel
    side_info[:, -1] = 0.0  # padding mask all zeros — the bug trigger

    x_noisy, x_hist = net._split_history_noise(x, side_info).unbind(dim=1)

    expected_hist = x.clone()
    expected_hist[..., 2] = 0.0  # clean history keeps slots 0,1; zero on target
    expected_noisy = x.clone()
    expected_noisy[..., :2] = 0.0  # noisy stream keeps the target slot only

    assert torch.allclose(x_hist, expected_hist)
    assert torch.allclose(x_noisy, expected_noisy)
    # Regression guard: the clean-history stream is NOT all-zero.
    assert x_hist.abs().sum() > 0


def test_split_ignores_padding_channel() -> None:
    """Changing only the padding-mask channel leaves the split unchanged."""
    B, d, L = 2, 2, 3
    side_dim = 5
    net = _unet(side_dim=side_dim, latent_dim=d, history_len=2)

    x = torch.randn(B, d, L)
    cond = torch.tensor([1.0, 1.0, 0.0]).view(1, 1, L).expand(B, d, L)

    base = torch.randn(B, side_dim, d, L)
    base[:, net.cond_mask_channel] = cond

    si_zeros = base.clone()
    si_zeros[:, -1] = 0.0
    si_ones = base.clone()
    si_ones[:, -1] = 1.0

    split_zeros = net._split_history_noise(x, si_zeros)
    split_ones = net._split_history_noise(x, si_ones)
    assert torch.allclose(split_zeros, split_ones)


def test_split_would_differ_if_it_read_padding() -> None:
    """Sanity check that cond_mask and padding_mask genuinely disagree here.

    Guards against a vacuous test: reading the padding channel instead would
    produce a different split, so the assertions above are discriminating.
    """
    B, d, L = 2, 2, 3
    side_dim = 5
    net = _unet(side_dim=side_dim, latent_dim=d, history_len=2)

    x = torch.randn(B, d, L)
    cond = torch.tensor([1.0, 1.0, 0.0]).view(1, 1, L).expand(B, d, L)
    pad = torch.tensor([0.0, 0.0, 1.0]).view(1, 1, L).expand(B, d, L)  # != cond

    side_info = torch.randn(B, side_dim, d, L)
    side_info[:, net.cond_mask_channel] = cond
    side_info[:, -1] = pad

    correct = net._split_history_noise(x, side_info)
    # Emulate the old bug by pointing the index at the padding channel.
    net.cond_mask_channel = side_dim - 1
    buggy = net._split_history_noise(x, side_info)
    assert not torch.allclose(correct, buggy)


def test_transition_wires_cond_mask_channel() -> None:
    """DiffusionTransition passes the explicit cond_mask index to its U-Net."""
    emb_time = 8
    transition = DiffusionTransition(
        baseline=MLPBaseline(latent_dim=2, j=1),
        latent_dim=2,
        j=1,
        emb_time_dim=emb_time,
        T_max=10,
        unet=partial(
            CSDIUnet,
            channels=16,
            n_layers=1,
            embedding_dim=16,
            residual_block=DiffResidualBlockConfig(
                feature=FeatureMixerConfig(nheads=2, n_layers=1)
            ),
        ),
        schedule=DiffusionScheduleConfig(
            S_k=1,
            k_chunk=1,
            num_steps=20,
            beta_min=0.1,
            beta_max=20.0,
            tau_min=1e-3,
            k_sampling_mode="uniform",
        ),
    )
    assert transition.diffmodel.cond_mask_channel == transition.side_dim - 2
