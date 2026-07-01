"""Smoke tests for :mod:`ddssm.dist_heads`."""

from __future__ import annotations

import torch

from ddssm.nn.dist_heads import GaussianDistHead

B = 3
D = 4
IN = 16


def test_gaussian_dist_head_forward_shape():
    head = GaussianDistHead(in_features=IN, latent_dim=D)
    assert head.is_gaussian_family
    x = torch.randn(B, IN)
    z, logq, step_params = head(x)
    assert z.shape == (B, D)
    assert logq.shape == (B,)
    assert step_params.keys() == {"mu", "logvar"}
    assert step_params["mu"].shape == (B, D)
    assert step_params["logvar"].shape == (B, D)


def test_gaussian_dist_head_mean_offset_shifts_mu():
    """mean_offset adds the persistence frame: μ = free-μ + offset (exactly), with
    σ and the sampling noise unchanged. Backs the GaussianEncoder additive frame.
    """
    head = GaussianDistHead(in_features=IN, latent_dim=D)
    x = torch.randn(B, IN)
    offset = torch.randn(B, D)
    torch.manual_seed(0)
    z0, _, p0 = head(x)
    torch.manual_seed(0)
    z1, _, p1 = head(x, mean_offset=offset)
    assert torch.allclose(p1["mu"], p0["mu"] + offset, atol=1e-6)
    assert torch.allclose(p1["logvar"], p0["logvar"], atol=1e-6)
    assert torch.allclose(z1, z0 + offset, atol=1e-6)  # same eps, shifted mean


def test_gaussian_dist_head_stack_stats():
    head = GaussianDistHead(in_features=IN, latent_dim=D)
    T = 5
    step_params_list = []
    for _ in range(T):
        x = torch.randn(B, IN)
        _, _, sp = head(x)
        step_params_list.append(sp)
    stats = head.stack_stats(step_params_list)
    assert stats.keys() == {"mus", "logvars"}
    assert stats["mus"].shape == (B, D, T)
    assert stats["logvars"].shape == (B, D, T)


def test_gaussian_dist_head_entropy():
    head = GaussianDistHead(in_features=IN, latent_dim=D)
    T = 5
    S = 2
    stats = {
        "mus": torch.randn(B, S, D, T),
        "logvars": torch.randn(B, S, D, T),
    }
    e_init = head.entropy_init(stats, steps=2)
    e_trans = head.entropy_transition(stats, j=2)
    assert e_init.ndim == 0
    assert e_trans.ndim == 0
    assert torch.isfinite(e_init) and torch.isfinite(e_trans)
