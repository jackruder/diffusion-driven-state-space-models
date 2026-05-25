"""Importance sampling tests for ``DiffusionV2Transition``."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from ddssm.transitions.diffusion_v2 import DiffusionV2ScheduleConfig

from .conftest import make_transition, compute_per_sample_loss


def _build_pair(num_steps: int = 16):
    """Build matching uniform / lsgm_is transitions sharing weights."""
    torch.manual_seed(0)
    cfg_uni = DiffusionV2ScheduleConfig(
        S_k=1, k_chunk=1, num_steps=num_steps, k_sampling_mode="uniform",
    )
    cfg_is = DiffusionV2ScheduleConfig(
        S_k=1, k_chunk=1, num_steps=num_steps, k_sampling_mode="lsgm_is",
    )
    t_uni = make_transition(schedule=cfg_uni)
    t_is = make_transition(schedule=cfg_is)
    t_is.load_state_dict(t_uni.state_dict())
    return t_uni, t_is


@pytest.mark.slow
def test_lsgm_is_unbiased(fixed_batch):
    """Uniform and LSGM-IS yield the same expected loss (within MC tolerance)."""
    t_uni, t_is = _build_pair(num_steps=16)
    N = 200
    losses_uni = compute_per_sample_loss(t_uni, fixed_batch, N, seed=100)
    losses_is = compute_per_sample_loss(t_is, fixed_batch, N, seed=200)
    sem = max(np.std(losses_uni), np.std(losses_is)) / np.sqrt(N)
    assert abs(np.mean(losses_uni) - np.mean(losses_is)) < 6.0 * max(sem, 1e-8)


@pytest.mark.slow
def test_lsgm_is_variance_not_explosive(fixed_batch):
    """Smoke check: IS variance should not blow up vs. uniform."""
    t_uni, t_is = _build_pair(num_steps=16)
    N = 200
    losses_uni = compute_per_sample_loss(t_uni, fixed_batch, N, seed=300)
    losses_is = compute_per_sample_loss(t_is, fixed_batch, N, seed=400)
    # With an untrained network the IS-vs-uniform variance ordering can go
    # either way; we only require IS not to be catastrophically worse.
    assert np.var(losses_is) <= 5.0 * np.var(losses_uni) + 1e-8


def test_p_k_lsgm_formula():
    """Verify ``p_k`` in lsgm_is mode matches the closed-form ``beta / (1 - alpha**2)``."""
    cfg = DiffusionV2ScheduleConfig(
        num_steps=20, k_sampling_mode="lsgm_is", pk_gamma=1.0, pk_floor=1e-12
    )
    t = make_transition(schedule=cfg)
    expected = t.beta.double() / (1.0 - t.alpha.double() ** 2).clamp_min(1e-30)
    expected = expected.clamp_min(1e-12).float()
    expected = expected / expected.sum()
    assert torch.allclose(t.p_k, expected, atol=1e-4)
