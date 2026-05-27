"""Schedule and precompute tests for ``DiffusionV2Transition``."""

from __future__ import annotations

import torch
import pytest

from ddssm.transitions.diffusion_v2 import (
    DiffusionV2ScheduleConfig,
)

from .conftest import make_transition


def test_schedule_monotonicity(transition):
    """Verify monotonicity / boundary properties of the precomputed schedule."""
    alpha = transition.alpha
    sigma_tilde = transition.sigma_tilde
    K = transition.num_steps

    # alpha monotonically non-increasing
    assert torch.all(alpha[1:] <= alpha[:-1] + 1e-7)
    # sigma_tilde monotonically non-decreasing
    assert torch.all(sigma_tilde[1:] >= sigma_tilde[:-1] - 1e-7)
    # boundary magnitudes
    assert sigma_tilde[0].item() < 0.5
    assert sigma_tilde[-1].item() > 5.0
    # all buffers length num_steps and finite
    for name in (
        "alpha",
        "sigma_tilde",
        "beta",
        "tau",
        "c_skip",
        "c_out",
        "c_in",
        "c_noise",
        "wtilde",
        "p_k",
    ):
        buf = getattr(transition, name)
        assert buf.shape == (K,), f"{name} shape={buf.shape}"
        assert torch.all(torch.isfinite(buf)), f"{name} not finite"


def test_vp_variance_preserving_identity(transition):
    """For VP-SDE: ``alpha**2 + sigma**2 == 1`` exactly (sigma**2 = 1 - alpha**2)."""
    alpha = transition.alpha.double()
    # In VP, sigma**2 := 1 - alpha**2 by construction; sigma_tilde**2 = sigma**2/alpha**2.
    # Verify via the EDM constants: c_out**2 = 1 - alpha**2.
    c_out = transition.c_out.double()
    assert torch.allclose(c_out * c_out + alpha * alpha, torch.ones_like(alpha), atol=1e-5)


def test_edm_constants_consistency(transition):
    """EDM identities with sigma_data=1: c_skip + c_out**2 == 1, c_in*sigma_tilde == c_out."""
    c_skip = transition.c_skip.double()
    c_out = transition.c_out.double()
    c_in = transition.c_in.double()
    sigma_tilde = transition.sigma_tilde.double()
    c_noise = transition.c_noise.double()

    # c_skip = alpha**2; c_out**2 = 1 - alpha**2  =>  c_skip + c_out**2 = 1
    assert torch.allclose(c_skip + c_out * c_out, torch.ones_like(c_skip), atol=1e-5)
    # c_in = alpha; c_in * sigma_tilde = alpha * (sigma/alpha) = sigma = c_out
    assert torch.allclose(c_in * sigma_tilde, c_out, atol=1e-5)
    # c_noise finite and monotone in sigma_tilde
    assert torch.all(torch.isfinite(c_noise))
    assert torch.all(c_noise[1:] >= c_noise[:-1] - 1e-7)


def test_invalid_config_rejected():
    """Schedule validation catches out-of-range / unknown values."""
    with pytest.raises(ValueError):
        make_transition(schedule=DiffusionV2ScheduleConfig(tau_min=0.0))
    with pytest.raises(ValueError):
        make_transition(schedule=DiffusionV2ScheduleConfig(tau_min=1.0))
    with pytest.raises(ValueError):
        make_transition(
            schedule=DiffusionV2ScheduleConfig(beta_min=10.0, beta_max=5.0)
        )
    with pytest.raises(ValueError):
        make_transition(
            schedule=DiffusionV2ScheduleConfig(k_sampling_mode="bogus")
        )
    with pytest.raises(ValueError):
        make_transition(schedule=DiffusionV2ScheduleConfig(objective="bogus"))


def test_p_k_normalization(transition):
    """``p_k`` sums to 1, all positive, finite."""
    p_k = transition.p_k
    assert torch.allclose(p_k.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(p_k > 0)
    assert torch.all(torch.isfinite(p_k))


def test_p_k_uniform_mode(transition):
    """In ``uniform`` mode, all p_k entries equal ``1/K``."""
    K = transition.num_steps
    expected = torch.full((K,), 1.0 / K)
    assert torch.allclose(transition.p_k, expected, atol=1e-5)


def test_p_k_lsgm_is_formula():
    """LSGM IS p_k matches ``beta / (1 - alpha**2)`` up to normalization."""
    cfg = DiffusionV2ScheduleConfig(
        num_steps=20, k_sampling_mode="lsgm_is", pk_gamma=1.0, pk_floor=1e-12
    )
    t = make_transition(schedule=cfg)
    expected = t.beta.double() / (1.0 - t.alpha.double() ** 2).clamp_min(1e-30)
    expected = (expected.clamp_min(1e-12)).float()
    expected = expected / expected.sum()
    assert torch.allclose(t.p_k, expected, atol=1e-4)
