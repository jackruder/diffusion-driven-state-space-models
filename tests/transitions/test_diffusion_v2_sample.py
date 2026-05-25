"""Sampler tests for ``DiffusionV2Transition``."""

from __future__ import annotations

import torch
import pytest
import torch.nn as nn

from .conftest import EMB_TIME, LATENT_DIM, J, make_dummy_ctx


def test_sample_shape_and_finiteness(transition):
    """Sampler returns shape (B, 1, d) with finite entries."""
    torch.manual_seed(0)
    B = 4
    z_hist = torch.randn(B, LATENT_DIM, J)
    ctx = make_dummy_ctx(B, J, EMB_TIME)
    out = transition.sample(z_hist, ctx=ctx)
    assert out.shape == (B, 1, LATENT_DIM)
    assert torch.all(torch.isfinite(out))


def test_sample_deterministic_with_fixed_seed(transition):
    """Two sample calls with the same RNG seed produce identical results."""
    B = 3
    z_hist = torch.randn(B, LATENT_DIM, J)
    ctx = make_dummy_ctx(B, J, EMB_TIME)
    transition.eval()
    torch.manual_seed(42)
    a = transition.sample(z_hist, ctx=ctx)
    torch.manual_seed(42)
    b = transition.sample(z_hist, ctx=ctx)
    assert torch.equal(a, b)


def test_pf_ode_step_count(transition):
    """The score net is queried exactly ``num_steps - 1`` times during sampling."""
    B = 2
    z_hist = torch.randn(B, LATENT_DIM, J)
    ctx = make_dummy_ctx(B, J, EMB_TIME)

    counter = {"n": 0}
    original = transition.diffmodel

    class Counting(nn.Module):
        def forward(self, latent_w, side_w, c_noise):
            counter["n"] += 1
            return original(latent_w, side_w, c_noise)

    transition.diffmodel = Counting()
    try:
        transition.sample(z_hist, ctx=ctx)
    finally:
        transition.diffmodel = original

    assert counter["n"] == transition.num_steps - 1


# ---------------------------------------------------------------------------
# Oracle-score sampler test (gold-standard correctness check).
# ---------------------------------------------------------------------------


class _OracleGaussian(nn.Module):
    """Oracle that returns ``F`` such that the recovered VE-coords score is the
    analytic score of a target Gaussian ``N(mu_tgt, sigma_tgt**2 I)``.
    """

    def __init__(self, transition, mu_tgt: torch.Tensor, sigma_tgt: float):
        super().__init__()
        self.transition = transition
        self.register_buffer("mu_tgt", mu_tgt)
        self.sigma_tgt = float(sigma_tgt)

    def forward(self, latent_w, side_w, c_noise):
        # latent_w: (B, d, j+1); last-channel along time = c_in * z_tilde.
        # Recover sigma_tilde from c_noise = 0.25 * log(sigma_tilde) -> sigma_tilde = exp(4*c_noise).
        sigma_tilde = torch.exp(4.0 * c_noise)            # (B,)
        sigma_tilde2 = sigma_tilde * sigma_tilde
        # Reconstruct alpha from VP identity: sigma_tilde**2 = (1 - alpha**2)/alpha**2
        # -> alpha**2 = 1 / (1 + sigma_tilde**2).
        alpha2 = 1.0 / (1.0 + sigma_tilde2)
        alpha = alpha2.sqrt()                             # (B,) = c_in
        c_skip = alpha2                                   # (B,)
        c_out = (1.0 - alpha2).clamp_min(1e-12).sqrt()    # (B,)

        # Recover z_tilde
        z_in_last = latent_w[..., -1]                     # (B, d)
        z_tilde = z_in_last / alpha.unsqueeze(-1).clamp_min(1e-12)

        # VE-coord true score: -(z_tilde - mu_tgt) / (sigma_tgt**2 + sigma_tilde**2)
        var_total = self.sigma_tgt ** 2 + sigma_tilde2    # (B,)
        s_pred = -(z_tilde - self.mu_tgt) / var_total.unsqueeze(-1)
        D_pred = z_tilde + sigma_tilde2.unsqueeze(-1) * s_pred
        F_pred = (D_pred - c_skip.unsqueeze(-1) * z_tilde) / c_out.unsqueeze(-1)
        return F_pred.unsqueeze(-1)                       # (B, d, 1)


@pytest.mark.slow
def test_sample_recovers_known_unimodal_gaussian():
    """Oracle-score sampler recovers a known target Gaussian within MC tolerance."""
    from ddssm.transitions.diffusion_v2 import DiffusionV2ScheduleConfig

    from .conftest import make_transition

    torch.manual_seed(0)
    cfg = DiffusionV2ScheduleConfig(
        S_k=1, k_chunk=1, num_steps=100,
        beta_min=0.1, beta_max=20.0, tau_min=1e-3,
        k_sampling_mode="uniform",
    )
    transition = make_transition(schedule=cfg)
    transition.eval()

    mu_tgt = torch.tensor([1.5, -1.0])
    sigma_tgt = 0.7
    transition.diffmodel = _OracleGaussian(transition, mu_tgt, sigma_tgt)

    N = 1500
    z_hist = torch.zeros(N, LATENT_DIM, J)
    ctx = make_dummy_ctx(N, J, EMB_TIME)

    torch.manual_seed(123)
    samples = transition.sample(z_hist, ctx=ctx).squeeze(1)  # (N, d)

    emp_mean = samples.mean(dim=0)
    emp_std = samples.std(dim=0)
    # Loose tolerances for a probability-flow Euler sampler at modest K.
    assert torch.allclose(emp_mean, mu_tgt, atol=0.20), f"mean={emp_mean}"
    assert (emp_std - sigma_tgt).abs().max().item() < 0.25, f"std={emp_std}"
