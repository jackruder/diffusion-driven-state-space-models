"""Forward noising tests for ``DiffusionV2Transition`` (VP-SDE → N(0, I))."""

from __future__ import annotations

import torch
import pytest


def test_forward_marginal_at_tau_max_is_standard_normal(transition):
    """Forward marginal at the largest tau index should be approximately N(0, I)."""
    torch.manual_seed(0)
    N = 10_000
    d = transition.latent_dim
    z0 = 3.0 + 2.0 * torch.randn(N, d)

    k_last = transition.num_steps - 1
    alpha_last = transition.alpha[k_last]
    sigma_last = torch.sqrt(1.0 - alpha_last ** 2)
    eps = torch.randn_like(z0)
    z_tau = alpha_last * z0 + sigma_last * eps

    empirical_mean = z_tau.mean(dim=0)
    empirical_var = z_tau.var(dim=0)
    assert empirical_mean.abs().max().item() < 0.15, f"mean={empirical_mean}"
    assert (empirical_var - 1.0).abs().max().item() < 0.20, f"var={empirical_var}"


def test_forward_marginal_at_intermediate_tau_matches_closed_form(transition):
    """At any intermediate tau, noised stats match the closed-form Gaussian formula."""
    torch.manual_seed(1)
    d = transition.latent_dim
    mu_0 = torch.tensor([1.5, -0.5])[:d]
    sigma2_0 = torch.tensor([0.25, 4.0])[:d]
    k = transition.num_steps // 2

    alpha_k = transition.alpha[k]
    sigma_k = torch.sqrt(1.0 - alpha_k ** 2)

    N = 50_000
    z0 = mu_0 + sigma2_0.sqrt() * torch.randn(N, d)
    eps = torch.randn_like(z0)
    z_tau = alpha_k * z0 + sigma_k * eps

    expected_mean = alpha_k * mu_0
    expected_var = alpha_k ** 2 * sigma2_0 + sigma_k ** 2
    assert torch.allclose(z_tau.mean(dim=0), expected_mean, atol=0.05)
    assert torch.allclose(z_tau.var(dim=0), expected_var, rtol=0.05)


def test_forward_kernel_score_matches_analytical(transition):
    """In the degenerate ESM (mu_t = z_0, sigma2_t = 0) the recovered score is the
    analytic VE-coord DSM kernel score ``-(z_tilde - z_0) / sigma_tilde**2``.
    """
    torch.manual_seed(2)
    N, d = 64, transition.latent_dim
    z0 = torch.randn(N, d)
    sigma2_t = torch.zeros(N, d)
    k_val = 4
    k_idx = torch.full((N, 1), k_val, dtype=torch.long)
    eps = torch.randn(N, d, 1)

    _, F_target = transition._vp_precondition(z0, sigma2_t, k_idx, eps)

    sigma_tilde = transition.sigma_tilde[k_val]
    sigma_tilde2 = sigma_tilde ** 2
    c_skip = transition.c_skip[k_val]
    c_out = transition.c_out[k_val]

    z_tilde = z0.unsqueeze(-1) + sigma_tilde * eps
    D_star = c_skip * z_tilde + c_out * F_target
    s_q_recovered = (D_star - z_tilde) / sigma_tilde2
    s_q_expected = -(z_tilde - z0.unsqueeze(-1)) / sigma_tilde2
    assert torch.allclose(s_q_recovered, s_q_expected, atol=1e-4)


@pytest.mark.slow
def test_terminal_marginal_independent_of_initial_distribution(transition):
    """Two very different inits converge to ~N(0, I) at the terminal tau."""
    torch.manual_seed(3)
    N = 20_000
    d = transition.latent_dim
    k_last = transition.num_steps - 1
    alpha_last = transition.alpha[k_last]
    sigma_last = torch.sqrt(1.0 - alpha_last ** 2)

    z0_a = 10.0 + torch.randn(N, d)
    z0_b = -5.0 + 3.0 * torch.randn(N, d)
    z_a = alpha_last * z0_a + sigma_last * torch.randn_like(z0_a)
    z_b = alpha_last * z0_b + sigma_last * torch.randn_like(z0_b)

    assert (z_a.mean(dim=0) - z_b.mean(dim=0)).abs().max().item() < 0.3
    assert alpha_last.item() < 0.05
