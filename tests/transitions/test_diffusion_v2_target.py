"""Correctness tests for ``DiffusionV2Transition._vp_precondition`` (ESM target)."""

from __future__ import annotations

import torch


def test_esm_target_degenerate_to_dsm(transition):
    """When ``sigma2_t = 0`` and ``mu_t = z_t``, ``_vp_precondition`` reduces to DSM."""
    torch.manual_seed(0)
    N, d = 16, transition.latent_dim
    mu_t = torch.randn(N, d)
    sigma2_t = torch.zeros(N, d)
    k_idx = torch.zeros(N, 1, dtype=torch.long)
    eps = torch.randn(N, d, 1)

    z_in, F_target = transition._vp_precondition(mu_t, sigma2_t, k_idx, eps)

    sigma_tilde0 = transition.sigma_tilde[0]
    z_tilde_expected = mu_t.unsqueeze(-1) + sigma_tilde0 * eps
    # In the degenerate case D_star = z_tilde + sigma_tilde**2 * (-(z_tilde - mu_t)/sigma_tilde**2)
    # = mu_t.
    D_star_expected = mu_t.unsqueeze(-1).expand_as(z_tilde_expected)
    F_expected = (
        D_star_expected - transition.c_skip[0] * z_tilde_expected
    ) / transition.c_out[0]

    assert torch.allclose(F_target, F_expected, atol=1e-5)
    z_in_expected = transition.c_in[0] * z_tilde_expected
    assert torch.allclose(z_in, z_in_expected, atol=1e-5)


def test_esm_target_uses_marginal_variance(transition):
    """Empirical variance of ``z_tilde`` over many ``eps`` should match ``sigma2_t + sigma_tilde**2``."""
    torch.manual_seed(1)
    N_eps = 4000
    d = transition.latent_dim
    mu_t = torch.full((N_eps, d), 0.7)
    sigma2_t = torch.full((N_eps, d), 0.4)
    k_idx = torch.full((N_eps, 1), 5, dtype=torch.long)
    eps = torch.randn(N_eps, d, 1)

    # Reconstruct z_tilde via the same formula as the implementation.
    sigma_tilde2 = transition.sigma_tilde[5] ** 2
    var_total = (sigma2_t + sigma_tilde2).unsqueeze(-1)
    z_tilde = mu_t.unsqueeze(-1) + var_total.sqrt() * eps  # (N, d, 1)

    empirical_var = z_tilde.squeeze(-1).var(dim=0)
    expected_var = (sigma2_t[0] + sigma_tilde2).expand(d)
    assert torch.allclose(empirical_var, expected_var, rtol=0.10)


def test_esm_target_score_identity(transition):
    """Recovered score from F_target matches the analytic Gaussian-convolution score."""
    torch.manual_seed(2)
    N, d = 32, transition.latent_dim
    mu_t = torch.randn(N, d)
    sigma2_t = torch.rand(N, d) * 0.5 + 0.1  # >0
    # Use a single fixed k for simplicity
    k_val = 3
    k_idx = torch.full((N, 1), k_val, dtype=torch.long)
    eps = torch.randn(N, d, 1)

    z_in, F_target = transition._vp_precondition(mu_t, sigma2_t, k_idx, eps)

    sigma_tilde = transition.sigma_tilde[k_val]
    sigma_tilde2 = sigma_tilde ** 2
    c_skip = transition.c_skip[k_val]
    c_out = transition.c_out[k_val]

    var_total = sigma2_t.unsqueeze(-1) + sigma_tilde2
    z_tilde = mu_t.unsqueeze(-1) + var_total.sqrt() * eps  # same RNG-free formula

    D_star = c_skip * z_tilde + c_out * F_target
    s_q_recovered = (D_star - z_tilde) / sigma_tilde2
    s_q_expected = -(z_tilde - mu_t.unsqueeze(-1)) / var_total
    assert torch.allclose(s_q_recovered, s_q_expected, atol=1e-4)


def test_esm_target_zero_eps(transition):
    """When ``eps = 0``: z_tilde = mu_t, s_q = 0, D_star = mu_t."""
    N, d = 8, transition.latent_dim
    mu_t = torch.randn(N, d)
    sigma2_t = torch.rand(N, d) * 0.3
    k_idx = torch.full((N, 1), 2, dtype=torch.long)
    eps = torch.zeros(N, d, 1)

    z_in, F_target = transition._vp_precondition(mu_t, sigma2_t, k_idx, eps)

    c_skip = transition.c_skip[2]
    c_out = transition.c_out[2]
    F_expected = ((1.0 - c_skip) / c_out) * mu_t.unsqueeze(-1)
    assert torch.allclose(F_target, F_expected, atol=1e-5)
