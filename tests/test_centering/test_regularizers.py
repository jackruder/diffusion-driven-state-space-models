"""Unit tests for :mod:`ddssm.centering.regularizers`."""

from __future__ import annotations

import torch

from ddssm.centering.baselines import IdentityBaseline, MLPBaseline
from ddssm.centering.regularizers import r_mu_p_loss, r_sigma_p_loss


B = 4
D = 3


def test_r_sigma_p_zero_at_unit_variance() -> None:
    """``R_σp`` is zero when ``log σ_p² ≡ 0`` (σ_p ≡ 1)."""
    baseline = IdentityBaseline(latent_dim=D, j=1)
    # Manually zero the sigma_head's output by zeroing its final layer.
    for layer in reversed(list(baseline.sigma_head.body)):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            break
    z_hist = torch.randn(B, D, 1)
    val = r_sigma_p_loss(baseline, z_hist, lambda_sigma_p=1.0)
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-6)


def test_r_sigma_p_zero_when_lambda_zero() -> None:
    """``R_σp`` is zero when ``λ_σp = 0``."""
    baseline = MLPBaseline(latent_dim=D, j=2, hidden_dim=8, n_layers=2)
    z_hist = torch.randn(B, D, 2)
    val = r_sigma_p_loss(baseline, z_hist, lambda_sigma_p=0.0)
    assert torch.isclose(val, torch.tensor(0.0))


def test_r_sigma_p_grows_with_lambda() -> None:
    """``R_σp`` scales linearly with λ_σp."""
    torch.manual_seed(0)
    baseline = MLPBaseline(latent_dim=D, j=1, hidden_dim=8, n_layers=2)
    # Force a non-trivial mean(log σ_p²) by editing the logvar head's bias.
    with torch.no_grad():
        baseline.logvar_head.bias.fill_(1.0)
    z_hist = torch.randn(B, D, 1)
    val_lo = r_sigma_p_loss(baseline, z_hist, lambda_sigma_p=1.0)
    val_hi = r_sigma_p_loss(baseline, z_hist, lambda_sigma_p=4.0)
    assert torch.isclose(val_hi, 4.0 * val_lo, atol=1e-6)


def test_r_mu_p_zero_immediately_after_snapshot() -> None:
    """``R_μp`` is exactly zero against a fresh snapshot."""
    torch.manual_seed(0)
    baseline = MLPBaseline(latent_dim=D, j=1, hidden_dim=8, n_layers=2)
    anchor = baseline.snapshot()
    z_hist = torch.randn(B, D, 1)
    val = r_mu_p_loss(baseline, anchor, z_hist, lambda_mu_p=2.5)
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-6)


def test_r_mu_p_grows_with_drift() -> None:
    """``R_μp`` increases when μ_p drifts from μ_p^(0)."""
    torch.manual_seed(0)
    baseline = MLPBaseline(latent_dim=D, j=1, hidden_dim=8, n_layers=2)
    anchor = baseline.snapshot()
    # Drift the live baseline's μ-head.
    with torch.no_grad():
        baseline.mu_head.bias.add_(0.5)
    z_hist = torch.randn(B, D, 1)
    val = r_mu_p_loss(baseline, anchor, z_hist, lambda_mu_p=1.0)
    assert float(val.item()) > 0.0


def test_r_mu_p_zero_when_lambda_zero() -> None:
    """``R_μp`` is zero when ``λ_μp = 0``."""
    baseline = MLPBaseline(latent_dim=D, j=1, hidden_dim=8, n_layers=2)
    anchor = baseline.snapshot()
    # Drift the live baseline so the anchor distance is non-trivial.
    with torch.no_grad():
        baseline.mu_head.bias.add_(0.5)
    z_hist = torch.randn(B, D, 1)
    val = r_mu_p_loss(baseline, anchor, z_hist, lambda_mu_p=0.0)
    assert torch.isclose(val, torch.tensor(0.0))


def test_r_mu_p_anchor_gets_no_gradient() -> None:
    """Backprop through ``R_μp`` updates only μ_p, not the anchor."""
    torch.manual_seed(0)
    baseline = MLPBaseline(latent_dim=D, j=1, hidden_dim=8, n_layers=2)
    anchor = baseline.snapshot()
    # Drift live μ_p so the loss is non-zero.
    with torch.no_grad():
        baseline.mu_head.bias.add_(0.5)
    z_hist = torch.randn(B, D, 1)
    val = r_mu_p_loss(baseline, anchor, z_hist, lambda_mu_p=1.0)
    val.backward()
    assert baseline.mu_head.weight.grad is not None
    assert all(p.grad is None for p in anchor.parameters())
