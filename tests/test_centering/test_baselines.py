"""Unit tests for :mod:`ddssm.centering.baselines`."""

from __future__ import annotations

import torch
import pytest

from ddssm.model.centering.baselines import (
    BaseBaseline,
    ZeroBaseline,
    PersistenceBaseline,
)

B = 4
D = 3
J_VALUES = [1, 2]


def _all_forms(latent_dim: int, j: int) -> list[BaseBaseline]:
    return [
        ZeroBaseline(latent_dim=latent_dim, j=j),
        PersistenceBaseline(latent_dim=latent_dim, j=j),
    ]


@pytest.mark.parametrize("j", J_VALUES)
def test_baselines_mean_shape(j: int) -> None:
    """Both parameter-free forms return mu with shape (B, d)."""
    z_hist = torch.randn(B, D, j)
    for baseline in _all_forms(latent_dim=D, j=j):
        mu = baseline.mean(z_hist)
        assert mu.shape == (B, D), f"{type(baseline).__name__}: mu shape {mu.shape}"


@pytest.mark.parametrize("j", J_VALUES)
def test_baselines_mean_and_logvar_shape(j: int) -> None:
    """Both parameter-free forms return (mu, logvar) each of shape (B, d)."""
    z_hist = torch.randn(B, D, j)
    for baseline in _all_forms(latent_dim=D, j=j):
        mu, logvar = baseline.mean_and_logvar(z_hist)
        assert mu.shape == (B, D)
        assert logvar.shape == (B, D)


@pytest.mark.parametrize("j", J_VALUES)
def test_zero_baseline_mean_is_zero(j: int) -> None:
    """ZeroBaseline.mean returns exact zeros (μ_p ≡ 0, no parameters)."""
    z_hist = torch.randn(B, D, j)
    baseline = ZeroBaseline(latent_dim=D, j=j)
    mu = baseline.mean(z_hist)
    assert torch.equal(mu, torch.zeros_like(mu))


@pytest.mark.parametrize("j", J_VALUES)
def test_persistence_baseline_mean_is_last_slot(j: int) -> None:
    """PersistenceBaseline.mean returns ``z_hist[..., -1]`` (last-value-carried-forward)."""
    z_hist = torch.randn(B, D, j)
    baseline = PersistenceBaseline(latent_dim=D, j=j)
    mu = baseline.mean(z_hist)
    assert torch.equal(mu, z_hist[..., -1])


@pytest.mark.parametrize("j", J_VALUES)
def test_baselines_are_parameter_free(j: int) -> None:
    """Zero / Persistence baselines carry no parameters (post-refactor contract)."""
    for baseline in _all_forms(latent_dim=D, j=j):
        assert list(baseline.parameters()) == [], (
            f"{type(baseline).__name__} unexpectedly has parameters"
        )


@pytest.mark.parametrize("j", J_VALUES)
def test_baselines_init_logvar_is_zero(j: int) -> None:
    """At init every baseline emits ``log σ_p² ≡ 0`` (σ_p² = 1)."""
    z_hist = torch.randn(B, D, j)
    for baseline in _all_forms(latent_dim=D, j=j):
        _, logvar = baseline.mean_and_logvar(z_hist)
        assert torch.allclose(logvar, torch.zeros_like(logvar), atol=1e-5), (
            f"{type(baseline).__name__}: init logvar {logvar.abs().max().item()=}"
        )


@pytest.mark.parametrize("j", J_VALUES)
def test_baselines_reject_wrong_shape(j: int) -> None:
    """Inputs with the wrong shape are rejected."""
    for baseline in _all_forms(latent_dim=D, j=j):
        with pytest.raises(ValueError):
            baseline.mean(torch.randn(B, D + 1, j))
        with pytest.raises(ValueError):
            baseline.mean(torch.randn(B, D))  # wrong rank
        if j > 1:
            with pytest.raises(ValueError):
                baseline.mean(torch.randn(B, D, j - 1))
