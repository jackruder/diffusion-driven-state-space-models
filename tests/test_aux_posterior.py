"""Unit tests for :mod:`ddssm.aux_posterior`."""

from __future__ import annotations

import torch
import pytest

from ddssm.nn.aux_posterior import AuxPosterior

B = 3
D = 4


@pytest.mark.parametrize("j", [1, 2, 3])
def test_aux_posterior_forward_shape(j: int) -> None:
    """``forward`` returns the right shape for several j values."""
    aux = AuxPosterior(latent_dim=D, j=j)
    z_init = torch.randn(B, D, j)
    aux_mu, aux_logvar = aux(z_init)
    assert aux_mu.shape == (B, D, j)
    assert aux_logvar.shape == (B, D, j)


@pytest.mark.parametrize("j", [1, 2])
def test_aux_posterior_sample_shape_and_grad(j: int) -> None:
    """Reparameterised ``sample`` returns the right shape and is differentiable."""
    aux = AuxPosterior(latent_dim=D, j=j)
    z_init = torch.randn(B, D, j)
    z_aux, aux_mu, aux_logvar = aux.sample(z_init)
    assert z_aux.shape == (B, D, j)
    assert aux_mu.shape == (B, D, j)
    assert aux_logvar.shape == (B, D, j)

    # Gradient flows through aux_mu and aux_logvar via reparameterisation.
    loss = z_aux.pow(2).sum()
    loss.backward()
    grad_params = [p.grad for p in aux.parameters() if p.grad is not None]
    assert len(grad_params) > 0, "expected gradient on at least one parameter"


def test_aux_posterior_kl_zero_at_standard_normal() -> None:
    """The KL is exactly zero when q_Φ equals the prior N(0, I)."""
    aux_mu = torch.zeros(B, D, 2)
    aux_logvar = torch.zeros(B, D, 2)  # log 1 = 0
    kl = AuxPosterior.kl_against_standard_normal(aux_mu, aux_logvar)
    assert torch.isclose(kl, torch.tensor(0.0))


def test_aux_posterior_kl_matches_analytic() -> None:
    """KL[N(mu, σ²) || N(0,1)] = 0.5 (μ² + σ² − 1 − log σ²)."""
    aux_mu = torch.tensor([[[1.0, -1.0], [0.5, 0.0], [2.0, 0.5], [0.0, 0.0]]])
    aux_logvar = torch.tensor([[[0.0, 0.5], [-0.2, 0.0], [0.1, -0.1], [0.0, 0.0]]])
    # Shape (1, D=4, j=2)
    expected_per_elem = 0.5 * (aux_mu.pow(2) + aux_logvar.exp() - 1.0 - aux_logvar)
    expected = expected_per_elem.sum(dim=(1, 2)).mean()
    got = AuxPosterior.kl_against_standard_normal(aux_mu, aux_logvar)
    assert torch.isclose(got, expected, atol=1e-6)


def test_aux_posterior_kl_grows_with_drift() -> None:
    """KL should increase with |mu| and with |log σ²|."""
    base_mu = torch.zeros(B, D, 2)
    base_lv = torch.zeros(B, D, 2)
    kl0 = AuxPosterior.kl_against_standard_normal(base_mu, base_lv)

    mu_far = torch.full_like(base_mu, 1.0)
    kl1 = AuxPosterior.kl_against_standard_normal(mu_far, base_lv)
    assert kl1 > kl0

    lv_inflated = torch.full_like(base_lv, 2.0)
    kl2 = AuxPosterior.kl_against_standard_normal(base_mu, lv_inflated)
    assert kl2 > kl0


@pytest.mark.parametrize("j", [1, 2, 3])
def test_aux_posterior_init_logvar_is_zero(j: int) -> None:
    """At init ``q_Φ``'s log-variance is anchored at 0 (σ² = I).

    Guarantees ``KL[q_Φ || N(0, I)]`` reduces to ``0.5·E ‖μ‖²`` at the
    start of training — the logvar contribution is exactly zero.
    """
    aux = AuxPosterior(latent_dim=D, j=j)
    z_init = torch.randn(B, D, j)
    _, aux_logvar = aux(z_init)
    assert torch.allclose(aux_logvar, torch.zeros_like(aux_logvar), atol=1e-5)


def test_aux_posterior_rejects_bad_shape() -> None:
    """Inputs with the wrong rank or shape are rejected."""
    aux = AuxPosterior(latent_dim=D, j=2)
    with pytest.raises(ValueError):
        aux(torch.randn(B, D, 1))  # wrong j
    with pytest.raises(ValueError):
        aux(torch.randn(B, D + 1, 2))  # wrong d
    with pytest.raises(ValueError):
        aux(torch.randn(B, D))  # wrong rank
