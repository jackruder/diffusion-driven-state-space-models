"""Unit tests for :mod:`ddssm.likelihood.iwae`."""

from __future__ import annotations

import torch

from ddssm.model.likelihood.iwae import logmeanexp, iwae_log_likelihood


def test_iwae_collapses_to_log_marginal_when_q_matches_posterior() -> None:
    """All IWAE weights coincide when q_φ equals the true posterior.

    Cycle-4 tracer (model-v2.org § Reduction sanity check #1).  When
    ``log q_φ(z | x) = log p_ψ(z | x)``, each importance weight
    ``w_k = log p_ψ(z, x) − log q_φ(z | x) = log p_ψ(x)`` is the
    deterministic constant ``log p_ψ(x)``, so ``logmeanexp_k(w_k) =
    log p_ψ(x)`` exactly — independent of K and trajectory draws.
    """
    torch.manual_seed(0)
    B, K = 3, 4

    log_p_x = torch.tensor([1.0, 2.0, 3.0])
    log_p_z_given_x = torch.randn(B, K)

    log_p_xz = log_p_x.unsqueeze(-1) + log_p_z_given_x
    log_q_z = log_p_z_given_x

    iwae = iwae_log_likelihood(log_p_xz, log_q_z, dim=-1)

    assert iwae.shape == (B,)
    assert torch.allclose(iwae, log_p_x, atol=1e-6)


def test_logmeanexp_matches_naive_form() -> None:
    """``logmeanexp(x) = log(mean(exp(x)))`` numerically-stably."""
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    actual = logmeanexp(x, dim=-1)
    expected = x.exp().mean(dim=-1).log()
    assert torch.allclose(actual, expected, atol=1e-6)
