"""Unit tests for :mod:`ddssm.likelihood.vhp`."""

from __future__ import annotations

import torch

from ddssm.model.likelihood.vhp import vhp_log_prob_init


def test_vhp_log_prob_init_collapses_when_q_is_true_posterior() -> None:
    """Importance weights coincide when ``q_Φ`` equals the true posterior.

    Cycle-5 tracer.  By construction
        w_j = log p_ψ^ode(z_1 | z_0^j) + log N(z_0^j; 0, I) − log q_Φ(z_0^j | z_1),
    and when ``q_Φ(z_0 | z_1) = p_ψ(z_0 | z_1) = p_ψ(z_0, z_1) / p_ψ(z_1)``
    each ``w_j`` reduces to the deterministic constant ``log p_ψ(z_1)``
    regardless of which ``z_0^j`` was drawn.  Then ``logmeanexp`` of the
    weights equals ``log p_ψ(z_1)`` exactly — the IS estimator
    collapses to the marginal, mirroring the IWAE sanity check.
    """
    torch.manual_seed(0)
    B, J = 3, 4

    log_p_z1 = torch.tensor([0.5, -1.2, 2.0])
    log_p_z1_given_z0 = torch.randn(B, J)
    log_prior_z0 = torch.randn(B, J)
    log_q_z0_given_z1 = log_p_z1_given_z0 + log_prior_z0 - log_p_z1.unsqueeze(-1)

    vhp = vhp_log_prob_init(
        log_p_z1_given_z0=log_p_z1_given_z0,
        log_q_z0_given_z1=log_q_z0_given_z1,
        log_prior_z0=log_prior_z0,
        dim=-1,
    )

    assert vhp.shape == (B,)
    assert torch.allclose(vhp, log_p_z1, atol=1e-6)
