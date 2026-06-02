"""VHP initial-state estimator (Layer 4 of the exact-likelihood evaluator).

See ``model-v2.org`` § "VHP initial state".  The initial-state prior::

    p_ψ(z_1) = E_{z_0 ∼ N(0, I)}[ p_ψ^ode(z_1 | z_0) ]

is estimated by importance sampling under the trained auxiliary
posterior ``q_Φ(z_0 | z_1)``::

    log p_ψ(z_1)
        ≈ logmeanexp_j[
              log p_ψ^ode(z_1 | z_0^j)
            + log N(z_0^j; 0, I)
            − log q_Φ(z_0^j | z_1)
          ],
    z_0^j ∼ q_Φ(· | z_1).

When the per-transition ``p_ψ^ode`` is evaluated exactly this is an
IWAE lower bound on ``log p_ψ(z_1)``; under Hutchinson trace the
strict-bound property is lost (cf. model-v2.org § Validity of the IWAE
bound under Hutchinson).
"""

from __future__ import annotations

import torch

from ddssm.model.likelihood.iwae import logmeanexp


def vhp_log_prob_init(
    log_p_z1_given_z0: torch.Tensor,
    log_q_z0_given_z1: torch.Tensor,
    log_prior_z0: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Importance-sampled estimate of ``log p_ψ(z_1)``.

    Args:
        log_p_z1_given_z0: ``(..., J, ...)`` per-draw transition
            log-density ``log p_ψ^ode(z_1 | z_0^j)``.
        log_q_z0_given_z1: ``(..., J, ...)`` per-draw proposal
            log-density ``log q_Φ(z_0^j | z_1)``.
        log_prior_z0: ``(..., J, ...)`` per-draw standard-Gaussian
            log-density ``log N(z_0^j; 0, I)``.
        dim: axis along which the ``J`` IS draws lie.

    Returns:
        ``logmeanexp`` of the IS log-weights along ``dim``.
    """
    log_w = log_p_z1_given_z0 + log_prior_z0 - log_q_z0_given_z1
    return logmeanexp(log_w, dim=dim)
