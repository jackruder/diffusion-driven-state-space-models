"""IWAE assembly utilities (Layer 3 of the exact-likelihood evaluator).

See ``model-v2.org`` § "IWAE over trajectories".  Given ``K`` trajectory
samples ``z_{1:T}^{(k)} ∼ q_φ(· | x_{1:T})`` and the per-sample log
ratio ``w_k = log p_ψ(z, x) − log q_φ(z | x)``, the IWAE estimator is

    \\widehat{log p_ψ}^{(K)}(x) = logmeanexp_k(w_k).

Under exact-trace divergence the resulting estimator is a strict lower
bound on ``log p_ψ(x)`` for any ``K``, tight as ``K → ∞`` (Burda et
al. 2016).  Hutchinson breaks the strict-bound property — see
``model-v2.org § Validity of the IWAE bound under Hutchinson``.

The helpers here are deliberately low-level: callers assemble the
per-trajectory ``log p_ψ(z, x)`` and ``log q_φ(z | x)`` totals (summing
across the time axis, the initial-state contribution, the decoder
log-likelihood, and the per-transition ``log p_ψ^ode``) and call
:func:`iwae_log_likelihood` to reduce over the K dimension.
"""

from __future__ import annotations

import math

import torch


def logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically-stable ``log(mean(exp(x), dim=dim))``."""
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


def iwae_log_likelihood(
    log_p_xz: torch.Tensor,
    log_q_z: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Assemble the IWAE log-likelihood estimator over ``K`` samples.

    Args:
        log_p_xz: per-sample joint ``log p_ψ(z^(k), x)`` with a ``K``
            axis at ``dim``.
        log_q_z: per-sample proposal ``log q_φ(z^(k) | x)`` with the
            same shape.
        dim: axis along which to reduce (the K axis).

    Returns:
        ``logmeanexp`` of the importance log-weights along ``dim``.
    """
    return logmeanexp(log_p_xz - log_q_z, dim=dim)
