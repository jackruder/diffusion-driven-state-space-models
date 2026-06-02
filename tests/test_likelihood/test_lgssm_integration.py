"""Integration test: prob-flow IWAE reproduces the exact LGSSM likelihood.

For a linear-Gaussian state-space model the marginal likelihood
``log p(x_{1:T})`` is available in closed form (Kalman filter), and the
exact posterior ``q(z_{1:T} | x_{1:T})`` is the RTS smoother.  When the
*exact* posterior is used as the IWAE proposal, every importance weight
collapses to the deterministic constant ``log p(x_{1:T})`` (model-v2.org
§ Reduction sanity check #1) — independent of the sampled trajectory.

This test assembles the marginal likelihood the way the exact-likelihood
evaluator does — analytic emission + initial-state densities, the
**probability-flow ODE** for each transition density, and
:func:`iwae_log_likelihood` for the trajectory reduction — and checks it
against the Kalman ground truth.  The per-transition densities go through
``solve_prob_flow_logdensity`` with the analytic diffused-conditional
score, so a bug in the ODE drift, the Liouville accumulator, the endpoint
density, or the IWAE assembly would shift the result off the Kalman value.
"""

from __future__ import annotations

import math

import torch

from ddssm.model.likelihood import iwae_log_likelihood, solve_prob_flow_logdensity

# Scalar LGSSM:  z_1 ~ N(m0, P0);  z_t = a z_{t-1} + N(0, q);  x_t = z_t + N(0, r).
A = 0.8
Q = 0.5
R = 0.3
M0 = 0.0
P0 = 1.0

BETA_MIN = 0.1
BETA_MAX = 50.0  # stiff: α(1) ≈ 4e-6 so the N(0, I) endpoint approx is sub-tolerance
TAU_MIN = 1e-3


def _normal_logpdf(x, mean, var):
    return -0.5 * ((x - mean).pow(2) / var + var.log() + math.log(2.0 * math.pi))


def _kalman_loglik(x):
    """Exact ``log p(x_{1:T})`` per sequence.  ``x``: (B, T).  Returns (B,)."""
    B, T = x.shape
    m = torch.full((B,), M0, dtype=x.dtype)
    P = torch.full((B,), P0, dtype=x.dtype)
    loglik = torch.zeros(B, dtype=x.dtype)
    m_filt, P_filt = [], []
    for t in range(T):
        if t > 0:
            m = A * m
            P = A * A * P + Q
        S = P + R
        nu = x[:, t] - m
        loglik = loglik - 0.5 * (nu.pow(2) / S + S.log() + math.log(2.0 * math.pi))
        Kg = P / S
        m = m + Kg * nu
        P = (1.0 - Kg) * P
        m_filt.append(m)
        P_filt.append(P)
    return loglik, torch.stack(m_filt, 1), torch.stack(P_filt, 1)


def _backward_sample(m_filt, P_filt, K, generator):
    """FFBS draw from the exact posterior + its analytic log-density.

    ``m_filt``/``P_filt``: (B, T).  Returns ``z`` (B, K, T) and
    ``log_q`` (B, K) = exact ``log q(z_{1:T} | x_{1:T})``.
    """
    B, T = m_filt.shape
    dtype = m_filt.dtype
    z = torch.zeros(B, K, T, dtype=dtype)
    log_q = torch.zeros(B, K, dtype=dtype)

    mT = m_filt[:, T - 1].unsqueeze(1).expand(B, K)
    PT = P_filt[:, T - 1].unsqueeze(1).expand(B, K)
    z[:, :, T - 1] = mT + PT.sqrt() * torch.randn(B, K, generator=generator, dtype=dtype)
    log_q = log_q + _normal_logpdf(z[:, :, T - 1], mT, PT)

    for t in range(T - 2, -1, -1):
        Pf = P_filt[:, t].unsqueeze(1).expand(B, K)
        mf = m_filt[:, t].unsqueeze(1).expand(B, K)
        P_pred_next = A * A * Pf + Q
        J = Pf * A / P_pred_next
        mean_t = mf + J * (z[:, :, t + 1] - A * mf)
        var_t = Pf - J.pow(2) * P_pred_next
        z[:, :, t] = mean_t + var_t.sqrt() * torch.randn(
            B, K, generator=generator, dtype=dtype
        )
        log_q = log_q + _normal_logpdf(z[:, :, t], mean_t, var_t)
    return z, log_q


def _transition_score(z_prev, dtype):
    """Analytic score of the diffused conditional ``N(a z_prev, q)``."""

    def score_fn(z, tau):
        if tau.dim() == 0:
            tau = tau.expand(z.shape[0])
        int_beta = BETA_MIN * tau + 0.5 * (BETA_MAX - BETA_MIN) * tau.pow(2)
        alpha = torch.exp(-0.5 * int_beta).unsqueeze(-1)
        alpha2 = alpha.pow(2)
        var = alpha2 * Q + (1.0 - alpha2)
        mean = alpha * A * z_prev
        return -(z - mean) / var

    return score_fn


def test_probflow_iwae_matches_kalman_loglik() -> None:
    """IWAE with the exact RTS proposal reproduces the Kalman log-likelihood."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    dtype = torch.float64

    B, T, K = 2, 4, 4
    x = torch.randn(B, T, dtype=dtype)

    true_loglik, m_filt, P_filt = _kalman_loglik(x)
    z, log_q = _backward_sample(m_filt, P_filt, K, gen)  # z: (B,K,T), log_q: (B,K)

    # log p(z_1) — analytic initial-state prior.
    log_p_init = _normal_logpdf(
        z[:, :, 0], torch.tensor(M0, dtype=dtype), torch.tensor(P0, dtype=dtype)
    )  # (B, K)

    # log p(x_t | z_t) — analytic emission, summed over t.
    log_p_dec = _normal_logpdf(
        x.unsqueeze(1), z, torch.tensor(R, dtype=dtype)
    ).sum(dim=-1)  # (B, K)

    # log p(z_t | z_{t-1}) — via the probability-flow ODE, summed over t = 2..T.
    log_p_trans = torch.zeros(B, K, dtype=dtype)
    for t in range(1, T):
        z_prev = z[:, :, t - 1].reshape(B * K, 1)
        z_curr = z[:, :, t].reshape(B * K, 1)
        lp = solve_prob_flow_logdensity(
            score_fn=_transition_score(z_prev, dtype),
            z0=z_curr,
            beta_min=BETA_MIN,
            beta_max=BETA_MAX,
            tau_min=TAU_MIN,
            rtol=1e-9,
            atol=1e-9,
            divergence_mode="exact",
        )
        log_p_trans = log_p_trans + lp.reshape(B, K)

    log_p_xz = log_p_init + log_p_trans + log_p_dec
    iwae = iwae_log_likelihood(log_p_xz, log_q, dim=-1)  # (B,)

    # Error budget: ODE-solver tolerance + endpoint (α(1)≈4e-6) + tau_min,
    # accumulated over T-1 transitions.
    assert torch.allclose(iwae, true_loglik, atol=2e-3, rtol=0.0)
