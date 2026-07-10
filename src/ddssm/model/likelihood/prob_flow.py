"""Probability-flow ODE log-density evaluation.

Implements Layer 1 of the exact-likelihood evaluator from
``model-v2.org`` § "Exact likelihood evaluation".  The forward VP-SDE
is converted to a deterministic probability-flow ODE; the per-transition
log-density follows from the instantaneous change-of-variables
(Liouville) identity

    log p_ψ^ode(z) = log π(z(1)) + ∫_0^1 (∇ · F_τ)(z(τ)) dτ,
    F_τ(z) = -½ β(τ) z - ½ β(τ) s_ψ(z, τ),  π = N(0, I).

The integrand is the trace of the drift Jacobian.  For VP-SDE the
drift divergence is constant (``-½ β(τ) · D``); only the score
divergence ``∇ · s_ψ`` requires estimation. ``exact`` mode computes
it as the sum of ``D`` reverse-mode passes; ``hutchinson`` mode
estimates it with a single reverse-mode pass against a fixed probe
vector.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torchdiffeq import odeint


def exact_score_divergence(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``s(z, τ)`` and ``∇_z · s(z, τ)`` exactly.

    ``D`` reverse-mode passes over the latent dimension; cost is
    ``O(D)`` per ODE evaluation but variance is zero.  Returns the
    score and its divergence together so the caller does not have to
    re-evaluate the network.

    Args:
        score_fn: callable ``(z, τ) → (B, d)``.
        z: ``(B, d)`` input latent.
        tau: scalar ``τ`` tensor.

    Returns:
        ``(s, div_s)`` with shapes ``(B, d)`` and ``(B,)``.
    """
    B, d = z.shape
    z_g = z.detach().requires_grad_(True)
    with torch.enable_grad():
        s = score_fn(z_g, tau)
        div = torch.zeros(B, device=z_g.device, dtype=z_g.dtype)
        for i in range(d):
            grad_i = torch.autograd.grad(
                outputs=s[:, i].sum(),
                inputs=z_g,
                retain_graph=(i < d - 1),
                create_graph=False,
            )[0][:, i]
            div = div + grad_i
    return s.detach(), div.detach()


def hutchinson_score_divergence(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    tau: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``s(z, τ)`` and the Hutchinson trace estimate ``vᵀ J v``.

    One reverse-mode pass via ``grad_outputs=v`` returns ``vᵀ J`` (the
    same shape as ``z``); the elementwise product with ``v`` and a sum
    over the latent dim yields ``vᵀ J v`` per row.  ``E_v[vᵀ J v] =
    tr(J)`` for any ``v`` with ``E[vvᵀ] = I``; we use Rademacher
    (``v_i ∈ {±1}``) which has minimum variance for diagonal-heavy ``J``.

    Args:
        score_fn: ``(z, τ) → (B, d)``.
        z: ``(B, d)`` input.
        tau: scalar ``τ``.
        v: ``(B, d)`` probe vector (caller-supplied; held fixed across
            the ODE solve per FFJORD / Song & Ermon).

    Returns:
        ``(s, vᵀ J v)`` with shapes ``(B, d)`` and ``(B,)``.
    """
    z_g = z.detach().requires_grad_(True)
    with torch.enable_grad():
        s = score_fn(z_g, tau)
        vJ = torch.autograd.grad(
            outputs=s,
            inputs=z_g,
            grad_outputs=v,
            create_graph=False,
            retain_graph=False,
        )[0]
    return s.detach(), (vJ * v).sum(dim=-1).detach()


def solve_prob_flow_logdensity(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    *,
    beta_min: float,
    beta_max: float,
    tau_min: float = 1e-3,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    method: str = "dopri5",
    divergence_mode: str = "exact",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Native-coord prob-flow ODE log-density at ``z0``.

    Integrates the augmented dynamics

        dz/dτ = -½ β(τ) (z + s(z, τ))
        dA/dτ = -½ β(τ) · D - ½ β(τ) · ∇·s(z(τ), τ)

    from ``τ = tau_min`` to ``τ = 1``, then returns

        log p_ψ^ode(z0) ≈ log N(z(1); 0, I) + A(1).

    The ``tau_min`` cutoff avoids the ``σ̃(τ→0)`` singularity in the
    VP-SDE score; the residual bias scales with ``β_min · tau_min``.

    Args:
        score_fn: ``(z, τ) → (B, d)`` native-coord score.
        z0: ``(B, d)`` evaluation point.
        beta_min, beta_max: VP-SDE schedule endpoints.
        tau_min: integration lower bound (≈ 0).
        rtol, atol: adaptive solver tolerances.
        method: torchdiffeq solver name (``dopri5``, ``dopri8``, ...).
        divergence_mode: ``"exact"`` (``D`` reverse-mode passes per ODE
            evaluation, zero variance) or ``"hutchinson"`` (one
            reverse-mode pass per evaluation against a Rademacher probe
            vector ``v`` held fixed over the ODE solve, unbiased on the
            linear log-density scale).
        generator: optional ``torch.Generator`` for reproducible
            Hutchinson draws.

    Returns:
        ``(B,)`` per-row log-density.
    """
    if divergence_mode not in {"exact", "hutchinson"}:
        raise ValueError(
            f"divergence_mode must be 'exact' or 'hutchinson'; got {divergence_mode!r}"
        )

    B, d = z0.shape
    device = z0.device
    dtype = z0.dtype

    drift_div_const = -0.5 * float(d)

    if divergence_mode == "hutchinson":
        v = (
            torch
            .randint(0, 2, (B, d), device=device, generator=generator)
            .to(dtype=dtype)
            .mul_(2.0)
            .sub_(1.0)
        )
    else:
        v = None

    def dynamics(tau: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        z, _A = state
        beta = beta_min + (beta_max - beta_min) * tau
        if v is None:
            s, div_s = exact_score_divergence(score_fn, z, tau)
        else:
            s, div_s = hutchinson_score_divergence(score_fn, z, tau, v)
        F = -0.5 * beta * (z + s)
        div_F = beta * drift_div_const - 0.5 * beta * div_s
        return (F, div_F)

    A0 = torch.zeros(B, device=device, dtype=dtype)
    tau_grid = torch.tensor([tau_min, 1.0], device=device, dtype=dtype)

    # Cap dt at the integration window so an aggressive step-size controller
    # (dopri5 grows dt ×ifactor when error_ratio=0) cannot overshoot τ=1. An
    # overshoot drives α(τ) = exp(-½·∫β) to underflow, z/α to overflow, and
    # the score net to NaN — after which dt_next = NaN → clamped to min_step
    # (default 0) → the next step trips the ``t0 + dt > t0`` assertion.
    tau_window = 1.0 - float(tau_min)
    z_traj, A_traj = odeint(
        dynamics,
        (z0, A0),
        tau_grid,
        method=method,
        rtol=rtol,
        atol=atol,
        options={"max_step": tau_window},
    )
    z_end = z_traj[-1]
    A_end = A_traj[-1]

    log_prior = -0.5 * z_end.pow(2).sum(dim=-1) - 0.5 * float(d) * math.log(
        2.0 * math.pi
    )
    return log_prior + A_end
