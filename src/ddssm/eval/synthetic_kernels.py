"""Closed-form ground-truth transition kernels for synthetic data modes.

Used by the ``gt_latent_jsd`` metric in :mod:`ddssm.eval.metrics` to
sample from the *known* data-generating transition kernel and compare
against the model's learned ``p_ψ(z_t | z_{t-1})``.

Each kernel is a callable
``kernel(z_hist: np.ndarray (B, d, j), S: int, ...) -> np.ndarray (B, S, d)``
returning ``S`` i.i.d. samples of ``z_t`` from the closed-form
ground-truth transition, conditioned on the GT history.

Registered kernels:

- ``lgssm`` — ``z_t = 0.9·z_{t-1} + 0.1·N(0, I)``.
- ``nonlinear-bimodal-lift`` (d=1) —
  ``z_t = tanh(z_{t-1}) + δ·s_t + σ_z·N(0, 1)`` with ``s_t ∈ {-1, +1}``.
- ``nonlinear-bimodal-lift-mv`` (d=NLBL_MV_LATENT_D) —
  ``z_t = tanh(A·z_{t-1}) + δ·s_t + σ_z·N(0, I)`` with
  ``s_t ∈ {-1, +1}^d`` (per-dim independent) and ``A`` reconstructed
  from ``NLBL_MV_A_SEED`` so it matches the data generator's matrix.

The δ / σ_z constants live in :mod:`ddssm.data.synthetic` (single
source of truth for the data generator) and are imported here.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from ddssm.data.synthetic import (
    NLBL_DELTA,
    NLBL_SIGMA_Z,
    NLBL_MV_A_SEED,
    NLBL_MV_LATENT_D,
)


KernelFn = Callable[..., np.ndarray]
KERNEL_REGISTRY: Dict[str, KernelFn] = {}


def register_kernel(name: str) -> Callable[[KernelFn], KernelFn]:
    """Decorator-style registration for new synthetic-mode kernels."""

    def _wrap(fn: KernelFn) -> KernelFn:
        if name in KERNEL_REGISTRY:
            raise ValueError(f"Kernel {name!r} already registered")
        KERNEL_REGISTRY[name] = fn
        return fn

    return _wrap


@register_kernel("lgssm")
def lgssm_kernel(
    z_hist: np.ndarray,  # (B, d, j)
    S: int,
    *,
    batch_idx: int = 0,
    t: int = 0,
    a: float = 0.9,
    sigma: float = 0.1,
) -> np.ndarray:
    """LGSSM transition: ``z_t = a·z_{t-1} + sigma·N(0, I)``.

    Matches ``SyntheticDataset._generate_data`` for the ``lgssm`` mode.
    Independent across the B batch + S samples.  Uses the last slot of
    ``z_hist`` (i.e., ``z_{t-1}``); the j>1 case still conditions only
    on the most recent latent because LGSSM is Markov-1.
    """
    rng = np.random.default_rng(seed=10_000 * (batch_idx + 1) + (t + 1))
    B, d, j = z_hist.shape
    z_prev = z_hist[..., -1]  # (B, d)
    eps = rng.standard_normal(size=(B, S, d))  # (B, S, d)
    z_next = a * z_prev[:, None, :] + sigma * eps
    return z_next


@register_kernel("nonlinear-bimodal-lift")
def nonlinear_bimodal_lift_kernel(
    z_hist: np.ndarray,  # (B, d=1, j)
    S: int,
    *,
    batch_idx: int = 0,
    t: int = 0,
) -> np.ndarray:
    """``z_t = tanh(z_{t-1}) + δ·s_t + σ_z·N(0, 1)``, ``s_t ∈ {-1, +1}``.

    Markov-1 (uses only the last slot of ``z_hist``). Bimodal because
    ``s_t`` is a fresh per-(batch, sample) Rademacher draw; the score
    net's learned transition has to capture that the conditional
    distribution of ``z_t | z_{t-1}`` is bimodal with modes at
    ``tanh(z_{t-1}) ± δ``.
    """
    rng = np.random.default_rng(seed=10_000 * (batch_idx + 1) + (t + 1))
    B, d, j = z_hist.shape
    z_prev = z_hist[..., -1]  # (B, d)
    signs = rng.choice([-1.0, 1.0], size=(B, S, d)).astype(z_hist.dtype)
    eps = rng.standard_normal(size=(B, S, d)).astype(z_hist.dtype)
    z_next = np.tanh(z_prev[:, None, :]) + NLBL_DELTA * signs + NLBL_SIGMA_Z * eps
    return z_next


def _mv_mixing_matrix() -> np.ndarray:
    """Reconstruct the multivariate mixing matrix ``A`` deterministically.

    Matches ``SyntheticDataset._generate_data`` for
    ``nonlinear-bimodal-lift-mv``: a (d, d) draw under a fixed
    ``torch.Generator`` seed (``NLBL_MV_A_SEED``). Numpy is sufficient
    here because we only need the matrix values — we run this in numpy
    to keep the kernel framework-pure.
    """
    import torch  # local import to keep top-level kernel imports light

    gen = torch.Generator().manual_seed(NLBL_MV_A_SEED)
    A = torch.randn(NLBL_MV_LATENT_D, NLBL_MV_LATENT_D, generator=gen)
    return A.numpy()


@register_kernel("nonlinear-bimodal-lift-mv")
def nonlinear_bimodal_lift_mv_kernel(
    z_hist: np.ndarray,  # (B, d, j)
    S: int,
    *,
    batch_idx: int = 0,
    t: int = 0,
) -> np.ndarray:
    """``z_t = tanh(A·z_{t-1}) + δ·s_t + σ_z·N(0, I)``, per-dim signs.

    ``A`` is the same matrix the data generator used (reconstructed via
    :func:`_mv_mixing_matrix` from ``NLBL_MV_A_SEED``). The per-dim
    Rademacher ``s_t ∈ {-1, +1}^d`` yields ``2^d`` attractors.
    """
    rng = np.random.default_rng(seed=10_000 * (batch_idx + 1) + (t + 1))
    B, d, j = z_hist.shape
    A = _mv_mixing_matrix()  # (d, d)
    z_prev = z_hist[..., -1]  # (B, d)
    Az = z_prev @ A.T  # (B, d)
    signs = rng.choice([-1.0, 1.0], size=(B, S, d)).astype(z_hist.dtype)
    eps = rng.standard_normal(size=(B, S, d)).astype(z_hist.dtype)
    z_next = np.tanh(Az[:, None, :]) + NLBL_DELTA * signs + NLBL_SIGMA_Z * eps
    return z_next
