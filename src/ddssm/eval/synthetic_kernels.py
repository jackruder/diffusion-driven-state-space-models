"""Closed-form ground-truth transition kernels for synthetic data modes.

Used by the ``gt_latent_jsd`` metric in :mod:`ddssm.eval.metrics` to
sample from the *known* data-generating transition kernel and compare
against the model's learned ``p_ψ(z_t | z_{t-1})``.

Each kernel is a callable
``kernel(z_hist: np.ndarray (B, d, j), S: int, ...) -> np.ndarray (B, S, d)``
returning ``S`` i.i.d. samples of ``z_t`` from the closed-form
ground-truth transition, conditioned on the GT history.

The current registry covers ``lgssm`` only — the canonical
linear-Gaussian case where the kernel is exactly ``z_t = 0.9·z_{t-1} +
0.1·N(0, I)`` (matching ``SyntheticDataset._generate_data`` for the
``lgssm`` mode in ``src/ddssm/data/synthetic.py:59-67``).  Other modes
register over time as the metric's coverage expands.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


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
