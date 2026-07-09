"""Baseline μ_p(z_{t-1}) centering head.

Per ``model-v2.org`` § Baseline-form variants, the centering function
``μ_p(z_{t-1})`` admits two parameter-free families: Zero and
Persistence.  ("Persistence" was previously called "Identity"; renamed
because at j>1 it's the persistence/last-value baseline, not the
identity-on-the-window — see docs/adr/0010-persistence-baseline-rename.md.)

The prior variance is fixed at ``σ_p² = 1`` (``log σ_p² = 0``): the
baselines carry **no parameters**.  ``mean_and_logvar`` returns a zero
log-variance alongside the mean so the module stays a drop-in for the
:class:`ddssm.nn.gaussians.GaussianHead` contract.

The ``BaseBaseline`` interface exposes two access patterns:

* ``mean(z_hist)`` — μ_p alone.  Used by the diffusion transition for the
  centering shift ``ẑ_t = z̃_t − μ_p(z_{t-1})``.
* ``mean_and_logvar(z_hist)`` — μ_p plus the (identically-zero) log σ_p².

Both concrete forms support general j ≥ 1.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseBaseline(nn.Module, metaclass=abc.ABCMeta):
    """Abstract μ_p head with fixed unit prior variance.

    Subclasses must implement :meth:`mean` returning ``(B, d)`` from
    ``(B, d, j)``.  :meth:`mean_and_logvar` pairs that mean with a
    zero log-variance (σ_p² = 1).

    The default :meth:`forward` delegates to :meth:`mean_and_logvar` so
    the module is a drop-in replacement for the
    :class:`ddssm.nn.gaussians.GaussianHead` contract.
    """

    latent_dim: int
    j: int

    @abc.abstractmethod
    def mean(self, z_hist: torch.Tensor) -> torch.Tensor:
        """Return μ_p(z_hist) with shape ``(B, d)``."""

    def mean_and_logvar(
        self, z_hist: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(μ_p, log σ_p²)`` with ``log σ_p² ≡ 0`` (σ_p² = 1)."""
        mu = self.mean(z_hist)
        return mu, torch.zeros_like(mu)

    def forward(self, z_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop-in replacement for ``GaussianHead.forward``."""
        return self.mean_and_logvar(z_hist)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_z_hist(z_hist: torch.Tensor, latent_dim: int, j: int) -> None:
    if z_hist.dim() != 3:
        raise ValueError(f"z_hist must be (B, d, j); got shape {tuple(z_hist.shape)}")
    if z_hist.shape[1] != latent_dim or z_hist.shape[2] != j:
        raise ValueError(
            "z_hist shape mismatch: expected (B, "
            f"{latent_dim}, {j}); got (B, {z_hist.shape[1]}, {z_hist.shape[2]})"
        )


# ---------------------------------------------------------------------------
# Concrete forms
# ---------------------------------------------------------------------------


class ZeroBaseline(BaseBaseline):
    """``μ_p ≡ 0`` with fixed unit prior variance. Parameter-free."""

    def __init__(self, latent_dim: int, j: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.j = int(j)

    def mean(self, z_hist: torch.Tensor) -> torch.Tensor:
        _validate_z_hist(z_hist, self.latent_dim, self.j)
        return torch.zeros(
            z_hist.shape[0], self.latent_dim, device=z_hist.device, dtype=z_hist.dtype
        )


class PersistenceBaseline(BaseBaseline):
    """``μ_p(z_{t-1}) = z_hist[..., -1]`` — the persistence (last-value-carried-forward) baseline.

    At j=1 this is equivalently ``μ_p(z_{t-1}) = z_{t-1}`` and was originally
    called "identity". At j>1, however, taking ``z_hist[..., -1]`` selects the
    most-recent slot of the window — it is NOT the identity map on the window
    (the input is `(B, d, j)` and the output is `(B, d)`); it is the standard
    *persistence* / no-change forecast. See
    docs/adr/0010-persistence-baseline-rename.md for the rename rationale.

    Fixed unit prior variance. Parameter-free.
    """

    def __init__(self, latent_dim: int, j: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.j = int(j)

    def mean(self, z_hist: torch.Tensor) -> torch.Tensor:
        _validate_z_hist(z_hist, self.latent_dim, self.j)
        return z_hist[..., -1]
