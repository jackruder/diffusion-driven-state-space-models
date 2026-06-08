"""Baseline μ_p(z_{t-1}) head with state-conditional σ_p sibling.

Per ``model-v2.org`` § Baseline-form variants, the centering function
``μ_p(z_{t-1})`` admits four parametric families (Zero, Identity,
Linear, MLP).  Per § State-conditional prior variance, the stage-1
Gaussian transition prior is
``N(μ_p(z_{t-1}), diag(σ_p²(z_{t-1})))`` — so a sibling
state-conditional ``σ_p`` head exists alongside ``μ_p`` and (for the
parametric forms) shares the backbone.

The ``BaseBaseline`` interface exposes two access patterns:

* ``mean(z_hist)`` — μ_p alone.  Used by the stage-2 diffusion transition
  for the centering shift ``ẑ_t = z̃_t − μ_p(z_{t-1})``; σ_p plays
  no role in stage 2.
* ``mean_and_logvar(z_hist)`` — both heads.  Used by the stage-1
  Gaussian transition for the closed-form KL and by the
  log-variance regularizer ``R_σp``.

The four concrete forms all support general j ≥ 1.  The doc writes
the linear form as ``μ_p(z_{t-1}) = A z_{t-1} + b`` with ``A ∈
R^{D×D}``; we generalise to ``A ∈ R^{D×(j·D)}`` (linear over the
flattened history), reducing to the doc's expression at j = 1.
"""

from __future__ import annotations

import abc
import copy
from typing import Tuple

import torch
import torch.nn as nn

from ddssm.nn.gaussians import LogvarHead


class BaseBaseline(nn.Module, metaclass=abc.ABCMeta):
    """Abstract μ_p / σ_p head.

    Subclasses must:
      - implement :meth:`mean` returning ``(B, d)`` from ``(B, d, j)``.
      - implement :meth:`mean_and_logvar` returning ``((B, d), (B, d))``.

    The default :meth:`forward` delegates to :meth:`mean_and_logvar` so
    the module is a drop-in replacement for the existing
    :class:`ddssm.nn.gaussians.GaussianHead` contract used by the legacy
    :class:`ddssm.model.transitions.transitions.GaussianTransition`.
    """

    latent_dim: int
    j: int

    @abc.abstractmethod
    def mean(self, z_hist: torch.Tensor) -> torch.Tensor:
        """Return μ_p(z_hist) with shape ``(B, d)``."""

    @abc.abstractmethod
    def mean_and_logvar(
        self, z_hist: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(μ_p, log σ_p²)``, each ``(B, d)``."""

    def forward(
        self, z_hist: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drop-in replacement for ``GaussianHead.forward``."""
        return self.mean_and_logvar(z_hist)

    def snapshot(self) -> "BaseBaseline":
        """Return a deep-copied, frozen eval-mode copy.

        Used as the anchor target μ_p^(0) for the Learnable
        baseline-mode regularizer R_μp.  Mutating parameters of the
        snapshot does not affect the live baseline (and vice versa).
        """
        clone = copy.deepcopy(self)
        clone.eval()
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_z_hist(z_hist: torch.Tensor, latent_dim: int, j: int) -> None:
    if z_hist.dim() != 3:
        raise ValueError(
            f"z_hist must be (B, d, j); got shape {tuple(z_hist.shape)}"
        )
    if z_hist.shape[1] != latent_dim or z_hist.shape[2] != j:
        raise ValueError(
            "z_hist shape mismatch: expected (B, "
            f"{latent_dim}, {j}); got (B, {z_hist.shape[1]}, {z_hist.shape[2]})"
        )


class _StateConditionalSigmaHead(nn.Module):
    """Small MLP body + :class:`LogvarHead` producing per-dim ``log σ_p²``.

    Used by the parameter-free baseline forms (Zero, Identity) so that
    σ_p remains state-conditional per ``model-v2.org`` § State-conditional
    prior variance.  Output starts at ``init_logvar`` (default 0 → σ_p² = I)
    regardless of the input, matching ``GaussianHead``'s convention.
    """

    def __init__(
        self,
        latent_dim: int,
        j: int,
        hidden_dim: int,
        n_layers: int,
        init_logvar: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = latent_dim * j
        body: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(0, n_layers - 1)):
            body.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        self.body = nn.Sequential(*body)
        self.logvar_head = LogvarHead(
            in_features=hidden_dim,
            out_features=latent_dim,
            init_logvar=init_logvar,
        )

    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        B = z_hist.shape[0]
        h = self.body(z_hist.reshape(B, -1))
        return self.logvar_head(h)


# ---------------------------------------------------------------------------
# Concrete forms
# ---------------------------------------------------------------------------


class ZeroBaseline(BaseBaseline):
    """``μ_p ≡ 0``; σ_p from a small state-conditional MLP head."""

    def __init__(
        self,
        latent_dim: int,
        j: int,
        hidden_dim: int = 32,
        n_layers: int = 2,
        init_logvar: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.j = int(j)
        self.sigma_head = _StateConditionalSigmaHead(
            self.latent_dim, self.j, hidden_dim, n_layers, init_logvar=init_logvar
        )

    def mean(self, z_hist: torch.Tensor) -> torch.Tensor:
        _validate_z_hist(z_hist, self.latent_dim, self.j)
        return torch.zeros(
            z_hist.shape[0], self.latent_dim, device=z_hist.device, dtype=z_hist.dtype
        )

    def mean_and_logvar(
        self, z_hist: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_z_hist(z_hist, self.latent_dim, self.j)
        mu = torch.zeros(
            z_hist.shape[0], self.latent_dim, device=z_hist.device, dtype=z_hist.dtype
        )
        logvar = self.sigma_head(z_hist)
        return mu, logvar


class IdentityBaseline(BaseBaseline):
    """``μ_p(z_{t-1}) = z_{t-1}`` (GenCast-style random walk).

    For ``j > 1`` we take ``z_hist[..., -1]``, i.e. the most recent
    history slot.  σ_p comes from a small state-conditional MLP head.
    """

    def __init__(
        self,
        latent_dim: int,
        j: int,
        hidden_dim: int = 32,
        n_layers: int = 2,
        init_logvar: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.j = int(j)
        self.sigma_head = _StateConditionalSigmaHead(
            self.latent_dim, self.j, hidden_dim, n_layers, init_logvar=init_logvar
        )

    def mean(self, z_hist: torch.Tensor) -> torch.Tensor:
        _validate_z_hist(z_hist, self.latent_dim, self.j)
        return z_hist[..., -1]

    def mean_and_logvar(
        self, z_hist: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_z_hist(z_hist, self.latent_dim, self.j)
        mu = z_hist[..., -1]
        logvar = self.sigma_head(z_hist)
        return mu, logvar


class LinearBaseline(BaseBaseline):
    """``μ_p(z_{t-1}) = A · vec(z_hist) + b``; σ_p from a sibling linear head.

    At j = 1 reduces to the doc's ``A z_{t-1} + b`` with ``A ∈ R^{D×D}``.
    μ_p and σ_p share the same flat input vector but use separate
    linear projections (DKF "two-headed" convention).
    """

    def __init__(
        self, latent_dim: int, j: int, init_logvar: float = 0.0
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.j = int(j)
        in_dim = self.latent_dim * self.j
        self.mu_head = nn.Linear(in_dim, self.latent_dim)
        # Match GaussianHead.mu_head convention: xavier-uniform weight (small)
        # + zero bias. At z_hist=0 ⇒ μ_p=0; under typical-scale z_hist μ_p is
        # a small projection that the model can grow. Going further and
        # zeroing the weight too lands the score-net likelihood in a config
        # where dopri5 is pathologically slow on the integration tests.
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.5)
        nn.init.zeros_(self.mu_head.bias)
        self.logvar_head = LogvarHead(
            in_features=in_dim,
            out_features=self.latent_dim,
            init_logvar=init_logvar,
        )

    def _flatten(self, z_hist: torch.Tensor) -> torch.Tensor:
        return z_hist.reshape(z_hist.shape[0], -1)

    def mean(self, z_hist: torch.Tensor) -> torch.Tensor:
        _validate_z_hist(z_hist, self.latent_dim, self.j)
        return self.mu_head(self._flatten(z_hist))

    def mean_and_logvar(
        self, z_hist: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_z_hist(z_hist, self.latent_dim, self.j)
        flat = self._flatten(z_hist)
        return self.mu_head(flat), self.logvar_head(flat)


class MLPBaseline(BaseBaseline):
    """DKF-style nonlinear baseline: shared MLP backbone, two output heads.

    Per ``model-v2.org`` § State-conditional prior variance, ``μ_p``
    and ``log σ_p²`` are produced from a *shared* backbone with two
    output linear heads (standard Gaussian-head convention).  This is
    the variant the smoke test uses and the one the doc is written
    around.
    """

    def __init__(
        self,
        latent_dim: int,
        j: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        init_logvar: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.j = int(j)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)

        in_dim = self.latent_dim * self.j
        body: list[nn.Module] = [nn.Linear(in_dim, self.hidden_dim), nn.SiLU()]
        for _ in range(max(0, self.n_layers - 1)):
            body.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU()])
        self.backbone = nn.Sequential(*body)

        self.mu_head = nn.Linear(self.hidden_dim, self.latent_dim)
        # Same convention as GaussianHead.mu_head — see LinearBaseline above.
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.5)
        nn.init.zeros_(self.mu_head.bias)
        self.logvar_head = LogvarHead(
            in_features=self.hidden_dim,
            out_features=self.latent_dim,
            init_logvar=init_logvar,
        )

    def _hidden(self, z_hist: torch.Tensor) -> torch.Tensor:
        return self.backbone(z_hist.reshape(z_hist.shape[0], -1))

    def mean(self, z_hist: torch.Tensor) -> torch.Tensor:
        _validate_z_hist(z_hist, self.latent_dim, self.j)
        return self.mu_head(self._hidden(z_hist))

    def mean_and_logvar(
        self, z_hist: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_z_hist(z_hist, self.latent_dim, self.j)
        h = self._hidden(z_hist)
        return self.mu_head(h), self.logvar_head(h)
