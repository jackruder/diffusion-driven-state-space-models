"""Loss-object abstraction (ADR-0004).

`DDSSM_base.forward()` returns a `LossComponents` bag of unweighted
per-term tensors; a `Loss` object weights and sums them into the scalar
that the trainer backprops. The loss object holds its own λ schedule
shape (pure function `step → float`); the trainer drives the step.

See `docs/adr/0004-loss-object-split.md` and `CONTEXT.md` (Training
infrastructure section) for the design contract.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from collections.abc import Callable

import torch


@dataclass
class LossComponents:
    """Unweighted per-term ELBO tensors returned by `DDSSM_base.forward()`.

    Fields are scalars (0-d tensors), already batch-aggregated. Loss
    objects apply their own weights; consumers wanting an unweighted
    sum for diagnostics use `.elbo()` / `.elbo_reg()` / `.total()`.

    The KL terms are split into a ``*_phith`` side (the ELBO-weighted
    scalar that trains encoder/decoder/baseline, i.e. φθ) and a
    ``*_psi`` side (the unit-weighted score-matching scalar for the
    score net ψ, used only under split-loss mode). The legacy names
    ``init_kl`` / ``trans_kl`` remain available as **read-only**
    property aliases of the ``*_phith`` fields — they cannot be set or
    passed as constructor kwargs. ``elbo()`` / ``elbo_reg()`` /
    ``total()`` keep their pre-split semantics (phith fields only).
    """

    recon: torch.Tensor
    init_kl_phith: torch.Tensor
    init_kl_psi: torch.Tensor
    trans_kl_phith: torch.Tensor
    trans_kl_psi: torch.Tensor
    r_sigma_p: torch.Tensor
    r_mu_p: torch.Tensor

    @property
    def init_kl(self) -> torch.Tensor:
        """Read-only alias for :attr:`init_kl_phith` (legacy name)."""
        return self.init_kl_phith

    @property
    def trans_kl(self) -> torch.Tensor:
        """Read-only alias for :attr:`trans_kl_phith` (legacy name)."""
        return self.trans_kl_phith

    def elbo(self) -> torch.Tensor:
        """Unweighted ELBO: ``recon + init_kl + trans_kl``."""
        return self.recon + self.init_kl + self.trans_kl

    def elbo_reg(self) -> torch.Tensor:
        """ELBO plus the (unweighted) centering regularizers."""
        return self.elbo() + self.r_sigma_p + self.r_mu_p

    def total(self) -> torch.Tensor:
        """Alias for :meth:`elbo_reg` — the full unweighted diagnostic sum."""
        return self.elbo_reg()


@dataclass
class SplitLoss:
    """Two-sided loss returned by split-mode loss objects.

    ``phith`` is the ELBO-weighted scalar backpropped into the
    encoder/decoder/baseline (φθ) parameters; ``psi`` is the
    unit-weighted score-matching scalar backpropped into the score
    net (ψ). Shim methods mirror the ``torch.Tensor`` surface the
    trainer's fit loop touches (``detach``/``item``/``float``/``/``).
    """

    phith: torch.Tensor
    psi: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        """Sum of both sides, for logging/diagnostics."""
        return self.phith + self.psi

    def detach(self) -> SplitLoss:
        """Return a new :class:`SplitLoss` with both sides detached."""
        return SplitLoss(phith=self.phith.detach(), psi=self.psi.detach())

    def item(self) -> float:
        """Combined scalar value: ``float((phith + psi).item())``."""
        return float((self.phith + self.psi).item())

    def __float__(self) -> float:
        """Alias for :meth:`item` — the fit loop does ``float(loss)``."""
        return self.item()

    def __truediv__(self, scalar: float | int) -> SplitLoss:
        """Scale both sides by ``1/scalar`` (grad-accumulation scaling)."""
        return SplitLoss(phith=self.phith / scalar, psi=self.psi / scalar)


class Loss(abc.ABC):
    """Combines `LossComponents` into the scalar the trainer backprops.

    Pure function of `(components, step_within_stage)`. Holds its own
    schedule shape as config; the trainer owns the step counter.
    """

    @abc.abstractmethod
    def __call__(
        self, components: LossComponents, step: int
    ) -> torch.Tensor | SplitLoss:
        """Weight and sum ``components`` into the loss to backprop.

        Args:
            components: Unweighted per-term ELBO tensors.
            step: Step index within the current stage, driving any λ
                schedule.

        Returns:
            Scalar loss tensor, or a :class:`SplitLoss` pair under
            split-loss mode.
        """
        ...

    def lambda_at(self, step: int) -> float | None:
        """Rate-λ in effect at ``step``, for logging. ``None`` if not applicable.

        Lets the trainer surface ``optim/lambda`` for any loss with a rate
        schedule, not just :class:`FullELBO`.
        """
        return None


@dataclass
class FullELBO(Loss):
    """Default loss: ELBO with always-on centering regularizers.

    Computes::

        loss = recon + rate_lambda(step) * (init_kl + trans_kl)
               + lambda_sigma_p * r_sigma_p
               + lambda_mu_p * r_mu_p

    The centering regularizers (``r_sigma_p``, ``r_mu_p``) carry their own
    per-term weights and are deliberately NOT gated by `rate_lambda`:
    σ_p collapse is most likely during recon-only warmup (λ→0) when
    the KL isn't yet pulling against the prior, so the σ_p anchor must
    stay on the whole time. See `handoff_protocol_invariants`
    — `sigma_pert > 0` is mandatory protocol.

    With ``use_split_loss=True`` the return value is a :class:`SplitLoss`
    instead: the φθ side is the composition above (over the ``*_phith``
    KL fields), and the ψ side is ``trans_kl_psi + init_kl_psi`` with NO
    ``rate_lambda`` gating — the score net trains at full strength
    through recon-only warmup (score matching is invariant to positive
    rescaling; the λ ramp's job is protecting φθ from KL through an
    imperfect ψ).
    """

    rate_lambda: Callable[[int], float]
    lambda_sigma_p: float = 0.0
    lambda_mu_p: float = 0.0
    use_split_loss: bool = False

    def lambda_at(self, step: int) -> float | None:
        """Rate-λ in effect at ``step``, for logging."""
        return float(self.rate_lambda(step))

    def __call__(
        self, components: LossComponents, step: int
    ) -> torch.Tensor | SplitLoss:
        """Compose the loss; ``SplitLoss`` when ``use_split_loss`` is set."""
        lam = self.rate_lambda(step)
        rate = components.init_kl + components.trans_kl
        reg = (
            self.lambda_sigma_p * components.r_sigma_p
            + self.lambda_mu_p * components.r_mu_p
        )
        loss_phith = components.recon + lam * rate + reg
        if not self.use_split_loss:
            return loss_phith
        loss_psi = components.trans_kl_psi + components.init_kl_psi
        return SplitLoss(phith=loss_phith, psi=loss_psi)
