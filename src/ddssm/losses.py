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
from typing import Callable
from dataclasses import dataclass

import torch


@dataclass
class LossComponents:
    """Unweighted per-term ELBO tensors returned by `DDSSM_base.forward()`.

    Fields are scalars (0-d tensors), already batch-aggregated. Loss
    objects apply their own weights; consumers wanting an unweighted
    sum for diagnostics use `.elbo()` / `.elbo_reg()` / `.total()`.
    """

    recon: torch.Tensor
    init_kl: torch.Tensor
    trans_kl: torch.Tensor
    r_sigma_p: torch.Tensor
    r_mu_p: torch.Tensor

    def elbo(self) -> torch.Tensor:
        return self.recon + self.init_kl + self.trans_kl

    def elbo_reg(self) -> torch.Tensor:
        return self.elbo() + self.r_sigma_p + self.r_mu_p

    def total(self) -> torch.Tensor:
        return self.elbo_reg()


class Loss(abc.ABC):
    """Combines `LossComponents` into the scalar the trainer backprops.

    Pure function of `(components, step_within_stage)`. Holds its own
    schedule shape as config; the trainer owns the step counter.
    """

    @abc.abstractmethod
    def __call__(
        self, components: LossComponents, step: int
    ) -> torch.Tensor: ...


@dataclass
class FullELBO(Loss):
    """Default loss: ELBO with always-on centering regularizers.

    `loss = recon + rate_lambda(step) * (init_kl + trans_kl)
            + lambda_sigma_p * r_sigma_p
            + lambda_mu_p * r_mu_p`

    The centering regularizers (`r_sigma_p`, `r_mu_p`) carry their own
    per-term weights and are deliberately NOT gated by `rate_lambda`:
    σ_p collapse is most likely during recon-only warmup (λ→0) when
    the KL isn't yet pulling against the prior, so the σ_p anchor must
    stay on the whole time. See `project_handoff_protocol_invariants`
    — `sigma_pert > 0` is mandatory protocol.
    """

    rate_lambda: Callable[[int], float]
    lambda_sigma_p: float = 0.0
    lambda_mu_p: float = 0.0

    def __call__(
        self, components: LossComponents, step: int
    ) -> torch.Tensor:
        lam = self.rate_lambda(step)
        rate = components.init_kl + components.trans_kl
        reg = (
            self.lambda_sigma_p * components.r_sigma_p
            + self.lambda_mu_p * components.r_mu_p
        )
        return components.recon + lam * rate + reg
