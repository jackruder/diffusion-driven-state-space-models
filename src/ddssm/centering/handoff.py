"""Stage-1 → stage-2 centering handoff.

Implements the seven-step protocol from ``model-v2.org`` § Stage-1 → stage-2
handoff (the items relevant for this PR — zero-init F_ψ is build-time, LR
warmup is deferred).  Steps in this exact order:

1. *Snapshot μ_p* — ``trainer.model.baseline_anchor =
   trainer.model.baseline.snapshot()``.
2. *Rebuild optimizer* — ``trainer._rebuild_optimizer(new_lrs)``.  Discards
   Adam moments and the step counter; the rationale is that stage-1
   moments are calibrated to the reconstruction + Gaussian-KL surface,
   while stage-2 gradients have substantially different structure.
3. *Perturb encoder* — ``φ ← φ + σ_pert · ε`` with ``ε ∼ N(0, I)``.
   Only ``model.encoder.parameters()``: per
   ``init-experiment.org`` § Fixed handoff-protocol decisions, the
   noise-injection target is "full encoder weights only" — not the aux
   posterior, baseline, decoder, transition, or σ_data buffer.
4. *Reset σ_data EMA schedule* — ``trainer.model.sigma_data.reset_schedule()``.
   Buffer values persist (they have accumulated passively throughout
   stage 1); only the EMA *schedule* (step counter, "fixed" frozen
   flag) resets.

Zero-initialising F_ψ's final layer is **not** a handoff step — V3
constructs its U-Net with ``zero_init_output=True`` and stage 1 does not
train it, so the zero-init naturally carries through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..stages import StageLrsConf
    from ..train import DDSSMTrainer


@dataclass
class CenteringHandoffConf:
    """Configuration for :func:`perform_centering_handoff`.

    Only ``σ_pert`` is tunable — the remaining protocol decisions
    (encoder-only noise target, full optimizer reset, σ_data buffer
    continuity) are fixed per ``init-experiment.org`` § Fixed handoff-
    protocol decisions.
    """

    sigma_pert: float = 1e-2


@torch.no_grad()
def perform_centering_handoff(
    trainer: "DDSSMTrainer",
    spec: CenteringHandoffConf,
    *,
    new_lrs: "StageLrsConf",
) -> None:
    """Execute the stage-1 → stage-2 handoff in order."""
    model = trainer.model

    # ---- 1. Snapshot μ_p ----
    if getattr(model, "baseline", None) is None:
        raise AttributeError(
            "perform_centering_handoff: trainer.model.baseline is None; "
            "the V3 path requires a Baseline to snapshot."
        )
    model.baseline_anchor = model.baseline.snapshot()

    # ---- 2. Rebuild optimizer ----
    trainer._rebuild_optimizer(new_lrs)

    # ---- 3. Perturb encoder weights ----
    sigma_pert = float(spec.sigma_pert)
    if sigma_pert > 0.0:
        if getattr(model, "encoder", None) is None:
            raise AttributeError(
                "perform_centering_handoff: trainer.model.encoder is None"
            )
        for p in model.encoder.parameters():
            p.data.add_(sigma_pert * torch.randn_like(p))

    # ---- 4. Reset σ_data EMA schedule ----
    if getattr(model, "sigma_data", None) is not None:
        model.sigma_data.reset_schedule()
