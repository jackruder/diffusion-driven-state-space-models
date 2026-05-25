"""Multi-stage training orchestration via StageOrchestrator.

A stage = a contiguous block of training with a per-stage trainable
mask, per-stage learning rates, and an optional one-time
``centering_handoff`` hook that fires *before* the stage's training
loop (used at the stage-1 → stage-2 boundary per
``model-v2.org`` § Stage-1 → stage-2 handoff).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List

from omegaconf import MISSING

from .centering.handoff import CenteringHandoffConf, perform_centering_handoff

if TYPE_CHECKING:
    from .train import DDSSMTrainer


@dataclass
class StageLrsConf:
    """Per-stage learning rates passed into ``trainer._rebuild_optimizer``."""

    enc_lr: float = 5e-4
    dec_lr: float = 5e-4
    zinit_lr: float = 5e-4
    trans_lr: float = 5e-4


@dataclass
class StageTrainableConf:
    """Per-module ``requires_grad`` mask for the stage.

    Matches the slot names used by :meth:`DDSSMTrainer._set_trainable`
    (encoder / decoder / zinit / transition).  Note: ``z_init`` is the
    legacy InitPrior; under the model-v2 VHP-via-diffusion path the
    aux posterior is part of the *encoder* family (via DDSSM_base's
    ``aux_posterior`` slot) and shares the encoder flag.
    """

    encoder: bool = True
    decoder: bool = True
    z_init: bool = True
    transition: bool = True


@dataclass
class StageSchedulerConf:
    warmup_steps: int = 0
    final_lr_scale: float = 1.0


@dataclass
class LambdaRampConf:
    end: float | None = 1.0
    delay: int = 0
    steps: int | None = None
    start: float = 0.001


@dataclass
class StageSpecConf:
    """A single stage of multi-stage training.

    Optional ``centering_handoff`` fires *before* this stage's training
    loop.  When set, the handoff rebuilds the optimizer itself; the
    orchestrator then skips its own ``_rebuild_optimizer`` call.
    """

    steps: int = MISSING
    trainable: StageTrainableConf = field(default_factory=StageTrainableConf)
    lrs: StageLrsConf = field(default_factory=StageLrsConf)
    scheduler: StageSchedulerConf = field(default_factory=StageSchedulerConf)
    lambda_ramp: LambdaRampConf = field(default_factory=LambdaRampConf)
    log_every: int = 10
    val_every: int = 100
    checkpoint_every: int = 1000
    centering_handoff: CenteringHandoffConf | None = None


@dataclass
class StagesConf:
    stage_1: StageSpecConf | None = None
    stage_2: StageSpecConf | None = None
    stage_3: StageSpecConf | None = None
    run: List[str] = field(default_factory=lambda: ["stage_1", "stage_2"])


def make_lambda_cosine(spec: LambdaRampConf, total_steps: int, default_end: float) -> Callable[[int], float]:
    """Build a cosine λ-ramp schedule from a ``LambdaRamp`` spec.

    Args:
        spec: ``LambdaRampConf`` with ``start``, ``end``, ``delay``, ``steps``.
        total_steps: Fallback total step count when ``spec.steps`` is ``None``.
        default_end: Fallback end value when ``spec.end`` is ``None``.

    Returns:
        A callable ``f(step_idx: int) -> float`` returning λ at a given
        1-based step.
    """
    end = spec.end if spec.end is not None else default_end
    ramp_T = spec.steps if spec.steps is not None else total_steps
    delay = max(0, int(spec.delay))

    def f(step_idx: int) -> float:
        t = max(0, step_idx - delay)
        T = max(1, ramp_T - delay)
        u = min(1.0, t / T)
        return float(end + 0.5 * (spec.start - end) * (1.0 + math.cos(math.pi * u)))

    return f


class StageOrchestrator:
    """Runs a sequence of training stages defined in ``DDSSMConfig.stages``.

    For each stage in ``stages.run``:

    1. Flip ``trainer.model.stage_selector`` to the stage key.
    2. If ``stage.centering_handoff`` is set, call
       :func:`perform_centering_handoff` with the stage's LRs.  The
       handoff rebuilds the optimizer itself, so the orchestrator
       *skips* its own ``_rebuild_optimizer`` step.
    3. Otherwise, rebuild the optimizer with the stage's LRs.
    4. Set per-module trainable flags.
    5. Drive ``trainer.fit`` for ``stage.steps`` training steps.
    """

    def __init__(self, trainer: "DDSSMTrainer", config) -> None:
        self.trainer = trainer
        self.cfg = config

    def run(
        self,
        train_loader,
        val_loader=None,
        amp: bool = False,
        batch_transform=None,
    ) -> None:
        """Execute every stage listed in ``config.stages.run``."""
        stages = self.cfg.stages
        if stages is None:
            raise AttributeError(
                "StageOrchestrator.run requires config.stages to be set"
            )

        for key in stages.run:
            stage: StageSpecConf | None = getattr(stages, key, None)
            if stage is None:
                continue
            print(f"\n=== Running {key} for {stage.steps} steps ===")

            # 1. Flip stage selector so DDSSM_base's dispatch picks the right
            # transition + correctly handles entropy / regularizers per stage.
            if hasattr(self.trainer.model, "stage_selector"):
                self.trainer.model.stage_selector = key

            # 2. Optional centering handoff.  When set, the handoff rebuilds
            # the optimizer itself.
            if stage.centering_handoff is not None:
                perform_centering_handoff(
                    self.trainer, stage.centering_handoff, new_lrs=stage.lrs,
                )
            else:
                self.trainer._rebuild_optimizer(stage.lrs)

            # 3. Per-module trainable flags.
            self.trainer._set_trainable(stage.trainable)

            # 4. Run the stage's training loop.
            self.trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                total_steps=stage.steps,
                validate_every=stage.val_every,
                log_every=stage.log_every,
                checkpoint_every=stage.checkpoint_every,
                checkpoint_prefix=f"{key}",
                amp=amp,
                batch_transform=batch_transform,
            )
