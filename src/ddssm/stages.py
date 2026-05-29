"""Multi-stage training orchestration via StageOrchestrator.

A stage = a contiguous block of training with a per-stage trainable
mask, per-stage learning rates, and an optional one-time
``centering_handoff`` hook that fires *before* the stage's training
loop (used at the stage-1 → stage-2 boundary per
``model-v2.org`` § Stage-1 → stage-2 handoff).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Callable
from dataclasses import field, dataclass

from omegaconf import MISSING

from .centering.handoff import CenteringHandoffConf, perform_centering_handoff
from .losses import FullELBO, Loss

if TYPE_CHECKING:
    from .train import DDSSMTrainer


@dataclass
class StageLrsConf:
    """Per-stage learning rates passed into ``trainer._rebuild_optimizer``."""

    enc_lr: float = 5e-4
    dec_lr: float = 5e-4
    trans_lr: float = 5e-4


@dataclass
class StageTrainableConf:
    """Per-module ``requires_grad`` mask for the stage.

    Matches the slot names used by :meth:`DDSSMTrainer._set_trainable`
    (encoder / decoder / transition / baseline).  The aux posterior is
    part of the *encoder* family (via DDSSM_base's ``aux_posterior``
    slot) and shares the encoder flag.  ``baseline`` controls the
    optional μ_p head from ``model-v2.org`` § Generative baseline;
    stage 1 typically trains it and stage 2 freezes it under Pinned
    mode (the :func:`perform_centering_handoff` call also enforces the
    freeze independently as a belt-and-suspenders safeguard).
    """

    encoder: bool = True
    decoder: bool = True
    transition: bool = True
    baseline: bool = True


@dataclass
class EarlyStopSpec:
    """ELBO-plateau early-stop spec for a single stage.

    The trainer maintains a rolling window of ``loss/total`` values
    (one entry per logged train step).  Once at least ``window``
    entries are available *and* ``global_step >= warmup_steps``, the
    trainer compares the mean of the older half of the window against
    the mean of the newer half; if the relative drop
    ``(old_mean - new_mean) / max(|old_mean|, eps)`` is below
    ``min_improvement``, the stage exits early.

    Per ``init-experiment.org`` § Hyperparameters this lets the
    Optuna sweep over ``N_pretrain`` skip trials whose stage 1 has
    already flatlined.
    """

    enabled: bool = False
    window: int = 50
    min_improvement: float = 1e-4
    warmup_steps: int = 100


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
    early_stop: EarlyStopSpec | None = None
    # ADR-0004: per-stage loss object. None ⇒ orchestrator builds a
    # default `FullELBO` from `lambda_ramp` + model-side reg weights.
    loss: Loss | None = None


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
    """Runs a sequence of training stages defined in ``StagesConf``.

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

    def __init__(self, trainer: "DDSSMTrainer", stages: "StagesConf") -> None:
        self.trainer = trainer
        self.stages = stages

    def run(
        self,
        train_loader,
        val_loader=None,
        amp: bool = False,
        batch_transform=None,
    ) -> None:
        """Execute every stage listed in ``stages.run``."""
        stages = self.stages
        if stages is None:
            raise AttributeError(
                "StageOrchestrator.run requires a StagesConf"
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

            # 4. Install the per-stage λ schedule. Computed on the
            # stage-relative step counter (resets at every stage
            # boundary). Per ADR-0004 the loss object owns the
            # schedule shape; we still keep ``_stage_lambda_fn`` as a
            # back-compat handle so legacy callers can introspect.
            default_end = 1.0
            stage_rate_lambda = make_lambda_cosine(
                stage.lambda_ramp,
                total_steps=int(stage.steps),
                default_end=default_end,
            )
            self.trainer._stage_lambda_fn = stage_rate_lambda
            self.trainer._stage_start_step = int(
                getattr(self.trainer, "global_step", 0)
            )
            # Install the active loss object for this stage. If the
            # preset declared `stage.loss`, use it. Otherwise build a
            # default `FullELBO` from the stage's `lambda_ramp` and
            # the model's `anchor_lambda` (the latter is the only
            # non-loss-object source of λ_μp until anchor_lambda
            # itself migrates onto the loss object — see ADR-0004
            # follow-up).
            if stage.loss is not None:
                self.trainer._active_loss = stage.loss
            else:
                lambda_mu_p = float(
                    getattr(self.trainer.model, "anchor_lambda", 0.0) or 0.0
                )
                self.trainer._active_loss = FullELBO(
                    rate_lambda=stage_rate_lambda,
                    lambda_sigma_p=0.0,
                    lambda_mu_p=lambda_mu_p,
                )

            # 5. Run the stage's training loop.  ``trainer.fit``'s
            # ``total_steps`` is the *cumulative* max step, not per-stage,
            # so we add the global step counter to the stage's budget.
            target = int(self.trainer.global_step) + int(stage.steps)
            self.trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                total_steps=target,
                validate_every=stage.val_every,
                log_every=stage.log_every,
                checkpoint_every=stage.checkpoint_every,
                checkpoint_prefix=f"{key}",
                amp=amp,
                batch_transform=batch_transform,
                early_stop=stage.early_stop,
            )
