"""Multi-stage training orchestration via StageOrchestrator."""

import math
from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING

from .train import DDSSMTrainer


@dataclass
class StageLrsConf:
    dec_lr: float = 5e-4
    zinit_lr: float = 5e-4
    trans_lr: float = 0.0


@dataclass
class StageTrainableConf:
    decoder: bool = True
    z_init: bool = True
    transition: bool = False


@dataclass
class StageSchedulerConf:
    warmup_steps: int = 0
    final_lr_scale: float = 1.0


@dataclass
class LambdaRampConf:
    end: float | None = 1.0
    delay: int = 0
    steps: int | None = None


@dataclass
class StageSpecConf:
    steps: int = MISSING
    trainable: StageTrainableConf = field(default_factory=StageTrainableConf)
    lrs: StageLrsConf = field(default_factory=StageLrsConf)
    scheduler: StageSchedulerConf = field(default_factory=StageSchedulerConf)
    carry_diff_moments: bool = False
    lambda_ramp: LambdaRampConf = field(default_factory=LambdaRampConf)
    log_every: int = 10
    val_every: int = 100
    checkpoint_every: int = 1000


@dataclass
class StagesConf:
    stage_2: StageSpecConf | None = None
    stage_3: StageSpecConf | None = None
    run: List[str] = field(default_factory=lambda: ["stage_1", "stage_2", "stage_3"])


def make_lambda_cosine(spec, total_steps: int, default_end: float) -> callable:
    """Build a cosine λ-ramp schedule from a ``LambdaRamp`` spec.

    Args:
        spec: ``LambdaRamp`` config with ``start``, ``end``, ``delay``, and ``steps``.
        total_steps: Fallback total step count when ``spec.steps`` is ``None``.
        default_end: Fallback end value when ``spec.end`` is ``None``.

    Returns:
        A callable ``f(step_idx: int) -> float`` that returns the λ value at
        a given (1-based) step index.
    """
    end = spec.end if spec.end is not None else default_end
    ramp_T = spec.steps if spec.steps is not None else total_steps
    delay = max(0, int(spec.delay))

    def f(step_idx: int) -> float:
        # step_idx is 1..total_steps
        t = max(0, step_idx - delay)
        T = max(1, ramp_T - delay)
        u = min(1.0, t / T)
        # cosine from start -> end
        return float(end + 0.5 * (spec.start - end) * (1.0 + math.cos(math.pi * u)))

    return f


class StageOrchestrator:
    """Runs a sequence of training stages defined in ``DDSSMConfig.stages``.

    Each stage can target different subsets of model parameters, use
    independent learning rates, and apply its own λ-ramp schedule.  Stages
    are executed in the order specified by ``config.stages.run``.

    Args:
        trainer: The ``DDSSMTrainer`` instance to drive.
        config: The top-level model config containing ``stages``.
    """

    def __init__(self, trainer: "DDSSMTrainer", config):
        self.trainer = trainer
        self.cfg = config

    def run(
        self,
        train_loader,
        val_loader=None,
        amp=False,
        resume_path: str | None = None,
        batch_transform=None,
    ):
        """Execute all stages in sequence.

        Args:
            train_loader: Training ``DataLoader``.
            val_loader: Optional validation ``DataLoader``.
            amp: Whether to use automatic mixed precision.
            resume_path: Optional checkpoint path to restore at the start of the
                first matching stage.  Cleared after first use so subsequent
                stages start from scratch.
            batch_transform: Optional callable applied to each batch before the
                model forward pass.
        """
        stages = self.cfg.stages
        assert stages is not None

        for key in stages.run:
            stage = getattr(stages, key)
            print(f"\n=== Running {key} ({stage.mode}) for {stage.steps} steps ===")

            self.trainer._set_trainable(stage.trainable)
            self.trainer._rebuild_optimizer(stage.lrs)

            start_step = 0
            if resume_path:
                info = self.trainer.restore_from_checkpoint(resume_path, strict=True)
                if info["stage_key"] == key:
                    start_step = int(info["stage_step"])
                # after first use, clear resume_path so later stages start fresh
                resume_path = None

            self.trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                total_steps=stage.steps,
                validate_every=stage.val_every,
                log_every=stage.log_every,
                checkpoint_every=stage.checkpoint_every,
                checkpoint_prefix=f"{key}",
                amp=amp,
                start_step=start_step,
                stage_key=key,
                stage_mode=stage.mode,
                batch_transform=batch_transform,
            )
