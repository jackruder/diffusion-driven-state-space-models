import math

from .train import DSSDTrainer
from .config import DSSDConfig


def make_lambda_cosine(spec, total_steps: int, default_end: float):
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
    def __init__(self, trainer: "DSSDTrainer", config: DSSDConfig):
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
        stages = self.cfg.stages
        assert stages is not None

        prev_opt = None
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

            prev_opt = self.trainer.optimizer
