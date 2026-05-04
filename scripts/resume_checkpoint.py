import math
from typing import Literal, Optional

import torch
from torch.utils.data import DataLoader

from dkdm.train import DSSDTrainer
from dkdm.config import DSSDConfig
from dkdm.dataload import get_solar_loaders


def resume_training_stage(
    checkpoint_path: str,
    config_path: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    stage_key: Literal["stage_1", "stage_2", "stage_3"] = "stage_2",
    is_new_stage: bool = True,
    amp: bool = False,
    strict: bool = True,  # whether to strictly enforce that the keys in state_dict of checkpoint match the model
    csv_log_path: Optional[str] = None,
    tensorboard_dir: Optional[str] = None,
):
    """Resume training either mid-stage (is_new_stage=False) or start the next stage
    from a previous stage's checkpoint (is_new_stage=True).
    """
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )

    # Load config/model/trainer
    trainer = DSSDTrainer.load_from_yaml(
        config_path,
        device,
        csv_log_path=csv_log_path,
        tensorboard_dir=tensorboard_dir,
    )

    cfg = DSSDConfig.load_yaml(config_path)

    stage_spec = getattr(cfg.stages, stage_key)
    if not is_new_stage:
        trainer._set_trainable(stage_spec.trainable)
        trainer._rebuild_optimizer(stage_spec.lrs)

    meta = trainer.restore_from_checkpoint(checkpoint_path, strict=strict)
    print("Restored:", meta)

    if is_new_stage:
        # reset optimizer and EMA
        trainer._set_trainable(stage_spec.trainable)
        trainer._rebuild_optimizer(stage_spec.lrs)
        trainer.ema.shadow = {
            k: v.detach().clone()
            for k, v in trainer.model.diffmodel.state_dict().items()
        }
        start_step = 0
    else:
        start_step = int(meta.get("stage_step", 0))

    lambda_schedule = None
    if hasattr(stage_spec, "lambda_ramp"):
        ramp = stage_spec.lambda_ramp

        def lambda_schedule_fn(step: int):
            if step <= ramp.delay:
                return ramp.start
            t = min(1.0, (step - ramp.delay) / max(1, ramp.steps))
            # cosine ease from start -> end
            return ramp.start + (ramp.end - ramp.start) * 0.5 * (
                1 - math.cos(math.pi * t)
            )

        lambda_schedule = lambda_schedule_fn

    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        total_steps=stage_spec.steps,
        start_step=start_step,
        stage_key=stage_key,  # string key
        stage_mode=stage_spec.mode,
        validate_every=stage_spec.val_every,
        log_every=stage_spec.log_every,
        checkpoint_every=stage_spec.checkpoint_every,
        checkpoint_prefix=stage_key,  # string key
        amp=amp,
        lambda_schedule=lambda_schedule,
    )

    return trainer


if __name__ == "__main__":
    checkpoint_path = "./checkpoints/solar3/ckpt_stage_2_step600.pth"
    config_path = "./configs/solar3.yaml"
    config = DSSDConfig.load_yaml(config_path)
    train_loader, val_loader, test_loader, (mu_t, stds_t) = get_solar_loaders(
        batch_size=config.hyperparams.batch_size
    )
    resume_training_stage(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        train_loader=train_loader,
        val_loader=val_loader,
        stage_key="stage_3",
        is_new_stage=True,
        amp=False,
        strict=True,
        csv_log_path="./logs/solar3.csv",
        tensorboard_dir="./runs/solar3",
    )
