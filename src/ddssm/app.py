"""Hydra-based CLI entry point for training DDSSM models.

Usage::

    # Train with the default Gaussian transition (no dataset → just builds the
    # model and trainer, then exits).
    python -m ddssm.app

    # Run a full experiment preset end-to-end on synthetic data.
    python -m ddssm.app +experiment=synthetic_gauss
    python -m ddssm.app +experiment=synthetic_diffusion

    # Override individual params from the CLI.
    python -m ddssm.app +experiment=synthetic_gauss \\
        data_dim=2 latent_dim=8 hyperparams.batch_size=32 training.steps=200

    # Optuna sweep using a pre-defined search space.
    python -m ddssm.app --multirun \\
        +experiment=synthetic_gauss \\
        +sweep=synthetic_lr \\
        hydra/sweeper=ddssm_optuna \\
        hydra.sweeper.n_trials=20 \\
        hydra.sweeper.study_name=ddssm_synth_lr \\
        hydra.sweeper.storage=sqlite:///ddssm_synth_lr.db
"""

from __future__ import annotations

import csv
import logging
import math
import os
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# Importing conf populates Hydra's ConfigStore with model/trainer/transition
# groups via the side-effecting ``store.add_to_hydra_store(...)`` call at
# the bottom of ddssm.conf. Must precede @hydra.main.
from . import conf  # noqa: F401
from .data.dataload import parse_batch

log = logging.getLogger(__name__)


def _seed_everything(seed: int | None) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) RNGs."""
    if seed is None:
        return
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_dataset_disabled(dataset_cfg: Any) -> bool:
    """Return True for the ``none`` dataset preset (or any explicitly-null target)."""
    if dataset_cfg is None:
        return True
    if isinstance(dataset_cfg, DictConfig):
        target = OmegaConf.select(dataset_cfg, "_target_", default=None)
    else:
        target = getattr(dataset_cfg, "_target_", None)
    return target in (None, "null", "")


def _build_loader(
    dataset: Dataset, training_cfg: DictConfig, batch_size: int, shuffle: bool
) -> DataLoader:
    """Wrap ``dataset`` in a ``DataLoader`` using the trainer-side batch size."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", False)),
        drop_last=bool(training_cfg.get("drop_last", False)),
    )


def _read_final_objective(csv_path: str, column: str = "loss/total") -> float:
    """Return the mean of the final ~10% of values in ``column`` from the CSV log.

    Falls back to ``+inf`` when the CSV / column is unavailable, so failed
    trials produce a sensible Optuna objective value.
    """
    if not csv_path or not os.path.isfile(csv_path):
        return float("inf")
    chosen_column = column
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if chosen_column not in fieldnames:
                chosen_column = next(
                    (h for h in fieldnames if "loss" in h.lower()), ""
                )
                if not chosen_column:
                    return float("inf")
            values: list[float] = []
            for row in reader:
                raw = row.get(chosen_column, "")
                if raw in ("", None):
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    values.append(v)
    except OSError:
        return float("inf")
    if not values:
        return float("inf")
    tail_n = max(1, len(values) // 10)
    return float(sum(values[-tail_n:]) / tail_n)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> Any:
    """Build the DDSSM model from config; run training when a dataset is wired.

    Returns:
        * the trained ``DDSSMTrainer`` when no dataset is attached, or
        * a scalar final-loss value (suitable as an Optuna objective) when
          ``cfg.dataset`` resolves to a real ``torch.utils.data.Dataset`` and
          ``cfg.training.return_objective`` is true (the default).
    """
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    _seed_everything(cfg.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device: %s", device)

    model = instantiate(cfg.model).to(device)
    log.info("Model built: %d parameters", sum(p.numel() for p in model.parameters()))

    # Hydra's runtime working directory is unique per run/sweep trial; route
    # logs and checkpoints there so concurrent trials don't collide.
    run_dir = HydraConfig.get().runtime.output_dir
    csv_log_path = os.path.join(run_dir, "metrics.csv")
    tb_log_dir = os.path.join(run_dir, "tb_logs")

    trainer = instantiate(
        cfg.trainer,
        model=model,
        device=device,
        csv_log_path=csv_log_path,
        tensorboard_dir=tb_log_dir,
    )

    # Persist the resolved config alongside the run for reproducibility.
    try:
        with open(os.path.join(run_dir, "resolved_config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
    except Exception as e:  # pragma: no cover - best-effort artefact
        log.warning("Could not persist resolved_config.yaml: %s", e)

    dataset_cfg = cfg.get("dataset", None)
    if _is_dataset_disabled(dataset_cfg):
        log.info(
            "No dataset attached (dataset=none). Skipping fit(); returning trainer."
        )
        return trainer

    log.info("Instantiating dataset.")
    train_dataset = instantiate(dataset_cfg)

    training_cfg = cfg.get("training", None)
    if training_cfg is None:
        log.warning(
            "cfg.training missing -- using minimal defaults (steps=100, log_every=10)."
        )
        training_cfg = OmegaConf.create({"steps": 100, "log_every": 10, "amp": False})

    train_loader = _build_loader(
        train_dataset,
        training_cfg,
        batch_size=trainer.get_batch_size(),
        shuffle=bool(training_cfg.get("shuffle", True)),
    )

    val_loader = None
    val_dataset_cfg = cfg.get("dataset_val", None)
    if val_dataset_cfg is not None and not _is_dataset_disabled(val_dataset_cfg):
        val_dataset = instantiate(val_dataset_cfg)
        val_loader = _build_loader(
            val_dataset,
            training_cfg,
            batch_size=trainer.get_batch_size(),
            shuffle=False,
        )

    log.info(
        "Starting trainer.fit (steps=%s, log_every=%s, validate_every=%s, amp=%s)",
        int(training_cfg.steps),
        int(training_cfg.get("log_every", 10)),
        int(training_cfg.get("validate_every", 0)),
        bool(training_cfg.get("amp", False)),
    )
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        total_steps=int(training_cfg.steps),
        validate_every=int(training_cfg.get("validate_every", 0)),
        log_every=int(training_cfg.get("log_every", 10)),
        checkpoint_every=training_cfg.get("checkpoint_every", None),
        checkpoint_prefix=training_cfg.get("checkpoint_prefix", None),
        amp=bool(training_cfg.get("amp", False)),
        resume_from=training_cfg.get("resume_from", None),
        batch_transform=parse_batch,
        profile_steps=int(training_cfg.get("profile_steps", 0)),
    )

    log.info("Training complete. Run dir: %s", run_dir)

    if not bool(training_cfg.get("return_objective", True)):
        return trainer

    objective = _read_final_objective(csv_log_path, column="loss/total")
    log.info("Optuna objective (mean tail of loss/total) = %.6g", objective)
    return objective


if __name__ == "__main__":
    main()
