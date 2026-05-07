"""Experiment: composition root for a DDSSM training run.

An :class:`Experiment` ties together a :class:`DDSSMDataModule`, a model,
a trainer factory, training scalars, and an objective. ``run()`` is
called by :mod:`ddssm.app` after Hydra composes the config. The class is
intentionally a thin composition layer — no construction logic lives
here, no inheritance, no abstract methods. The Hydra config layer
handles wiring; this class handles orchestration.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from .data.datamodule import DDSSMDataModule
from .dssd import DDSSM_base
from .train import DDSSMTrainer

log = logging.getLogger(__name__)


@dataclass
class TrainingScalars:
    """Runtime knobs forwarded to :meth:`DDSSMTrainer.fit`."""

    steps: int = 1000
    log_every: int = 50
    validate_every: int = 0
    checkpoint_every: int | None = None
    checkpoint_prefix: str | None = None
    amp: bool = False
    profile_steps: int = 0
    resume_from: str | None = None
    compute_recon: bool = True
    compute_trans: bool = True

    def fit_kwargs(self) -> dict[str, Any]:
        return {
            "total_steps": int(self.steps),
            "log_every": int(self.log_every),
            "validate_every": int(self.validate_every),
            "checkpoint_every": self.checkpoint_every,
            "checkpoint_prefix": self.checkpoint_prefix,
            "amp": bool(self.amp),
            "profile_steps": int(self.profile_steps),
            "resume_from": self.resume_from,
            "compute_recon": bool(self.compute_recon),
            "compute_trans": bool(self.compute_trans),
        }


@dataclass
class ObjectiveSpec:
    """How the experiment turns a finished run into an Optuna objective.

    Reads the trainer's CSV log and returns the mean of the final
    ``tail_frac`` of values in ``metric``. ``+inf`` is returned when the
    log is missing or the column is absent so failed trials surface
    cleanly under Optuna's ``minimize`` direction.
    """

    metric: str = "loss/total"
    split: str = "train"
    tail_frac: float = 0.1

    def read(self, csv_path: str) -> float:
        if not csv_path or not os.path.isfile(csv_path):
            return float("inf")
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                col = self.metric if self.metric in fieldnames else next(
                    (h for h in fieldnames if "loss" in h.lower()), ""
                )
                if not col:
                    return float("inf")
                values: list[float] = []
                for row in reader:
                    if self.split and row.get("split", self.split) != self.split:
                        continue
                    raw = row.get(col, "")
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
        tail_n = max(1, int(len(values) * float(self.tail_frac)))
        return float(sum(values[-tail_n:]) / tail_n)


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Experiment:
    """Composition of a data module, a model, a trainer factory, and run scalars.

    The trainer is constructed lazily inside :meth:`run` because it
    needs the device and the per-trial run directory — both of which
    are owned by :mod:`ddssm.app`.

    The shape fields (``data_dim``, ``latent_dim`` etc.) are the single
    source of truth for downstream interpolation: ``DDSSMConf`` and the
    transition Confs read ``${experiment.data_dim}`` etc., so changing
    a value here propagates to the model + transition without
    duplication. The :class:`Experiment` instance carries them as
    plain fields so they remain inspectable at runtime (logging,
    debugging, checkpoints).
    """

    data: DDSSMDataModule
    model: DDSSM_base
    build_trainer: Callable[..., DDSSMTrainer]
    training: TrainingScalars = field(default_factory=TrainingScalars)
    objective: ObjectiveSpec | None = None
    seed: int | None = 0

    # Shape / wiring fields consumed by Hydra interpolation. These are
    # not used directly by ``run`` — they exist so ``DDSSMConf`` and the
    # transition Confs can interpolate from a single source of truth.
    data_dim: int = 1
    latent_dim: int = 4
    j: int = 1
    emb_time_dim: int = 16
    covariate_dim: int = 0
    use_observation_mask: bool = False
    checkpoint_dir: str = "./checkpoints"
    transition: Any = None
    hyperparams: Any = None

    def run(self, *, device: torch.device, run_dir: str) -> float | DDSSMTrainer:
        _seed_everything(self.seed)

        os.makedirs(run_dir, exist_ok=True)
        csv_log_path = os.path.join(run_dir, "metrics.csv")
        tensorboard_dir = os.path.join(run_dir, "tb_logs")

        log.info("Model: %d parameters", sum(p.numel() for p in self.model.parameters()))
        trainer = self.build_trainer(
            model=self.model,
            device=device,
            csv_log_path=csv_log_path,
            tensorboard_dir=tensorboard_dir,
        )

        train_loader = self.data.train_loader()
        if train_loader is None:
            log.info("No data attached. Returning trainer without fit().")
            return trainer

        val_loader = self.data.val_loader() if self.training.validate_every > 0 else None
        log.info(
            "Starting fit (steps=%d, log_every=%d, validate_every=%d, amp=%s)",
            self.training.steps,
            self.training.log_every,
            self.training.validate_every,
            self.training.amp,
        )
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            batch_transform=self.data.batch_transform,
            **self.training.fit_kwargs(),
        )

        if self.objective is None:
            return trainer

        value = self.objective.read(csv_log_path)
        log.info(
            "Objective[%s/%s tail=%.2f] = %.6g",
            self.objective.split, self.objective.metric, self.objective.tail_frac, value,
        )
        return value


__all__ = [
    "Experiment",
    "TrainingScalars",
    "ObjectiveSpec",
]
