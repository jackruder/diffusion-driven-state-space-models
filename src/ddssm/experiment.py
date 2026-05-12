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
class TrainableModules:
    """Per-module ``requires_grad`` flags applied via ``trainer._set_trainable``.

    Mirrors the ``StageTrainable`` shape used by the multi-stage
    pipeline so a single field can express recon-only, trans-only, or
    joint training without each experiment re-stating four flags.
    """

    encoder: bool = True
    decoder: bool = True
    z_init: bool = True
    transition: bool = True


@dataclass
class TrainingScalars:
    """Runtime knobs forwarded to :meth:`DDSSMTrainer.fit`.

    ``trainable`` is applied via ``trainer._set_trainable`` before
    ``fit``; ``compute_recon`` / ``compute_trans`` toggle which loss
    terms contribute (and thus which gradients flow). For "recon only"
    set ``trainable.transition=False`` *and* ``compute_trans=False``;
    leaving either out leaks gradients or wastes optimizer state.
    """

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
    trainable: TrainableModules | None = None

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
    """

    data: DDSSMDataModule
    model: DDSSM_base
    build_trainer: Callable[..., DDSSMTrainer]
    training: TrainingScalars = field(default_factory=TrainingScalars)
    objective: ObjectiveSpec | None = None
    eval: Any = None  # ddssm.eval.EvalSpec | None -- typed lazily to avoid circular import
    viz: Any = None  # ddssm.viz.VizSpec | None -- typed lazily to avoid circular import
    variance: Any = None  # ddssm.variance.ProbeSpec | None -- typed lazily
    seed: int | None = 0
    wandb_config: dict | None = None
    # Convenience: same Hparams instance the trainer reads via
    # ``self.model.config.hyperparams``. Exposed here so callers can
    # ``exp.hparams.lambda_warmup_steps=...`` or ``tweak(exp,
    # hparams__lr=1e-3)`` without descending into ``model.config``.
    hparams: Any = None

    def train(self, *, device: torch.device, run_dir: str) -> float | DDSSMTrainer:
        """Run training only. Eval and visualization are separate stages."""
        _seed_everything(self.seed)

        os.makedirs(run_dir, exist_ok=True)
        csv_log_path = os.path.join(run_dir, "metrics.csv")
        tensorboard_dir = os.path.join(run_dir, "tb_logs")

        # The trainer reads ``model.config.hyperparams.*``. When the caller
        # passed an :class:`Experiment`-level ``hparams`` (e.g. via
        # ``tweak(exp, hparams__lr=1e-3)``), make that the authoritative
        # value seen by the trainer.
        if self.hparams is not None:
            self.model.config.hyperparams = self.hparams

        log.info("Model: %d parameters", sum(p.numel() for p in self.model.parameters()))
        wandb_kwargs = self._wandb_kwargs(run_dir)
        trainer = self.build_trainer(
            model=self.model,
            device=device,
            csv_log_path=csv_log_path,
            tensorboard_dir=tensorboard_dir,
            wandb_config=wandb_kwargs,
        )

        if self.training.trainable is not None:
            log.info("Applying trainable mask: %s", self.training.trainable)
            trainer._set_trainable(self.training.trainable)

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

    # Backward-compat alias; ``ddssm.app`` calls ``train`` directly.
    run = train

    def evaluate(
        self,
        *,
        device: torch.device,
        run_dir: str,
        checkpoint_path: str | None = None,
        csv_path: str | None = None,
    ) -> dict:
        """Compute the metrics listed on ``self.eval`` and save metrics.json.

        Independent of ``train``: load a checkpoint, drive the data
        module's eval-split loader, write a single JSON. No training
        side effects.
        """
        if self.eval is None:
            raise ValueError(
                "Experiment.evaluate called but self.eval is None. Set an "
                "EvalSpec on the experiment to declare which metrics to "
                "compute."
            )
        # Local import keeps ``ddssm.eval`` out of the import path until
        # someone actually evaluates -- avoids importing matplotlib /
        # numpy-heavy modules during a vanilla training run.
        from .eval import evaluate as _run_evaluate

        return _run_evaluate(
            self,
            self.eval,
            device=device,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            csv_path=csv_path,
        )

    def visualize(
        self,
        *,
        device: torch.device,
        run_dir: str,
        checkpoint_path: str | None = None,
        csv_path: str | None = None,
    ) -> list[str]:
        """Run every plot listed on ``self.viz`` and return saved paths.

        Independent of ``train`` and ``evaluate``: load a checkpoint,
        produce PNGs, return the list of saved paths.
        """
        if self.viz is None:
            raise ValueError(
                "Experiment.visualize called but self.viz is None. Set a "
                "VizSpec on the experiment to declare which plots to draw."
            )
        from .viz import visualize as _run_visualize

        return _run_visualize(
            self,
            self.viz,
            device=device,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            csv_path=csv_path,
        )

    def variance_probe(
        self,
        *,
        device: torch.device,
        run_dir: str,
        checkpoint_path: str | None = None,
    ) -> dict:
        """Run the modular variance probe stage and persist outputs."""
        if self.variance is None:
            raise ValueError(
                "Experiment.variance_probe called but self.variance is None. "
                "Set a ProbeSpec on the experiment."
            )
        from .variance import variance as _run_variance

        return _run_variance(
            self,
            self.variance,
            device=device,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
        )

    def _wandb_kwargs(self, run_dir: str) -> dict | None:
        """Resolve ``wandb_config`` into kwargs for :class:`WandbLogger`.

        Returns ``None`` when wandb is disabled or unset (the trainer
        then skips constructing a ``WandbLogger``). Auto-fills the run
        directory so wandb artefacts colocate with TB / CSV under
        Hydra's per-run output dir.
        """
        cfg = self.wandb_config
        if cfg is None:
            return None
        if not bool(cfg.get("enabled", True)):
            return None
        kwargs = dict(cfg)
        kwargs.setdefault("run_dir", run_dir)
        return kwargs


__all__ = [
    "Experiment",
    "TrainingScalars",
    "TrainableModules",
    "ObjectiveSpec",
]
