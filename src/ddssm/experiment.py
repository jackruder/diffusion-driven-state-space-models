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
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

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
    ``fit``; freezing a submodule's ``requires_grad`` is what suppresses
    its gradient contribution for a stage.
    """

    steps: int = 1000
    log_every: int = 50
    validate_every: int = 0
    checkpoint_every: int | None = None
    checkpoint_prefix: str | None = None
    amp: bool = False
    profile_steps: int = 0
    resume_from: str | None = None
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
        }


@dataclass
class ObjectiveSpec:
    """How the experiment turns a finished run into an Optuna objective.

    Two sources:

    * ``source="csv"`` (default, legacy): reads the trainer's
      ``metrics.csv`` and returns the mean of the final ``tail_frac``
      of values in ``metric``.  ``split`` filters rows by the ``split``
      column.

    * ``source="json"``: reads the eval pipeline's ``metrics.json`` and
      returns the scalar at key ``metric``.  Used by Phase-C/D Optuna
      pilots whose objective is a post-training eval metric (e.g.
      :func:`ddssm.eval.metrics.eval_stage2_elbo_surrogate`).  ``split``
      and ``tail_frac`` are ignored in this mode.

    When the primary value is unavailable (file missing, key absent,
    value ``None`` or non-finite), the spec applies its ``penalty``:

    * ``"inf"`` (default) — return ``+inf`` so the trial sorts last.
    * ``"csv_tail_time"`` — substitute the last ``time/elapsed_s`` from
      ``metrics.csv``. Use for wall-clock-to-target style objectives
      where "never reached" should cost the trial's full training time
      (its compute budget) rather than an unbounded sentinel — keeps
      misses on the same units as hits.
    """

    metric: str = "loss/total"
    split: str = "train"
    tail_frac: float = 0.1
    source: Literal["csv", "json"] = "csv"
    penalty: Literal["inf", "csv_tail_time"] = "inf"

    def read(self, run_dir_or_csv: str) -> float:
        """Read the objective value from ``run_dir`` (or, legacy: a CSV path).

        Backward-compatibility: if ``run_dir_or_csv`` points at an
        existing file (not a directory) the CSV source is read directly
        from that path — preserving the pre-Phase-C call signature used
        by ``Experiment.train`` and the variance-probe family.
        """
        if self.source == "json":
            return self._read_json(run_dir_or_csv)
        return self._read_csv(run_dir_or_csv)

    def _apply_penalty(self, run_dir_or_csv: str) -> float:
        """Resolve the configured penalty when the primary value is unavailable."""
        if self.penalty == "csv_tail_time":
            return self._tail_time_from_csv(run_dir_or_csv)
        return float("inf")

    @staticmethod
    def _tail_time_from_csv(run_dir_or_csv: str) -> float:
        """Last finite ``time/elapsed_s`` from ``metrics.csv``, or ``+inf``."""
        if not run_dir_or_csv:
            return float("inf")
        csv_path = (
            os.path.join(run_dir_or_csv, "metrics.csv")
            if os.path.isdir(run_dir_or_csv) else run_dir_or_csv
        )
        if not os.path.isfile(csv_path):
            return float("inf")
        last_time: float | None = None
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw = row.get("time/elapsed_s", "")
                    if raw in ("", None):
                        continue
                    try:
                        v = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(v):
                        last_time = v
        except OSError:
            return float("inf")
        return last_time if last_time is not None else float("inf")

    def _read_csv(self, path: str) -> float:
        # If caller passed a run_dir, append the conventional filename.
        if path and os.path.isdir(path):
            csv_path = os.path.join(path, "metrics.csv")
        else:
            csv_path = path
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

    def _read_json(self, run_dir: str) -> float:
        if not run_dir:
            return float("inf")
        json_path = (
            run_dir if os.path.isfile(run_dir)
            else os.path.join(run_dir, "metrics.json")
        )
        if not os.path.isfile(json_path):
            return float("inf")
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return float("inf")
        value = data.get(self.metric)
        if value is None:
            return self._apply_penalty(run_dir)
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self._apply_penalty(run_dir)
        return v if math.isfinite(v) else self._apply_penalty(run_dir)


@dataclass
class SBatch:
    """Slurm resource request attached to an experiment.

    Read by ``experiments._sbatch.render_sbatch`` when emitting a
    submit script via ``python -m experiments sbatch <name>``. Ignored
    at training time. Most experiments leave this ``None`` on the
    :class:`Experiment` and inherit the project-default ``SBatch``
    from ``experiments._sbatch``; override here for runs that need
    e.g. ``time="12:00:00"`` or a different partition.
    """

    partition: str = "gpu"
    time: str = "04:00:00"
    gpus: int = 1
    cpus: int = 4
    mem: str = "32G"
    nodes: int = 1
    job_name: str | None = None
    extra_flags: tuple[str, ...] = ()


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
    objective: ObjectiveSpec | list[ObjectiveSpec] | None = None
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
    # Slurm resource request, consumed by ``python -m experiments
    # sbatch``. Purely metadata at training time.
    sbatch: SBatch | None = None

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

        # Anchor the checkpoint directory inside ``run_dir`` so a run's
        # outputs are self-contained — Hydra defaults to ``chdir=False``,
        # so the model's class-default ``./checkpoints`` would otherwise
        # land next to the invocation CWD rather than the run.
        self.model.config.checkpoint_dir = os.path.join(run_dir, "checkpoints")

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

        stages_cfg = getattr(self.model.config, "stages", None)
        if stages_cfg is not None and getattr(stages_cfg, "run", None):
            # Multi-stage path: drive StageOrchestrator instead of a single fit.
            from .stages import StageOrchestrator

            log.info(
                "Starting multi-stage run via StageOrchestrator (stages=%s)",
                stages_cfg.run,
            )
            orchestrator = StageOrchestrator(trainer, self.model.config)
            orchestrator.run(
                train_loader=train_loader,
                val_loader=val_loader,
                amp=self.training.amp,
                batch_transform=self.data.batch_transform,
            )
        else:
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

        # Multi-objective: ``objective`` is a list of ObjectiveSpec.
        # Each is resolved the same way as a single-objective spec; the
        # returned list is passed straight through to Optuna, which
        # interprets it according to the sweeper's ``direction:`` list.
        #
        # When the experiment came through Hydra, ``self.objective`` is
        # a ListConfig (not a Python list) and the list elements may
        # still be DictConfig / dataclass configs (recursive
        # instantiation doesn't dive into ``Any``-typed list fields).
        # Lazy-instantiate so the same code handles "configured" and
        # "raw Python" call sites.
        try:
            from omegaconf import ListConfig
        except ImportError:
            ListConfig = ()  # type: ignore[misc,assignment]
        is_multi = isinstance(self.objective, (list, tuple, ListConfig))
        raw_objectives: list = (
            list(self.objective) if is_multi else [self.objective]
        )
        # OmegaConf strips ``_target_`` from list-of-dataclass-conf
        # elements during outer instantiation, so we can't call
        # ``hydra.utils.instantiate`` on the leftover DictConfig. The
        # remaining fields ARE ObjectiveSpec.__init__ kwargs though,
        # so rebuild directly.
        objectives: list[ObjectiveSpec] = []
        for o in raw_objectives:
            if isinstance(o, ObjectiveSpec):
                objectives.append(o)
                continue
            try:
                from omegaconf import OmegaConf
                fields = OmegaConf.to_container(o, resolve=True)  # type: ignore[arg-type]
            except (ImportError, TypeError, ValueError):
                fields = dict(o) if hasattr(o, "keys") else {}
            fields = {k: v for k, v in fields.items() if k != "_target_"}
            objectives.append(ObjectiveSpec(**fields))

        # If any objective reads from metrics.json we need to evaluate
        # before reading. Do it once and reuse for all json-source specs.
        needs_eval = any(
            getattr(o, "source", "csv") == "json" for o in objectives
        )
        if needs_eval:
            if self.eval is None:
                log.warning(
                    "json-source objective configured but self.eval is "
                    "None; returning +inf for every objective so the "
                    "trial is skipped cleanly."
                )
                penalty_vals = [float("inf")] * len(raw_objectives)
                return penalty_vals if is_multi else penalty_vals[0]
            final_ckpt = os.path.join(
                self.model.config.checkpoint_dir, "ckpt_final.pth",
            )
            trainer.save_checkpoint(final_ckpt)
            log.info("Saved final checkpoint to %s", final_ckpt)
            self.evaluate(
                device=device, run_dir=run_dir,
                checkpoint_path=final_ckpt, csv_path=csv_log_path,
            )

        values: list[float] = []
        for o in objectives:
            if o.source == "json":
                v = o.read(run_dir)
                log.info("Objective[json/%s] = %.6g", o.metric, v)
            else:
                v = o.read(csv_log_path)
                log.info(
                    "Objective[%s/%s tail=%.2f] = %.6g",
                    o.split, o.metric, o.tail_frac, v,
                )
            values.append(v)

        # Preserve scalar return shape when caller configured a single
        # ObjectiveSpec — keeps the legacy single-objective Optuna path
        # untouched.
        return values if is_multi else values[0]

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
    "SBatch",
]
