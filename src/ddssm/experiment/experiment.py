"""Experiment: composition root for a DDSSM training run.

An :class:`Experiment` ties together a :class:`TimeSeriesDataModule`, a model,
a trainer factory, training scalars, and an objective. :meth:`Experiment.train`
is called by :mod:`ddssm.app` after Hydra composes the config. The class is
intentionally a thin composition layer — no construction logic lives
here, no inheritance, no abstract methods. The Hydra config layer
handles wiring; this class handles orchestration.
"""

from __future__ import annotations

import os
import csv
import json
import math
import random
from typing import Any, Literal
import logging
from dataclasses import field, dataclass
from collections.abc import Callable

import numpy as np
import torch

from ddssm.model.dssd import DDSSM_base
from ddssm.training.train import DDSSMTrainer
from ddssm.data.datamodule import TimeSeriesDataModule

log = logging.getLogger(__name__)

# Dedup objective-misconfiguration warnings: an Optuna sweep calls
# ObjectiveSpec.read once per trial, so a misconfigured metric/split would
# otherwise warn on every trial. Keyed by message text (module-level so it
# persists across trials in one process).
_OBJECTIVE_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _OBJECTIVE_WARNED:
        _OBJECTIVE_WARNED.add(msg)
        log.warning(msg)


@dataclass
class TrainingScalars:
    """Runtime knobs forwarded to :meth:`DDSSMTrainer.fit`."""

    steps: int = 1000
    log_every: int = 50
    validate_every: int = 0
    checkpoint_every: int | None = None
    checkpoint_prefix: str | None = None
    amp: bool = True  # bf16 autocast (see train.py); project default
    profile_steps: int = 0
    resume_from: str | None = None

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
    * ``"csv_tail_step"`` — substitute the last ``step`` from
      ``metrics.csv``. The step-denominated sibling of ``csv_tail_time``
      for *steps*-to-target objectives (e.g.
      ``wallclock_to_target_step``): a miss costs the full step budget,
      keeping misses on the same (step) units as hits. Unlike the
      seconds penalty this is contention-invariant, so it survives GPU
      packing / hardware differences across cells.
    """

    metric: str = "loss/total"
    split: str = "train"
    tail_frac: float = 0.1
    source: Literal["csv", "json"] = "csv"
    penalty: Literal["inf", "csv_tail_time", "csv_tail_step"] = "inf"

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
            return self._tail_column_from_csv(run_dir_or_csv, "time/elapsed_s")
        if self.penalty == "csv_tail_step":
            return self._tail_column_from_csv(run_dir_or_csv, "step")
        return float("inf")

    @staticmethod
    def _tail_column_from_csv(run_dir_or_csv: str, column: str) -> float:
        """Last finite value of ``column`` in ``metrics.csv``, or ``+inf``.

        Backs both ``csv_tail_time`` (``column="time/elapsed_s"``) and
        ``csv_tail_step`` (``column="step"``): a miss costs the trial's
        full budget on whichever axis the objective is denominated in.
        """
        if not run_dir_or_csv:
            return float("inf")
        csv_path = (
            os.path.join(run_dir_or_csv, "metrics.csv")
            if os.path.isdir(run_dir_or_csv)
            else run_dir_or_csv
        )
        if not os.path.isfile(csv_path):
            return float("inf")
        last_value: float | None = None
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw = row.get(column, "")
                    if raw in ("", None):
                        continue
                    try:
                        v = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(v):
                        last_value = v
        except OSError:
            return float("inf")
        return last_value if last_value is not None else float("inf")

    def _read_csv(self, path: str) -> float:
        # If caller passed a run_dir, append the conventional filename.
        if path and os.path.isdir(path):
            csv_path = os.path.join(path, "metrics.csv")
        else:
            csv_path = path
        if not csv_path or not os.path.isfile(csv_path):
            return float("inf")
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                if self.metric in fieldnames:
                    col = self.metric
                else:
                    col = next((h for h in fieldnames if "loss" in h.lower()), "")
                    if col:
                        _warn_once(
                            f"ObjectiveSpec metric {self.metric!r} not in "
                            f"{csv_path} columns; falling back to {col!r}."
                        )
                    else:
                        _warn_once(
                            f"ObjectiveSpec metric {self.metric!r} not in "
                            f"{csv_path} and no 'loss' column found; "
                            f"returning +inf."
                        )
                if not col:
                    return float("inf")
                # If a split filter is configured but the CSV has no split
                # column, the filter silently passes every row — surface it.
                if self.split and "split" not in fieldnames:
                    _warn_once(
                        f"ObjectiveSpec split={self.split!r} requested but "
                        f"{csv_path} has no 'split' column; not filtering by "
                        f"split."
                    )
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
                    values.append(v)
        except OSError:
            return float("inf")
        if not values:
            return float("inf")
        tail_n = max(1, int(len(values) * float(self.tail_frac)))
        tail = values[-tail_n:]
        # Non-finite rows count as divergence rather than being dropped:
        # scoring a trial whose tail went NaN/Inf on its earlier (better)
        # finite rows would let the sweeper rank diverged configs above
        # stable ones. Route through the configured penalty, same as any
        # other unavailable value.
        if not all(math.isfinite(v) for v in tail):
            return self._apply_penalty(csv_path)
        return float(sum(tail) / tail_n)

    def _read_json(self, run_dir: str) -> float:
        if not run_dir:
            return float("inf")
        json_path = (
            run_dir
            if os.path.isfile(run_dir)
            else os.path.join(run_dir, "metrics.json")
        )
        if not os.path.isfile(json_path):
            return float("inf")
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return float("inf")
        value = data.get(self.metric)
        if value is None:
            _warn_once(
                f"ObjectiveSpec metric {self.metric!r} not in {json_path}; "
                f"applying penalty {self.penalty!r}."
            )
            return self._apply_penalty(run_dir)
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self._apply_penalty(run_dir)
        return v if math.isfinite(v) else self._apply_penalty(run_dir)


def _as_objective_spec(o: object) -> ObjectiveSpec:
    """Return a live :class:`ObjectiveSpec`, rebuilding it if needed.

    When ``Objectives.specs`` carries nested specs, OmegaConf coerces
    each element into an ``ObjectiveSpec``-typed ``DictConfig`` and
    drops its ``_target_``, so ``hydra.utils.instantiate`` never turns
    it back into a real object with a ``.read`` method. Detect that case
    (anything that isn't already an ``ObjectiveSpec`` with ``read``) and
    reconstruct from the carried fields.
    """
    if isinstance(o, ObjectiveSpec):
        return o
    return ObjectiveSpec(
        metric=getattr(o, "metric", "loss/total"),
        split=getattr(o, "split", "train"),
        tail_frac=getattr(o, "tail_frac", 0.1),
        source=getattr(o, "source", "csv"),
        penalty=getattr(o, "penalty", "inf"),
    )


@dataclass
class Objectives:
    """Wrapper for multi-objective Optuna runs.

    Holds an ordered list of :class:`ObjectiveSpec` whose values are
    returned as a Python list — matched against the sweeper's
    ``direction:`` list. Use this explicit wrapper so the Hydra-zen
    config carries a proper dataclass type (not a bare list of dicts);
    that lets ``hydra.utils.instantiate`` recurse into the elements
    instead of stripping ``_target_`` from list members.

    ``specs`` is typed as bare ``list`` (no inner type) so OmegaConf's
    strict validator accepts the zen-built ``Builds_ObjectiveSpec``
    alongside plain :class:`ObjectiveSpec` instances; both work at
    runtime via duck-typing on ``.metric`` / ``.source`` / ``.read``.
    """

    specs: list = field(default_factory=list)


@dataclass
class SBatch:
    """Slurm resource request attached to an experiment.

    Read by ``ddssm.cluster.sbatch.render_sbatch`` when emitting a submit script
    via ``python -m experiments sbatch <name>``. Ignored at training time.
    Most experiments leave this ``None`` on the :class:`Experiment` and
    inherit the project-default ``SBatch`` from ``ddssm.cluster.sbatch``; override
    here for runs that need e.g. ``time="12:00:00"`` or a different
    partition.

    For Study launches the resource shape is read from each point's
    ``PointLaunch.resources`` instead (ADR-0008); this field is only the
    single-job ``python -m experiments sbatch`` path.
    """

    partition: str = "gpu"
    time: str = "04:00:00"
    gpus: int = 1
    cpus: int = 4
    mem: str = "32G"
    nodes: int = 1
    job_name: str | None = None
    extra_flags: tuple[str, ...] = ()
    # Shell lines emitted after ``cd "$SLURM_SUBMIT_DIR"`` and before the python
    # invocation — e.g. cluster ``module load`` + venv ``activate`` so ``python``
    # resolves on the compute node. Empty by default (inherit the submit env).
    setup: tuple[str, ...] = ()


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

    The trainer is constructed lazily inside :meth:`train` because it
    needs the device and the per-trial run directory — both of which
    are owned by :mod:`ddssm.app`.
    """

    data: TimeSeriesDataModule
    model: DDSSM_base
    build_trainer: Callable[..., DDSSMTrainer]
    training: TrainingScalars = field(default_factory=TrainingScalars)
    objective: ObjectiveSpec | Objectives | None = None
    eval: Any = (
        None  # ddssm.eval.EvalSpec | None -- typed lazily to avoid circular import
    )
    viz: Any = None  # ddssm.viz.VizSpec | None -- typed lazily to avoid circular import
    variance: Any = None  # ddssm.variance.ProbeSpec | None -- typed lazily
    seed: int | None = 0
    wandb_config: dict | None = None
    # Single source of truth for optimiser-side hparams. Trainer reads
    # this directly per ADR-0004 (no longer routed through
    # ``model.config.hyperparams``). Exposed here so callers can
    # ``exp.hparams.enc_lr=...`` or ``tweak(exp, hparams__lr=1e-3)``.
    hparams: Any = None
    # Slurm resource request, consumed by ``python -m experiments
    # sbatch``. Purely metadata at training time.
    sbatch: SBatch | None = None
    # Resolved YAML of ``cfg.experiment.model`` (the hydra-zen
    # ``builds()`` shape, not the runtime SimpleNamespace) — set by the
    # Hydra entry points (``app.py``, ``evaluate.py``, ``visualize.py``,
    # ``variance/cli.py``) before calling :meth:`train` /
    # :meth:`evaluate` etc. Persisted into checkpoints so the load
    # path can cross-check against a future preset edit (ADR-0005).
    model_config_yaml: str | None = None

    def train(self, *, device: torch.device, run_dir: str) -> None:
        """Run training only. Eval, viz, and objective reads are separate stages.

        Stores the constructed trainer on ``self.trainer`` so
        :meth:`objective_value` can save a final checkpoint and trigger
        a json-source evaluate without rebuilding anything.
        """
        _seed_everything(self.seed)

        os.makedirs(run_dir, exist_ok=True)
        csv_log_path = os.path.join(run_dir, "metrics.csv")
        tensorboard_dir = os.path.join(run_dir, "tb_logs")
        # Anchor the checkpoint directory inside ``run_dir`` so a run's
        # outputs are self-contained — Hydra defaults to ``chdir=False``,
        # so the trainer's default ``./checkpoints`` would otherwise land
        # next to the invocation CWD rather than the run.
        checkpoint_dir = os.path.join(run_dir, "checkpoints")

        log.info(
            "Model: %d parameters", sum(p.numel() for p in self.model.parameters())
        )
        wandb_kwargs = self._wandb_kwargs(run_dir)
        # Per ADR-0004: caller-supplied ``exp.hparams`` is the single
        # source of truth at training time. Pass it through to the
        # trainer so ``tweak(exp, hparams__lr=...)`` reaches optim.
        # When ``build_trainer`` is the canonical ``TrainerPartial``
        # built by :func:`experiments._make.experiment`, ``hparams``
        # is already curried in; this ``hparams=`` override wins so
        # post-instantiation tweaks take effect.
        trainer_kwargs: dict[str, Any] = dict(
            model=self.model,
            device=device,
            csv_log_path=csv_log_path,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            wandb_config=wandb_kwargs,
        )
        if self.hparams is not None:
            trainer_kwargs["hparams"] = self.hparams
        if self.model_config_yaml is not None:
            trainer_kwargs["model_config_yaml"] = self.model_config_yaml
        trainer = self.build_trainer(**trainer_kwargs)
        self.trainer = trainer

        # hparams.batch_size is the single source of truth for the loader
        # batch size (ADR-0004: hparams owns runtime knobs). Reconcile it
        # onto the data module before building loaders so a CLI override of
        # experiment.hparams.batch_size actually takes effect — the data
        # preset's own batch_size is otherwise what the DataLoader would use.
        hp_bs = getattr(self.hparams, "batch_size", None)
        if hp_bs is not None and hasattr(self.data, "batch_size"):
            if self.data.batch_size != hp_bs:
                log.info(
                    "DataLoader batch_size := hparams.batch_size (%d); data "
                    "module configured %s, overridden.",
                    hp_bs,
                    self.data.batch_size,
                )
            self.data.batch_size = hp_bs

        train_loader = self.data.train_loader()
        if train_loader is None:
            log.info("No data attached. Skipping fit().")
            return

        val_loader = (
            self.data.val_loader() if self.training.validate_every > 0 else None
        )

        try:
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
        finally:
            # Logger lifecycle is owned by the run, not by fit(). Close the
            # CSV/TB/W&B sinks exactly once, after fit (or on the exception
            # path — W&B's close uploads the final checkpoint artifact
            # best-effort).
            trainer.metrics.close()

        # Emit run_summary.json so the run is self-describing (final loss,
        # λ-warmup state, σ_data² drift, val loss, non-finite count).
        try:
            from ddssm.cluster.report import write_run_summary

            write_run_summary(run_dir)
        except Exception as e:  # never let summary-writing fail a finished run
            log.warning("Could not write run_summary.json: %s", e)

    def objective_value(
        self,
        *,
        device: torch.device,
        run_dir: str,
    ) -> float | list[float] | None:
        """Resolve ``self.objective`` against the artefacts produced by :meth:`train`.

        Returns:
          * ``None`` when ``self.objective`` is unset.
          * A scalar ``float`` for a single :class:`ObjectiveSpec`.
          * A ``list[float]`` (matched against the sweeper's
            ``direction:`` list) when ``self.objective`` is an
            :class:`Objectives` wrapper.

        Requires :meth:`train` to have been called first when any
        objective has ``source="json"`` — the json-source branch saves
        a final checkpoint, runs :meth:`evaluate`, and reads the
        resulting ``metrics.json``.
        """
        if self.objective is None:
            return None

        csv_log_path = os.path.join(run_dir, "metrics.csv")

        # Multi-objective: ``Objectives(specs=[...])`` wraps an ordered
        # list whose values are returned as a Python list and matched
        # against the sweeper's ``direction:`` list. Single-objective:
        # ``ObjectiveSpec`` returns a scalar.
        is_multi = isinstance(self.objective, Objectives)
        objectives: list[ObjectiveSpec] = [
            _as_objective_spec(o)
            for o in (list(self.objective.specs) if is_multi else [self.objective])
        ]

        # If any objective reads from metrics.json we need to evaluate
        # before reading. Do it once and reuse for all json-source specs.
        needs_eval = any(getattr(o, "source", "csv") == "json" for o in objectives)
        if needs_eval:
            if self.eval is None:
                log.warning(
                    "json-source objective configured but self.eval is "
                    "None; returning +inf for every objective so the "
                    "trial is skipped cleanly."
                )
                penalty_vals = [float("inf")] * len(objectives)
                return penalty_vals if is_multi else penalty_vals[0]
            trainer = getattr(self, "trainer", None)
            if trainer is None:
                raise RuntimeError(
                    "Experiment.objective_value: json-source objective "
                    "requires that .train() ran first to populate "
                    "self.trainer."
                )
            final_ckpt = os.path.join(
                run_dir,
                "checkpoints",
                "ckpt_final.pth",
            )
            trainer.save_checkpoint(final_ckpt)
            log.info("Saved final checkpoint to %s", final_ckpt)
            self.evaluate(
                device=device,
                run_dir=run_dir,
                checkpoint_path=final_ckpt,
                csv_path=csv_log_path,
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
                    o.split,
                    o.metric,
                    o.tail_frac,
                    v,
                )
            values.append(v)

        # Preserve scalar return shape when caller configured a single
        # ObjectiveSpec — keeps the legacy single-objective Optuna path
        # untouched.
        return values if is_multi else values[0]

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
        from ddssm.eval import evaluate as _run_evaluate

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
        from ddssm.viz import visualize as _run_visualize

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
        from ddssm.variance import variance as _run_variance

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
    "ObjectiveSpec",
    "Objectives",
    "SBatch",
    "TrainingScalars",
]
