"""CSV, TensorBoard, and Weights & Biases logging utilities for tracking training metrics."""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
import csv
import math
import time
from typing import Any
import fnmatch
import logging
from dataclasses import dataclass
from collections.abc import Callable

import torch

log = logging.getLogger(__name__)


# ---------- meters ----------
class Meter(ABC):
    @abstractmethod
    def add(self, x: float, w: float = 1.0): ...

    @abstractmethod
    def value(self) -> float: ...

    @abstractmethod
    def reset(self): ...


class MeanMeter(Meter):
    def __init__(self):
        self.s = 0.0
        self.w = 0.0

    def add(self, x, w=1.0):
        self.s += float(x) * float(w)
        self.w += float(w)

    def value(self):
        return self.s / max(self.w, 1e-12)

    def reset(self):
        self.s = 0.0
        self.w = 0.0


class SumMeter(Meter):
    def __init__(self):
        self.s = 0.0

    def add(self, x, w=1.0):
        self.s += float(x) * float(w)  # sum of weighted values

    def value(self):
        return self.s

    def reset(self):
        self.s = 0.0


class LastMeter(Meter):
    def __init__(self):
        self.x = 0.0

    def add(self, x, w=1.0):
        self.x = float(x)

    def value(self):
        return self.x

    def reset(self):
        self.x = 0.0


class EMAMeter(Meter):
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._init = False
        self.m = 0.0

    def add(self, x, w=1.0):
        x = float(x)
        if not self._init:
            self.m = x
            self._init = True
        else:
            self.m = (1 - self.alpha) * self.m + self.alpha * x

    def value(self):
        return self.m

    def reset(self):
        self._init = False
        self.m = 0.0


METER_FACTORY: dict[str, Callable[[], Meter]] = {
    "mean": MeanMeter,
    "sum": SumMeter,
    "last": LastMeter,
    "ema": lambda: EMAMeter(alpha=0.2),
}


# ---------- loggers ----------
class Logger(ABC):
    @abstractmethod
    def on_step(self, split: str, step: int, row: dict[str, float]): ...

    @abstractmethod
    def on_epoch(self, split: str, epoch: int, row: dict[str, float]): ...


class ConsoleLogger(Logger):
    def __init__(self, every_steps: int = 0, fmt: str | None = None):
        self.every_steps = every_steps
        self.fmt = fmt  # optional custom printf string with {key}

    def _format(self, split, idx, row, prefix):
        if self.fmt:
            return self.fmt.format(**row)
        # default: compact, sorted keys
        keys = sorted(row.keys())
        kv = " ".join(f"{k}={row[k]:.4f}" for k in keys)
        return f"[{prefix} {idx}] {split} {kv}"

    def on_step(self, split, step, row):
        if self.every_steps and (step % self.every_steps == 0):
            print(self._format(split, step, row, "Step"))

    def on_epoch(self, split, epoch, row):
        print(self._format(split, epoch, row, "Epoch"))


class CSVLogger(Logger):
    """Append-only CSV logger that tolerates schema drift.

    The model emits metric keys CONDITIONALLY (stage-1 vs stage-2, train vs
    val, λ-warmup vs steady-state).  A naive append-with-fixed-header writer
    misaligns columns when a new key appears or an existing key is omitted.
    To keep downstream ``csv.DictReader`` consumers correct, this writer
    maintains a superset header: whenever a row introduces a new key, the
    file is rewritten in place with the expanded header, padding prior rows
    with the empty string.  Rows missing a known key likewise get an empty
    cell.  The rewrite only happens on first-seen keys (rare — stage
    transitions and λ-warmup edges), so steady-state cost is one append.

    Both ``on_step`` (per-step, e.g. train) and ``on_epoch`` (per-epoch,
    e.g. val) rows are written so the file carries every split a run logs.
    The second column is always ``step`` — for ``on_epoch`` callers we
    treat the epoch index as the row's step.
    """

    # First two columns are always split + the step/epoch index name.
    _IDX_COL = "step"

    def __init__(self, path: str):
        self.path = path
        self._known_keys: list[str] = []  # first-seen order
        self._known_set: set[str] = set()
        # If a file already exists, prime _known_keys from its header so we
        # append compatibly on resumed runs.
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                if header and len(header) >= 2:
                    # Header is: ["split", idx_name, *metric_keys].
                    self._IDX_COL = header[1]
                    for k in header[2:]:
                        if k not in self._known_set:
                            self._known_set.add(k)
                            self._known_keys.append(k)
            except (OSError, StopIteration):
                pass

    def _rewrite_with_expanded_header(self, new_keys: list[str]) -> None:
        """Rewrite the file with ``new_keys`` appended to the metric columns.

        Existing rows are preserved unchanged (the new columns are simply
        empty for them).  Called only when previously-unseen keys appear.
        """
        old_rows: list[list[str]] = []
        old_header: list[str] = []
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, newline="") as f:
                reader = csv.reader(f)
                old_header = next(reader, []) or []
                old_rows = list(reader)
        # Build expanded header. Preserve the leading two columns
        # (split, idx_name) from the old header when present.
        leading = old_header[:2] if len(old_header) >= 2 else ["split", self._IDX_COL]
        expanded_header = leading + list(self._known_keys)
        # Write to a sibling temp file then atomically replace the target so
        # a crash mid-rewrite never corrupts the existing metrics.csv.
        dirpath = os.path.dirname(self.path) or "."
        fd, tmppath = tempfile.mkstemp(prefix="_csv_rewrite_", dir=dirpath)
        try:
            with os.fdopen(fd, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(expanded_header)
                # Old rows already match leading + previously-known keys; pad
                # with empty strings for each newly-added key.
                pad = [""] * len(new_keys)
                for row in old_rows:
                    writer.writerow(row + pad)
            os.replace(tmppath, self.path)
        except Exception:
            try:
                os.remove(tmppath)
            except OSError:
                pass
            raise

    def _write(self, split: str, idx_name: str, idx: int, row: dict[str, float]):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        # Detect new keys (preserve first-seen order from ``row``).
        new_keys = [k for k in row if k not in self._known_set]
        if new_keys:
            for k in new_keys:
                self._known_set.add(k)
                self._known_keys.append(k)
            # If the file already had a header (or rows), rewrite with the
            # expanded header so column positions stay consistent.  If the
            # file doesn't exist yet, the upcoming append will create it
            # with the full header below.
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                self._rewrite_with_expanded_header(new_keys)

        file_existed = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_existed:
                self._IDX_COL = idx_name
                writer.writerow(["split", idx_name] + list(self._known_keys))
            # Pad missing keys with empty string so column alignment matches
            # the header.  Downstream ``csv.DictReader`` consumers see the
            # missing values as empty strings (treat as absent / NaN).
            writer.writerow(
                [split, idx]
                + [("" if k not in row else row[k]) for k in self._known_keys]
            )

    def on_step(self, split, step, row):
        self._write(split, "step", step, row)

    def on_epoch(self, split, epoch, row):
        # Per-epoch rows (val) share the "step" column with on_step rows.
        self._write(split, "step", epoch, row)


# ---------- spec & store ----------
@dataclass
class MetricSpec:
    pattern: str  # glob, e.g. "loss/*"
    kind: str = "mean"  # mean | sum | last | ema
    fmt: str = ".4f"  # for future pretty printers


class SplitStore:
    """Per-split meter set with a deferred-sync tensor path.

    ``add`` accepts either a Python float or a CUDA/CPU tensor. Tensor
    values are queued in ``_pending`` and batch-synced to floats in a
    single ``torch.stack(...).cpu().tolist()`` at ``values()`` time (or
    when the caller invokes :meth:`_flush_pending` directly). This
    avoids the per-tensor ``.item()`` host sync that used to fire once
    per metric per step (~50 syncs/step in ``_log_train_step``) and
    serialized CPU dispatch against the GPU.
    """

    def __init__(self, spec: list[MetricSpec]):
        self.spec = spec
        self.meters: dict[str, Meter] = {}
        self.order: list[str] = []  # first-seen order
        # Queued tensor updates awaiting batched host sync.
        self._pending: list[tuple[str, torch.Tensor, float]] = []

    def _make_meter(self, key: str) -> Meter:
        # find first matching spec else default mean
        for s in self.spec:
            if fnmatch.fnmatch(key, s.pattern):
                return METER_FACTORY.get(s.kind, MeanMeter)()
        return MeanMeter()

    def add(self, key: str, val: Any, w: float = 1.0):
        if key not in self.meters:
            self.meters[key] = self._make_meter(key)
            self.order.append(key)
        if torch.is_tensor(val):
            self._pending.append((key, val, float(w)))
        else:
            self.meters[key].add(float(val), float(w))

    def _flush_pending(
        self, on_finite_report: Callable[[str, float], None] | None = None
    ) -> None:
        """Batch-sync any queued tensor updates into their meters.

        Groups queued tensors by device so all CUDA tensors go through a
        single ``torch.stack(...).cpu().tolist()`` (one host sync per
        device instead of one per tensor), and already-CPU tensors are
        materialised without any device round-trip.
        """
        if not self._pending:
            return
        by_device: dict[torch.device, list[int]] = {}
        for i, (_, t, _) in enumerate(self._pending):
            by_device.setdefault(t.device, []).append(i)
        floats: list[float | None] = [None] * len(self._pending)
        for device, indices in by_device.items():
            batch = torch.stack([
                self._pending[i][1].detach().reshape(()) for i in indices
            ])
            if device.type != "cpu":
                batch = batch.cpu()
            batch_floats = batch.tolist()
            for i, f in zip(indices, batch_floats):
                floats[i] = f
        for (key, _tensor, weight), f in zip(self._pending, floats):
            if on_finite_report is not None:
                on_finite_report(key, f)
            self.meters[key].add(f, weight)
        self._pending.clear()

    def values(self) -> dict[str, float]:
        self._flush_pending()
        return {k: self.meters[k].value() for k in self.order}

    def reset(self):
        self._pending.clear()
        for m in self.meters.values():
            m.reset()


class MetricStore:
    """Per-split metric aggregator that fans flushed rows out to loggers.

    Accumulates metrics into per-split meters (kind chosen by glob from
    ``spec`` / ``split_spec``), then flushes mean/last/sum values to every
    attached :class:`Logger` on ``step_end`` / ``epoch_end``.

    Usage::

        metrics = MetricStore(
            spec=[
                MetricSpec(
                    "loss/*", "mean"
                ),
                MetricSpec(
                    "time/*", "sum"
                ),
            ],
            loggers=[
                ConsoleLogger(
                    every_steps=50
                ),
                CSVLogger(
                    "metrics.csv"
                ),
            ],
        )
        # step loop:
        metrics.update(
            "train",
            {
                "loss/total": loss,
                "loss/recon": Lrec,
            },
            weights={
                "loss/recon": obs
            },
        )
        metrics.step_end(
            "train", global_step
        )
        # epoch end:
        metrics.epoch_end(
            "train", epoch
        )  # -> dict (averaged)
    """

    def __init__(
        self,
        spec: list[MetricSpec] | None = None,
        loggers: list[Logger] | None = None,
        split_spec: dict[str, list[MetricSpec]] | None = None,
    ):
        self.spec = spec or [MetricSpec("loss/*", "mean")]
        # Per-split spec overrides. Validation, for instance, accumulates
        # over the whole val set within one ``epoch_end`` and wants mean
        # meters (weighted by batch size), whereas train samples the last
        # step before each flush.
        self.split_spec = split_spec or {}
        self.loggers = loggers or [ConsoleLogger()]
        self.splits: dict[str, SplitStore] = {}
        self._t0 = time.time()
        # Running count of non-finite (NaN/Inf) metric values seen, surfaced as
        # the ``nonfinite/total`` column so a diverged run is visible in the CSV
        # instead of silently persisting ``nan``. ``_nonfinite_warned`` dedups
        # the per-key warning.
        self._nonfinite_total: int = 0
        self._nonfinite_warned: set[str] = set()

    def _split(self, split: str) -> SplitStore:
        if split not in self.splits:
            self.splits[split] = SplitStore(self.split_spec.get(split, self.spec))
        return self.splits[split]

    @staticmethod
    def _tofloat(x: Any) -> float:
        """Legacy helper (still used for scalar Python inputs).

        Tensor inputs are now routed through :meth:`SplitStore.add`'s
        deferred-sync path — do NOT call this on tensors from the hot
        loop or you will re-introduce a per-metric host sync.
        """
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                x = x.mean()
            return float(x.detach().item())
        return float(x)

    def _mean_scalar(self, x: Any) -> Any:
        """Return a 0-d tensor when x is a multi-element tensor, else x."""
        if isinstance(x, torch.Tensor) and x.numel() != 1:
            return x.mean()
        return x

    def update(
        self,
        split: str,
        values: dict[str, Any],
        weight: float = 1.0,
        weights: dict[str, float] | None = None,
    ):
        """values: dict of metric -> scalar (tensor ok).
        weight: default weight (e.g., batch size)
        weights: per-metric override (e.g., observed elements for recon)

        Tensor values are queued for a single batched host sync at flush
        time (``SplitStore.values()``) — see :class:`SplitStore` for the
        rationale. The non-finite check + warning also runs at flush.
        """
        split_name = split
        ss = self._split(split)

        # Reused across the loop; captured only when we actually emit a
        # warning inside the flush callback.
        def _report_finite(key: str, value: float) -> None:
            if not math.isfinite(value):
                self._nonfinite_total += 1
                if key not in self._nonfinite_warned:
                    self._nonfinite_warned.add(key)
                    log.warning(
                        "Non-finite metric %r=%s in split %r; recorded but "
                        "surfaced via nonfinite/total.",
                        key, value, split_name,
                    )
        # Stash the reporter for the next flush; unified across update calls
        # so warnings still fire even if the caller triggers multiple
        # updates between flushes.
        self._finite_reporter_for_split = _report_finite

        for k, v in values.items():
            w = weights.get(k, weight) if weights else weight
            if isinstance(v, torch.Tensor):
                ss.add(k, self._mean_scalar(v), float(w))
            else:
                f = self._tofloat(v)
                _report_finite(k, f)
                ss.add(k, f, float(w))

    def _flush(self, split: str) -> None:
        """Force any queued tensor updates in ``split`` to sync + accumulate."""
        ss = self._split(split)
        ss._flush_pending(
            on_finite_report=getattr(self, "_finite_reporter_for_split", None)
        )

    def _with_health(self, row: dict[str, float]) -> dict[str, float]:
        """Attach the run-health counter to a flushed row."""
        row = dict(row)
        row["nonfinite/total"] = float(self._nonfinite_total)
        return row

    def step_end(self, split: str, step: int, also_log: bool = True):
        # Flush first (with the nonfinite reporter) so the values() call
        # doesn't do an untracked flush that skips warnings.
        self._flush(split)
        row = self._with_health(self._split(split).values())
        if also_log:
            for lg in self.loggers:
                lg.on_step(split, step, row)
        return row

    def epoch_end(self, split: str, epoch: int, reset: bool = True):
        self._flush(split)
        row = self._with_health(self._split(split).values())
        for lg in self.loggers:
            lg.on_epoch(split, epoch, row)
        if reset:
            self._split(split).reset()
        return row

    def close(self):
        for lg in self.loggers:
            if hasattr(lg, "close"):
                lg.close()


class TensorBoardLogger(Logger):
    """SummaryWriter-backed logger; a no-op if tensorboard isn't installed."""

    def __init__(self, log_dir: str = "runs/ddssm", flush_secs: int = 10):
        self._active = False
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print(
                "[TensorBoardLogger] tensorboard not installed — logging disabled. "
                "Install with `pip install tensorboard` to enable."
            )
            return
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self._active = True

    def on_step(self, split: str, step: int, row: dict[str, float]):
        if not self._active:
            return
        # log each metric as <split>/<name>, e.g. train/loss/total
        for k, v in row.items():
            self.writer.add_scalar(f"{split}/{k}", v, step)

    def on_epoch(self, split: str, epoch: int, row: dict[str, float]):
        if not self._active:
            return
        for k, v in row.items():
            self.writer.add_scalar(f"{split}_epoch/{k}", v, epoch)
        self.writer.flush()

    def close(self):
        if self._active:
            self.writer.close()


class WandbLogger(Logger):
    """Weights & Biases logger.

    Logs step metrics via ``wandb.log`` and epoch metrics with an
    ``"epoch/"`` prefix.  All metric keys are namespaced by split so they
    appear as ``train/loss/total``, ``val/loss/total``, etc. in the W&B UI.

    The logger is *soft-optional*: if ``wandb`` is not installed or
    ``enabled=False`` it silently becomes a no-op so that the rest of the
    training code need not be guarded.

    Args:
        project: W&B project name.
        entity: W&B entity (user or team).  ``None`` uses the default entity.
        name: Display name for the run.  ``None`` lets W&B auto-generate one.
        group: Optional group name; useful for collecting all trials of a
            sweep under a single grouping in the W&B UI.
        tags: List of string tags attached to the run.
        config: Arbitrary dict of hyperparameters to store with the run.
        base_url: URL of a self-hosted W&B server, e.g.
            ``"https://wandb.example.com"``.  When set it overrides the
            ``WANDB_BASE_URL`` environment variable for this process.
        run_dir: Directory where W&B run artefacts (and ``.wandb_run_id``)
            land. When set, the constructor persists the active run-id to
            ``<run_dir>/.wandb_run_id`` so post-training stages (eval,
            viz, variance) can ``wandb.init(resume="allow", id=...)``
            against the same run.
        watch_log: Forwarded to ``wandb.watch`` when not ``None``. One of
            ``"gradients"`` / ``"parameters"`` / ``"all"`` — see the W&B
            docs. The model to watch is supplied later via
            :meth:`watch_model` (the trainer calls it after building the
            logger).
        watch_log_freq: ``log_freq`` passed to ``wandb.watch``. Ignored
            unless ``watch_log`` is set.
        enabled: If ``False`` the logger is a no-op regardless of whether
            ``wandb`` is installed (useful for quick local runs where you
            don't want any W&B traffic).
    """

    # Filename used to persist the run-id under ``run_dir`` so post-training
    # stages can resume into the same W&B run (cross-stage reconnect).
    _RUN_ID_FILENAME = ".wandb_run_id"

    def __init__(
        self,
        project: str = "ddssm",
        entity: str | None = None,
        name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        base_url: str | None = None,
        run_dir: str | None = None,
        watch_log: str | None = None,
        watch_log_freq: int = 100,
        enabled: bool = True,
    ):
        self._active = False
        self._run_dir = run_dir
        self._watch_log = watch_log
        self._watch_log_freq = int(watch_log_freq)
        if not enabled:
            return

        try:
            import wandb
        except ImportError:
            print(
                "[WandbLogger] wandb not installed — logging disabled. "
                "Run `pip install wandb` to enable."
            )
            return

        if base_url:
            os.environ["WANDB_BASE_URL"] = base_url

        init_kwargs: dict[str, Any] = dict(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=tags or [],
            config=config or {},
            reinit="finish_previous",
        )
        if run_dir:
            init_kwargs["dir"] = run_dir
        wandb.init(**init_kwargs)

        # Each namespace gets its own monotonic step axis so train and
        # epoch logs don't fight over W&B's single per-run step counter.
        # ``train_step`` is the trainer's ``global_step``; ``epoch`` is
        # whatever counter the trainer passes into ``on_epoch``.
        wandb.define_metric("train_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("val/*", step_metric="train_step")
        wandb.define_metric("epoch/*", step_metric="epoch")

        self._wandb = wandb
        self._active = True

        # Persist the active run-id so post-training stages can resume
        # into this same W&B run (no env var, no Optuna user_attr — the
        # run-dir is the only handle every standalone stage already has).
        if run_dir:
            self._persist_run_id(run_dir)

    def _persist_run_id(self, run_dir: str) -> None:
        run = getattr(self._wandb, "run", None)
        run_id = getattr(run, "id", None) if run is not None else None
        if not run_id:
            return
        try:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, self._RUN_ID_FILENAME), "w") as f:
                f.write(str(run_id))
        except OSError as e:
            print(f"[WandbLogger] Could not persist run-id: {e}")

    def watch_model(self, model: Any) -> None:
        """Call ``wandb.watch`` on ``model`` if watch_log was configured.

        Trainer calls this after construction so the W&B run watches the
        live module (gradients / parameters per ``watch_log``).
        """
        if not self._active or self._watch_log is None or model is None:
            return
        try:
            self._wandb.watch(
                model,
                log=self._watch_log,
                log_freq=self._watch_log_freq,
            )
        except Exception as e:
            print(f"[WandbLogger] wandb.watch failed: {e}")

    def _log(
        self, prefix: str, step_key: str, step: int, row: dict[str, float]
    ) -> None:
        if not self._active:
            return
        payload: dict[str, Any] = {f"{prefix}/{k}": v for k, v in row.items()}
        # Embed the step into the payload; let W&B's per-metric step axis
        # ordering handle monotonicity. Do not pass ``step=`` -- that
        # collides between train/val/epoch namespaces.
        payload[step_key] = int(step)
        self._wandb.log(payload)

    def on_step(self, split: str, step: int, row: dict[str, float]) -> None:
        self._log(split, "train_step", step, row)

    def on_epoch(self, split: str, epoch: int, row: dict[str, float]) -> None:
        self._log(f"epoch/{split}", "epoch", epoch, row)

    def _upload_artifacts(self) -> None:
        """Upload final checkpoint + resolved config as W&B artifacts.

        Best-effort: any failure (missing file, network hiccup, partial
        W&B install) is swallowed with a warning — close() must always
        finish so the trainer's ``finally:`` cleanup doesn't mask the
        underlying training error.
        """
        if not self._active or not self._run_dir:
            return
        artifact_targets = [
            (
                "model",
                "pth",
                os.path.join(self._run_dir, "checkpoints", "ckpt_latest.pth"),
            ),
            ("config", "yaml", os.path.join(self._run_dir, "resolved_config.yaml")),
        ]
        run = getattr(self._wandb, "run", None)
        run_id = getattr(run, "id", "") if run is not None else ""
        for kind, ext, path in artifact_targets:
            if not os.path.exists(path):
                continue
            try:
                # Name artifacts off the run-id so reruns don't collide
                # under one canonical name (W&B versions them anyway, but
                # the human-readable identity stays unique per run).
                artifact = self._wandb.Artifact(
                    name=f"{kind}-{run_id or 'run'}",
                    type=kind,
                )
                artifact.add_file(path)
                self._wandb.log_artifact(artifact)
            except Exception as e:
                print(f"[WandbLogger] Failed to upload {kind} artifact ({path}): {e}")

    def close(self) -> None:
        if self._active:
            self._upload_artifacts()
            self._wandb.finish()
            self._active = False


def resume_run_from_dir(
    run_dir: str,
    wandb_config: dict[str, Any] | None,
) -> Any:
    """Re-open the train run's W&B session from ``<run_dir>/.wandb_run_id``.

    Returns the live ``wandb`` module on success (with a resumed run
    attached) or ``None`` if W&B is disabled, the package is missing,
    the run-id file isn't present, or anything else trips the soft path.
    Post-training stages (viz / eval / variance) use this so their PNGs
    and metrics land on the same W&B run produced by training.
    """
    if not wandb_config or not bool(wandb_config.get("enabled", True)):
        return None
    id_path = os.path.join(run_dir, WandbLogger._RUN_ID_FILENAME)
    if not os.path.isfile(id_path):
        return None
    try:
        with open(id_path) as f:
            run_id = f.read().strip()
    except OSError:
        return None
    if not run_id:
        return None
    try:
        import wandb
    except ImportError:
        return None
    base_url = wandb_config.get("base_url")
    if base_url:
        os.environ["WANDB_BASE_URL"] = base_url
    try:
        wandb.init(
            project=wandb_config.get("project", "ddssm"),
            entity=wandb_config.get("entity"),
            id=run_id,
            resume="allow",
            dir=run_dir,
        )
    except Exception as e:
        print(f"[WandbLogger] resume failed: {e}")
        return None
    return wandb
