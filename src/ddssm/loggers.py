"""CSV, TensorBoard, and Weights & Biases logging utilities for tracking training metrics."""

from __future__ import annotations

import os
import csv
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Optional
import fnmatch
from dataclasses import dataclass

import torch


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


METER_FACTORY: Dict[str, Callable[[], Meter]] = {
    "mean": MeanMeter,
    "sum": SumMeter,
    "last": LastMeter,
    "ema": lambda: EMAMeter(alpha=0.2),
}


# ---------- loggers ----------
class Logger(ABC):
    @abstractmethod
    def on_step(self, split: str, step: int, row: Dict[str, float]): ...

    @abstractmethod
    def on_epoch(self, split: str, epoch: int, row: Dict[str, float]): ...


class ConsoleLogger(Logger):
    def __init__(self, every_steps: int = 0, fmt: Optional[str] = None):
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
    def __init__(self, path: str):
        self.path = path
        self._header_written = os.path.exists(path) and os.path.getsize(path) > 0

    def _write(self, split: str, idx_name: str, idx: int, row: Dict[str, float]):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._header_written:
                writer.writerow(["split", idx_name] + list(row.keys()))
                self._header_written = True
            writer.writerow([split, idx] + [row[k] for k in row])

    def on_step(self, split, step, row):
        self._write(split, "step", step, row)

    def on_epoch(self, split, epoch, row):
        # Intentionally no-op: this logger currently captures per-step metrics only.
        pass


# ---------- spec & store ----------
@dataclass
class MetricSpec:
    pattern: str  # glob, e.g. "loss/*"
    kind: str = "mean"  # mean | sum | last | ema
    fmt: str = ".4f"  # for future pretty printers


class SplitStore:
    def __init__(self, spec: list[MetricSpec]):
        self.spec = spec
        self.meters: Dict[str, Meter] = {}
        self.order: list[str] = []  # first-seen order

    def _make_meter(self, key: str) -> Meter:
        # find first matching spec else default mean
        for s in self.spec:
            if fnmatch.fnmatch(key, s.pattern):
                return METER_FACTORY.get(s.kind, MeanMeter)()
        return MeanMeter()

    def add(self, key: str, val: float, w: float = 1.0):
        if key not in self.meters:
            self.meters[key] = self._make_meter(key)
            self.order.append(key)
        self.meters[key].add(val, w)

    def values(self) -> Dict[str, float]:
        return {k: self.meters[k].value() for k in self.order}

    def reset(self):
        for m in self.meters.values():
            m.reset()


class MetricStore:
    """Usage:
    metrics = MetricStore(
       spec=[MetricSpec("loss/*","mean"), MetricSpec("time/*","sum")],
       loggers=[ConsoleLogger(every_steps=50), CSVLogger("metrics.csv")]
    )
    # step loop:
    metrics.update("train", {"loss/total": loss, "loss/recon": Lrec}, weights={"loss/recon": obs})
    metrics.step_end("train", global_step)
    # epoch end:
    metrics.epoch_end("train", epoch) -> dict (averaged)
    """

    def __init__(
        self,
        spec: Optional[list[MetricSpec]] = None,
        loggers: Optional[list[Logger]] = None,
    ):
        self.spec = spec or [MetricSpec("loss/*", "mean")]
        self.loggers = loggers or [ConsoleLogger()]
        self.splits: Dict[str, SplitStore] = {}
        self._t0 = time.time()

    def _split(self, split: str) -> SplitStore:
        if split not in self.splits:
            self.splits[split] = SplitStore(self.spec)
        return self.splits[split]

    @staticmethod
    def _tofloat(x: Any) -> float:
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                x = x.mean()
            return float(x.detach().item())
        return float(x)

    def update(
        self,
        split: str,
        values: Dict[str, Any],
        weight: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
    ):
        """values: dict of metric -> scalar (tensor ok).
        weight: default weight (e.g., batch size)
        weights: per-metric override (e.g., observed elements for recon)
        """
        ss = self._split(split)
        for k, v in values.items():
            w = weights.get(k, weight) if weights else weight
            ss.add(k, self._tofloat(v), float(w))

    def step_end(self, split: str, step: int, also_log: bool = True):
        row = self._split(split).values()
        # include wall time (seconds per step EMA via meter? keep simple: last delta)
        row = dict(row)  # copy
        if also_log:
            for lg in self.loggers:
                lg.on_step(split, step, row)
        return row

    def epoch_end(self, split: str, epoch: int, reset: bool = True):
        row = self._split(split).values()
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
    def __init__(self, log_dir: str = "runs/ddssm", flush_secs: int = 10):
        self._active = False
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
        except ImportError:
            print(
                "[TensorBoardLogger] tensorboard not installed — logging disabled. "
                "Install with `pip install tensorboard` to enable."
            )
            return
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self._active = True

    def on_step(self, split: str, step: int, row: Dict[str, float]):
        if not self._active:
            return
        # log each metric as <split>/<name>, e.g. train/loss/total
        for k, v in row.items():
            self.writer.add_scalar(f"{split}/{k}", v, step)

    def on_epoch(self, split: str, epoch: int, row: Dict[str, float]):
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
        entity: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[list[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        run_dir: Optional[str] = None,
        watch_log: Optional[str] = None,
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
            import wandb  # noqa: PLC0415
        except ImportError:
            print(
                "[WandbLogger] wandb not installed — logging disabled. "
                "Run `pip install wandb` to enable."
            )
            return

        if base_url:
            os.environ["WANDB_BASE_URL"] = base_url

        init_kwargs: Dict[str, Any] = dict(
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
                model, log=self._watch_log, log_freq=self._watch_log_freq,
            )
        except Exception as e:  # noqa: BLE001 — best-effort; never break training
            print(f"[WandbLogger] wandb.watch failed: {e}")

    def _log(self, prefix: str, step_key: str, step: int, row: Dict[str, float]) -> None:
        if not self._active:
            return
        payload: Dict[str, Any] = {f"{prefix}/{k}": v for k, v in row.items()}
        # Embed the step into the payload; let W&B's per-metric step axis
        # ordering handle monotonicity. Do not pass ``step=`` -- that
        # collides between train/val/epoch namespaces.
        payload[step_key] = int(step)
        self._wandb.log(payload)

    def on_step(self, split: str, step: int, row: Dict[str, float]) -> None:
        self._log(split, "train_step", step, row)

    def on_epoch(self, split: str, epoch: int, row: Dict[str, float]) -> None:
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
            ("model", "pth",
             os.path.join(self._run_dir, "checkpoints", "ckpt_latest.pth")),
            ("config", "yaml",
             os.path.join(self._run_dir, "resolved_config.yaml")),
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
                    name=f"{kind}-{run_id or 'run'}", type=kind,
                )
                artifact.add_file(path)
                self._wandb.log_artifact(artifact)
            except Exception as e:  # noqa: BLE001 — best-effort
                print(
                    f"[WandbLogger] Failed to upload {kind} artifact "
                    f"({path}): {e}"
                )

    def close(self) -> None:
        if self._active:
            self._upload_artifacts()
            self._wandb.finish()
            self._active = False


def resume_run_from_dir(
    run_dir: str, wandb_config: Optional[Dict[str, Any]],
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
        import wandb  # noqa: PLC0415
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
    except Exception as e:  # noqa: BLE001 — best-effort
        print(f"[WandbLogger] resume failed: {e}")
        return None
    return wandb
