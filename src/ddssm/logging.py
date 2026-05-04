"""CSV and TensorBoard logging utilities for tracking training metrics."""

from __future__ import annotations

import os
import csv
import time
from typing import Any, Dict, Callable, Optional
import fnmatch
from dataclasses import dataclass

import torch
from torch.utils.tensorboard import SummaryWriter


# ---------- meters ----------
class Meter:
    def add(self, x: float, w: float = 1.0): ...
    def value(self) -> float: ...
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
class Logger:
    def on_step(self, split: str, step: int, row: Dict[str, float]): ...
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
        # self._write(split, "epoch", epoch, row)
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
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)

    def on_step(self, split: str, step: int, row: Dict[str, float]):
        # log each metric as <split>/<name>, e.g. train/loss/total
        for k, v in row.items():
            self.writer.add_scalar(f"{split}/{k}", v, step)

    def on_epoch(self, split: str, epoch: int, row: Dict[str, float]):
        for k, v in row.items():
            self.writer.add_scalar(f"{split}_epoch/{k}", v, epoch)
        self.writer.flush()

    def close(self):
        self.writer.close()
