"""Tests for the eval stage: metric registry, CSV-derived metrics, runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from ddssm.eval import EvalContext, EvalSpec, METRIC_REGISTRY, evaluate
from ddssm.eval.metrics import eval_loss_tail


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_registry_has_core_metrics():
    for name in ("mae", "crps_sum", "recon_mse", "loss_tail"):
        assert name in METRIC_REGISTRY


def test_loss_tail_reads_csv(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    rows = [{"split": "train", "step": str(i), "loss/total": str(1.0 - i * 0.01)} for i in range(100)]
    _write_csv(csv_path, rows)
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), csv_path=str(csv_path))
    out = eval_loss_tail(ctx)
    # Tail mean of the last 10% of [1.0, 0.99, ..., 0.01] is mean of last 10 values.
    assert "loss_total_tail" in out
    expected = sum(1.0 - i * 0.01 for i in range(90, 100)) / 10
    assert abs(out["loss_total_tail"] - expected) < 1e-6


def test_loss_tail_returns_nan_for_missing_csv(tmp_path):
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), csv_path=str(tmp_path / "nope.csv"))
    out = eval_loss_tail(ctx)
    val = list(out.values())[0]
    assert val != val  # NaN check


def test_evaluate_runner_writes_metrics_json(tmp_path):
    """Smoke test: the runner accepts an experiment-shaped object and writes JSON."""
    csv_path = tmp_path / "train_metrics.csv"
    rows = [{"split": "train", "step": str(i), "loss/total": str(0.5)} for i in range(20)]
    _write_csv(csv_path, rows)

    class _StubData:
        batch_transform = staticmethod(lambda b, d: b)
        metadata = type("_M", (), {"forecast_split": None})()

        def train_loader(self): return None
        def val_loader(self): return None
        def test_loader(self): return None

    class _StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def to(self, device):
            return self

    class _StubExpt:
        def __init__(self):
            self.data = _StubData()
            self.model = _StubModel()

    spec = EvalSpec(metrics=["loss_tail"], split="val", output_filename="m.json")
    out = evaluate(
        _StubExpt(), spec,
        device=torch.device("cpu"),
        run_dir=str(tmp_path),
        checkpoint_path=None,
        csv_path=str(csv_path),
    )
    assert "loss_total_tail" in out
    written = json.loads((tmp_path / "m.json").read_text())
    assert written == out


def test_unknown_metric_raises():
    spec = EvalSpec(metrics=["nope"], split="val")

    class _Stub:
        class data:
            batch_transform = staticmethod(lambda b, d: b)
            metadata = type("_M", (), {"forecast_split": None})()

            @staticmethod
            def train_loader(): return None
            @staticmethod
            def val_loader(): return None
            @staticmethod
            def test_loader(): return None

        class model(torch.nn.Module):
            def __init__(self): super().__init__(); self.lin = torch.nn.Linear(1, 1)
            def to(self, d): return self

        model = model()

    with pytest.raises(KeyError):
        evaluate(_Stub(), spec, device=torch.device("cpu"), run_dir="/tmp/_nope")
