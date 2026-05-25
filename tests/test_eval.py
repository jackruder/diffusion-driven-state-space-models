"""Tests for the eval stage: metric registry, CSV-derived metrics, runner."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
import pytest

from ddssm.eval import METRIC_REGISTRY, EvalSpec, EvalContext, evaluate
from ddssm.eval.metrics import eval_loss_tail, eval_energy_score


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_registry_has_core_metrics():
    for name in ("mae", "crps_sum", "recon_mse", "loss_tail", "energy_score"):
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


def test_evalspec_per_metric_kwargs_forwarded(tmp_path):
    """EvalSpec.kwargs[name] must be forwarded as **kwargs to that metric.

    Uses ``loss_tail`` because it accepts a ``column`` kwarg we can verify
    indirectly: pointing it at a column that exists vs one that doesn't
    changes the resulting key in the output.
    """
    csv_path = tmp_path / "m.csv"
    rows = [{"split": "train", "step": str(i), "metric_a": str(0.7)} for i in range(10)]
    _write_csv(csv_path, rows)

    class _StubData:
        batch_transform = staticmethod(lambda b, d: b)
        metadata = type("_M", (), {"forecast_split": None})()

        def train_loader(self): return None
        def val_loader(self): return None
        def test_loader(self): return None

    class _StubModel(torch.nn.Module):
        def __init__(self): super().__init__(); self.lin = torch.nn.Linear(1, 1)
        def to(self, d): return self

    class _StubExpt:
        def __init__(self):
            self.data = _StubData()
            self.model = _StubModel()

    spec = EvalSpec(
        metrics=["loss_tail"], split="val",
        kwargs={"loss_tail": {"column": "metric_a"}},
        output_filename="m.json",
    )
    out = evaluate(_StubExpt(), spec, device=torch.device("cpu"),
                   run_dir=str(tmp_path), checkpoint_path=None,
                   csv_path=str(csv_path))
    assert "metric_a_tail" in out  # not "loss_total_tail"
    assert abs(out["metric_a_tail"] - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# energy_score unit tests.
#
# We mock _iter_forecast_batches so these tests run without a real model or
# data loader. The mock returns a single (pred_samples, pred_mean, y_future)
# tuple with analytically known values so we can verify the formula exactly.
# ---------------------------------------------------------------------------

_MOCK_PATH = "ddssm.eval.metrics._iter_forecast_batches"


def test_energy_score_in_registry():
    assert "energy_score" in METRIC_REGISTRY


def test_energy_score_zero_for_perfect_prediction():
    """ES = 0 when every sample equals the truth: E[||X-y||]=0, E[||X-X'||]=0."""
    B, S, D, L2 = 2, 4, 1, 3
    y = torch.randn(B, D, L2)
    pred_samples = y.unsqueeze(1).expand(B, S, D, L2).contiguous()
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=4, num_samples=S)
    with patch(_MOCK_PATH, return_value=[(pred_samples, None, y)]):
        result = eval_energy_score(ctx)
    assert abs(result["energy_score"]) < 1e-5


def test_energy_score_known_value_point_mass():
    """S identical samples at +1, truth at 0 → E[||X-y||]=1, E[||X-X'||]=0 → ES=1."""
    B, S, D, L2 = 1, 4, 1, 1
    pred_samples = torch.ones(B, S, D, L2)
    y = torch.zeros(B, D, L2)
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=1, num_samples=S)
    with patch(_MOCK_PATH, return_value=[(pred_samples, None, y)]):
        result = eval_energy_score(ctx)
    assert abs(result["energy_score"] - 1.0) < 1e-5


def test_energy_score_nan_for_empty_loader():
    """Empty iterator must produce NaN, not raise."""
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=4)
    with patch(_MOCK_PATH, return_value=[]):
        result = eval_energy_score(ctx)
    assert math.isnan(result["energy_score"])


def test_energy_score_diverse_samples_lower_than_collapsed():
    """Diverse samples near the truth must score lower than a collapsed mass far away.

    This is the key proper-scoring property motivating the use of energy score for
    multimodal evaluation (bimodal / robot presets): a collapsed Gaussian prediction
    will be penalised more than a spread that covers both modes.
    """
    B, S, D, L2 = 1, 10, 1, 1
    y = torch.zeros(B, D, L2)
    # Collapsed far from truth (ES ≈ 3.0, since all samples at 3.0, no diversity discount)
    collapsed = torch.full((B, S, D, L2), 3.0)
    # Diverse near truth (half at -0.5, half at +0.5)
    half = S // 2
    diverse = torch.cat([
        torch.full((B, half, D, L2), -0.5),
        torch.full((B, half, D, L2), 0.5),
    ], dim=1)
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=1, num_samples=S)
    with patch(_MOCK_PATH, return_value=[(collapsed, None, y)]):
        es_collapsed = eval_energy_score(ctx)["energy_score"]
    with patch(_MOCK_PATH, return_value=[(diverse, None, y)]):
        es_diverse = eval_energy_score(ctx)["energy_score"]
    assert es_diverse < es_collapsed


def test_energy_score_accumulates_across_batches():
    """Result is the mean of per-batch scores, not just the last batch."""
    B, S, D, L2 = 1, 4, 1, 1
    batch_a = (torch.ones(B, S, D, L2), None, torch.zeros(B, D, L2))   # ES ≈ 1.0
    batch_b = (torch.full((B, S, D, L2), 2.0), None, torch.zeros(B, D, L2))  # ES ≈ 2.0
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=1, num_samples=S)
    with patch(_MOCK_PATH, return_value=[batch_a, batch_b]):
        result = eval_energy_score(ctx)
    assert abs(result["energy_score"] - 1.5) < 1e-4


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
