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
from ddssm.eval.metrics import (
    eval_mae,
    eval_rmse,
    eval_crps_sum,
    eval_loss_tail,
    eval_energy_score,
)
from ddssm.data.datamodule import DataMetadata, DDSSMDataModule


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
    rows = [
        {"split": "train", "step": str(i), "loss/total": str(1.0 - i * 0.01)}
        for i in range(100)
    ]
    _write_csv(csv_path, rows)
    ctx = EvalContext(
        model=None, loader=None, device=torch.device("cpu"), csv_path=str(csv_path)
    )
    out = eval_loss_tail(ctx)
    # Tail mean of the last 10% of [1.0, 0.99, ..., 0.01] is mean of last 10 values.
    assert "loss_total_tail" in out
    expected = sum(1.0 - i * 0.01 for i in range(90, 100)) / 10
    assert abs(out["loss_total_tail"] - expected) < 1e-6


def test_loss_tail_returns_nan_for_missing_csv(tmp_path):
    ctx = EvalContext(
        model=None,
        loader=None,
        device=torch.device("cpu"),
        csv_path=str(tmp_path / "nope.csv"),
    )
    out = eval_loss_tail(ctx)
    val = list(out.values())[0]
    assert val != val  # NaN check


def test_evaluate_runner_writes_metrics_json(tmp_path):
    """Smoke test: the runner accepts an experiment-shaped object and writes JSON."""
    csv_path = tmp_path / "train_metrics.csv"
    rows = [
        {"split": "train", "step": str(i), "loss/total": str(0.5)} for i in range(20)
    ]
    _write_csv(csv_path, rows)

    class _StubData(DDSSMDataModule):
        batch_transform = staticmethod(lambda b, d: b)
        metadata = DataMetadata(data_dim=1, forecast_split=None)

        def train_loader(self):
            return None

        def val_loader(self):
            return None

        def test_loader(self):
            return None

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
        _StubExpt(),
        spec,
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

    class _StubData(DDSSMDataModule):
        batch_transform = staticmethod(lambda b, d: b)
        metadata = DataMetadata(data_dim=1, forecast_split=None)

        def train_loader(self):
            return None

        def val_loader(self):
            return None

        def test_loader(self):
            return None

    class _StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)

        def to(self, d):
            return self

    class _StubExpt:
        def __init__(self):
            self.data = _StubData()
            self.model = _StubModel()

    spec = EvalSpec(
        metrics=["loss_tail"],
        split="val",
        kwargs={"loss_tail": {"column": "metric_a"}},
        output_filename="m.json",
    )
    out = evaluate(
        _StubExpt(),
        spec,
        device=torch.device("cpu"),
        run_dir=str(tmp_path),
        checkpoint_path=None,
        csv_path=str(csv_path),
    )
    assert "metric_a_tail" in out  # not "loss_total_tail"
    assert abs(out["metric_a_tail"] - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# energy_score unit tests.
#
# We mock _iter_forecast_batches so these tests run without a real model or
# data loader. The mock returns a single (pred_samples, pred_mean, y_future,
# y_mask) tuple with analytically known values so we can verify the formula
# exactly.
# ---------------------------------------------------------------------------

_MOCK_PATH = "ddssm.eval.metrics._iter_forecast_batches"


def _batch(pred_samples, pred_mean, y, mask=None):
    if mask is None:
        mask = torch.ones_like(y)
    return (pred_samples, pred_mean, y, mask)


def test_energy_score_in_registry():
    assert "energy_score" in METRIC_REGISTRY


def test_energy_score_zero_for_perfect_prediction():
    """ES = 0 when every sample equals the truth: E[||X-y||]=0, E[||X-X'||]=0."""
    B, S, D, L2 = 2, 4, 1, 3
    y = torch.randn(B, D, L2)
    pred_samples = y.unsqueeze(1).expand(B, S, D, L2).contiguous()
    ctx = EvalContext(
        model=None, loader=None, device=torch.device("cpu"), T_split=4, num_samples=S
    )
    with patch(_MOCK_PATH, return_value=[_batch(pred_samples, None, y)]):
        result = eval_energy_score(ctx)
    assert abs(result["energy_score"]) < 1e-5


def test_energy_score_known_value_point_mass():
    """S identical samples at +1, truth at 0 → E[||X-y||]=1, E[||X-X'||]=0 → ES=1."""
    B, S, D, L2 = 1, 4, 1, 1
    pred_samples = torch.ones(B, S, D, L2)
    y = torch.zeros(B, D, L2)
    ctx = EvalContext(
        model=None, loader=None, device=torch.device("cpu"), T_split=1, num_samples=S
    )
    with patch(_MOCK_PATH, return_value=[_batch(pred_samples, None, y)]):
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
    diverse = torch.cat(
        [
            torch.full((B, half, D, L2), -0.5),
            torch.full((B, half, D, L2), 0.5),
        ],
        dim=1,
    )
    ctx = EvalContext(
        model=None, loader=None, device=torch.device("cpu"), T_split=1, num_samples=S
    )
    with patch(_MOCK_PATH, return_value=[_batch(collapsed, None, y)]):
        es_collapsed = eval_energy_score(ctx)["energy_score"]
    with patch(_MOCK_PATH, return_value=[_batch(diverse, None, y)]):
        es_diverse = eval_energy_score(ctx)["energy_score"]
    assert es_diverse < es_collapsed


def test_energy_score_accumulates_across_batches():
    """Result is the mean of per-batch scores, not just the last batch."""
    B, S, D, L2 = 1, 4, 1, 1
    batch_a = _batch(torch.ones(B, S, D, L2), None, torch.zeros(B, D, L2))  # ES ≈ 1.0
    batch_b = _batch(
        torch.full((B, S, D, L2), 2.0), None, torch.zeros(B, D, L2)
    )  # ES ≈ 2.0
    ctx = EvalContext(
        model=None, loader=None, device=torch.device("cpu"), T_split=1, num_samples=S
    )
    with patch(_MOCK_PATH, return_value=[batch_a, batch_b]):
        result = eval_energy_score(ctx)
    assert abs(result["energy_score"] - 1.5) < 1e-4


# ---------------------------------------------------------------------------
# Observation-mask handling: forecast metrics must reduce over observed
# target entries only. Masked positions carry garbage values so any leak
# into the reduction shows up as a wrong (huge) result.
# ---------------------------------------------------------------------------


def _masked_fixture():
    torch.manual_seed(0)
    B, S, D, L2 = 2, 6, 3, 4
    y = torch.randn(B, D, L2)
    pred_mean = torch.randn(B, D, L2)
    pred_samples = torch.randn(B, S, D, L2)
    mask = (torch.rand(B, D, L2) > 0.4).float()
    y_bad = torch.where(mask.bool(), y, torch.full_like(y, 1e6))
    return pred_samples, pred_mean, y, y_bad, mask


def test_mae_ignores_masked_entries():
    pred_samples, pred_mean, y, y_bad, mask = _masked_fixture()
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=1)
    with patch(_MOCK_PATH, return_value=[(pred_samples, pred_mean, y_bad, mask)]):
        out = eval_mae(ctx)
    expected = (((pred_mean - y).abs() * mask).sum() / mask.sum()).item()
    assert abs(out["mae"] - expected) < 1e-6
    per_t_expected = ((pred_mean - y).abs() * mask).sum(dim=(0, 1)) / mask.sum(
        dim=(0, 1)
    ).clamp_min(1.0)
    assert torch.allclose(
        torch.tensor(out["mae_per_t"]), per_t_expected, atol=1e-6
    )


def test_rmse_ignores_masked_entries():
    pred_samples, pred_mean, y, y_bad, mask = _masked_fixture()
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=1)
    with patch(_MOCK_PATH, return_value=[(pred_samples, pred_mean, y_bad, mask)]):
        out = eval_rmse(ctx)
    expected = ((((pred_mean - y) ** 2) * mask).sum() / mask.sum()).sqrt().item()
    assert abs(out["rmse"] - expected) < 1e-6


def test_crps_sum_independent_of_masked_entries():
    """Corrupting masked positions (in target AND samples) must not move CRPS-sum."""
    pred_samples, _, y, y_bad, mask = _masked_fixture()
    samples_bad = torch.where(
        mask.bool().unsqueeze(1).expand_as(pred_samples),
        pred_samples,
        torch.full_like(pred_samples, -7e5),
    )
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=1)
    with patch(_MOCK_PATH, return_value=[(pred_samples, None, y, mask)]):
        clean = eval_crps_sum(ctx)["crps_sum"]
    with patch(_MOCK_PATH, return_value=[(samples_bad, None, y_bad, mask)]):
        corrupted = eval_crps_sum(ctx)["crps_sum"]
    assert abs(clean - corrupted) < 1e-6
    # Sanity: with a full mask the garbage DOES move the metric.
    ones = torch.ones_like(mask)
    with patch(_MOCK_PATH, return_value=[(samples_bad, None, y_bad, ones)]):
        unmasked = eval_crps_sum(ctx)["crps_sum"]
    assert abs(unmasked - clean) > 1e-3


def test_energy_score_independent_of_masked_entries():
    pred_samples, _, y, y_bad, mask = _masked_fixture()
    samples_bad = torch.where(
        mask.bool().unsqueeze(1).expand_as(pred_samples),
        pred_samples,
        torch.full_like(pred_samples, 3e5),
    )
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=1)
    with patch(_MOCK_PATH, return_value=[(pred_samples, None, y, mask)]):
        clean = eval_energy_score(ctx)["energy_score"]
    with patch(_MOCK_PATH, return_value=[(samples_bad, None, y_bad, mask)]):
        corrupted = eval_energy_score(ctx)["energy_score"]
    assert abs(clean - corrupted) < 1e-4


def test_masked_batch_pooling_weights_by_observed_count():
    """Cross-batch aggregation must weight by observed entries, not batches."""
    B, D, L2 = 1, 1, 2
    pred = torch.zeros(B, D, L2)
    # Batch A: two observed entries, error 1.0 each.
    y_a = torch.full((B, D, L2), 1.0)
    mask_a = torch.ones(B, D, L2)
    # Batch B: only one observed entry, error 3.0.
    y_b = torch.full((B, D, L2), 3.0)
    mask_b = torch.tensor([[[1.0, 0.0]]])
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=1)
    batches = [
        (pred.unsqueeze(1), pred, y_a, mask_a),
        (pred.unsqueeze(1), pred, y_b, mask_b),
    ]
    with patch(_MOCK_PATH, return_value=batches):
        out = eval_mae(ctx)
    # Pooled over observed entries: (2*1.0 + 1*3.0) / 3, not (1.0 + 3.0) / 2.
    assert abs(out["mae"] - 5.0 / 3.0) < 1e-6


# ---------------------------------------------------------------------------
# CRPS-sum ratio-of-means regression tests.
#
# The published GluonTS/CSDI ND convention is:
#   ND = Σ_{b,t} pinball(b,t) / Σ_{b,t} |y_sum(b,t)|
# (ratio-of-means), NOT mean over batches of (pinball_b / denom_b).
#
# With windows of very different magnitudes the two formulas diverge:
# mean-of-ratios overweights the low-magnitude window.
# ---------------------------------------------------------------------------


def _perfect_quantile_samples(y: torch.Tensor, n_samples: int = 200) -> torch.Tensor:
    """Return samples that match the empirical quantiles of y exactly.

    For a deterministic y this just tiles y across S — giving zero pinball
    at every quantile level (perfect point-mass prediction).
    """
    B, D, L2 = y.shape
    return y.unsqueeze(1).expand(B, n_samples, D, L2).contiguous()


def test_crps_sum_ratio_of_means_vs_mean_of_ratios():
    """Verify ratio-of-means formula and show it differs from mean-of-ratios.

    Two batch elements with very different magnitudes:
      - element 0: y_sum ~ 1.0 (small scale), constant prediction offset by 1.
      - element 1: y_sum ~ 1000.0 (large scale), constant prediction offset by 1.

    For a constant prediction q == c (a point mass at c) and target y,
    the pinball loss at quantile τ is:
      2·QL(τ) = 2·|τ - 1(y < c)| · |y - c|

    Averaged over τ in {0.05, 0.10, ..., 0.95} this equals |y - c| (the
    CRPS of a point mass). Since y_sum is constant per element and the
    prediction is a point mass offset by Δ:

      pinball_b = Δ   (both elements, Δ = 1 here, summed over t=1 step)
      denom_b   = |y_sum_b|

    mean-of-ratios:  (Δ / |y_sum_0| + Δ / |y_sum_1|) / 2
    ratio-of-means:  (2·Δ) / (|y_sum_0| + |y_sum_1|)

    These differ whenever |y_sum_0| ≠ |y_sum_1|.
    """
    from ddssm.eval.eval_metrics import crps_sum_metrics, crps_sum_components

    torch.manual_seed(42)
    S = 200
    # Two batch elements, D=1 channel, L2=1 timestep.
    y_small = torch.tensor([[[1.0]]])   # (1, 1, 1)
    y_large = torch.tensor([[[1000.0]]])  # (1, 1, 1)
    delta = 1.0  # constant prediction offset

    # Point-mass samples at y + delta (all samples identical).
    ps_small = (y_small + delta).expand(1, S, 1, 1).contiguous()
    ps_large = (y_large + delta).expand(1, S, 1, 1).contiguous()

    # Compute per-batch ND using ratio-of-means within single-batch call.
    nd_small, _ = crps_sum_metrics(ps_small, y_small)
    nd_large, _ = crps_sum_metrics(ps_large, y_large)

    # Manual values: for a point mass at y+Δ, CRPS = Δ = 1.0.
    # ND per element = Δ / |y_sum| = 1/1 = 1.0 and 1/1000.
    assert abs(float(nd_small) - 1.0) < 1e-3, f"nd_small={float(nd_small)}"
    assert abs(float(nd_large) - 1e-3) < 1e-4, f"nd_large={float(nd_large)}"

    # Now pool TWO batches using crps_sum_components (the correct cross-batch
    # accumulation used by eval_crps_sum).
    num0, denom0 = crps_sum_components(ps_small, y_small)
    num1, denom1 = crps_sum_components(ps_large, y_large)

    total_num = float(num0.sum() + num1.sum())
    total_denom = float(denom0.sum() + denom1.sum())
    ratio_of_means = total_num / total_denom

    # Analytic ratio-of-means: (1 + 1) / (1 + 1000) = 2 / 1001.
    expected_rom = 2.0 / (1.0 + 1000.0)
    assert abs(ratio_of_means - expected_rom) < 1e-3, (
        f"ratio_of_means={ratio_of_means:.6f}, expected={expected_rom:.6f}"
    )

    # The mean-of-ratios (old, wrong formula) gives a DIFFERENT value.
    mean_of_ratios = (float(nd_small) + float(nd_large)) / 2.0
    # mean-of-ratios ≈ (1.0 + 0.001) / 2 ≈ 0.5005
    # ratio-of-means ≈ 2 / 1001 ≈ 0.002
    assert abs(mean_of_ratios - ratio_of_means) > 0.1, (
        "mean-of-ratios and ratio-of-means should differ substantially "
        f"(mean_of_ratios={mean_of_ratios:.4f}, ratio_of_means={ratio_of_means:.4f})"
    )


def test_eval_crps_sum_uses_ratio_of_means():
    """eval_crps_sum across two batches of different magnitude matches ratio-of-means.

    Feeds two batches through the metric via the mock and checks that the
    global result equals the analytic ratio-of-means, not mean-of-ratios.
    """
    from ddssm.eval.eval_metrics import crps_sum_components

    S = 200
    delta = 1.0

    y_small = torch.tensor([[[1.0]]])   # (B=1, D=1, L2=1)
    y_large = torch.tensor([[[1000.0]]])

    ps_small = (y_small + delta).expand(1, S, 1, 1).contiguous()
    ps_large = (y_large + delta).expand(1, S, 1, 1).contiguous()

    mask_ones_small = torch.ones(1, 1, 1)
    mask_ones_large = torch.ones(1, 1, 1)

    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"), T_split=1)
    batches = [
        (ps_small, None, y_small, mask_ones_small),
        (ps_large, None, y_large, mask_ones_large),
    ]
    with patch(_MOCK_PATH, return_value=batches):
        out = eval_crps_sum(ctx)

    # Analytic ratio-of-means: 2 / (1 + 1000) = 2/1001.
    expected = 2.0 / 1001.0
    assert abs(out["crps_sum"] - expected) < 1e-3, (
        f"crps_sum={out['crps_sum']:.6f}, expected={expected:.6f}"
    )

    # per_t should sum to the global value (L2=1 so per_t has one element).
    assert len(out["crps_sum_per_t"]) == 1
    assert abs(out["crps_sum_per_t"][0] - out["crps_sum"]) < 1e-6, (
        "per_t[0] should equal global when L2=1"
    )


def test_unknown_metric_raises():
    spec = EvalSpec(metrics=["nope"], split="val")

    class _StubData(DDSSMDataModule):
        batch_transform = staticmethod(lambda b, d: b)
        metadata = DataMetadata(data_dim=1, forecast_split=None)

        def train_loader(self):
            return None

        def val_loader(self):
            return None

        def test_loader(self):
            return None

    class _StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)

        def to(self, d):
            return self

    class _StubExpt:
        def __init__(self):
            self.data = _StubData()
            self.model = _StubModel()

    with pytest.raises(KeyError):
        evaluate(_StubExpt(), spec, device=torch.device("cpu"), run_dir="/tmp/_nope")
