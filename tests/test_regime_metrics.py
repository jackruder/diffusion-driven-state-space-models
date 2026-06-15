"""Unit tests for system-agnostic regime metrics in ddssm.eval.regime.

Tests the pure tensor helpers independently of any model or data module,
then a smoke test of the registered metric that verifies key presence and
basic sanity bounds on synthetic-forecast outputs.
"""

from __future__ import annotations

import torch
import numpy as np
import pytest

from ddssm.eval.regime import (
    regime_labels,
    first_switch_times,
    run_lengths,
)


# ---------------------------------------------------------------------------
# regime_labels
# ---------------------------------------------------------------------------


def test_firm_labels_no_deadband() -> None:
    """Without a deadband all points get ±1."""
    x = torch.tensor([1.0, -1.0, 2.0, -2.0])
    lab = regime_labels(x)
    assert lab.tolist() == [1, -1, 1, -1]


def test_deadband_carries_last_firm() -> None:
    """Points inside the band inherit the last firm label."""
    x = torch.tensor([-1.0, 0.05, 0.05, 1.0])
    lab = regime_labels(x, deadband=0.1)
    # -1 firm, 0.05 in band -> carry -1, carry -1, +1 firm
    assert lab.tolist() == [-1, -1, -1, 1]


def test_deadband_backfills_leading_band() -> None:
    """Leading in-band points without initial are backfilled from the first firm."""
    x = torch.tensor([0.05, 0.05, 1.0])
    lab = regime_labels(x, deadband=0.1)
    # No initial -> backfill from first firm (+1 at index 2)
    assert lab.tolist() == [1, 1, 1]


def test_deadband_initial_fills_leading_band() -> None:
    """initial= is used for leading in-band points instead of backfill."""
    x = torch.tensor([[0.05, 0.05, 1.0]])  # (1, T)
    initial = torch.tensor([-1])
    lab = regime_labels(x, deadband=0.1, initial=initial)
    assert lab[0].tolist() == [-1, -1, 1]


def test_all_in_band_without_initial_stays_zero() -> None:
    """A row that never exits the band (and no initial) stays zero."""
    x = torch.tensor([0.0, 0.01, -0.01])
    lab = regime_labels(x, deadband=0.5)
    assert (lab == 0).all()


def test_batch_regime_labels() -> None:
    """Batched input (B, T) is processed independently per row."""
    x = torch.tensor([
        [1.0, -1.0],
        [-1.0, 1.0],
    ])
    lab = regime_labels(x)
    assert lab[0].tolist() == [1, -1]
    assert lab[1].tolist() == [-1, 1]


# ---------------------------------------------------------------------------
# first_switch_times
# ---------------------------------------------------------------------------


def test_switch_detected_at_correct_index() -> None:
    """Switch time is the first index where the label differs from ref."""
    lab = torch.tensor([[1, 1, -1, -1]])
    ref = torch.tensor([1])
    times, switched = first_switch_times(lab, ref)
    assert switched[0].item() is True
    assert times[0].item() == 2


def test_censored_when_no_switch() -> None:
    """Row that never switches gets times==T and switched==False."""
    lab = torch.tensor([[1, 1, 1]])
    ref = torch.tensor([1])
    times, switched = first_switch_times(lab, ref)
    assert switched[0].item() is False
    assert times[0].item() == 3  # censored at T=3


def test_zero_labels_not_treated_as_switch() -> None:
    """All-zero (ambiguous) positions are not counted as switches."""
    lab = torch.tensor([[0, 0, -1]])
    ref = torch.tensor([1])
    times, switched = first_switch_times(lab, ref)
    assert times[0].item() == 2  # the -1 at index 2


# ---------------------------------------------------------------------------
# run_lengths
# ---------------------------------------------------------------------------


def test_run_lengths_basic() -> None:
    """Interior run lengths are extracted correctly."""
    lab = np.array([[1, 1, -1, -1, -1, 1]])  # interior run: [-1,-1,-1] length 3
    runs = run_lengths(lab)
    assert list(runs) == [3]


def test_run_lengths_drop_censored_true() -> None:
    """Edge runs (touching start or end) are dropped by default."""
    lab = np.array([[1, -1, 1]])  # 3 runs; first and last are edge -> only middle kept
    runs = run_lengths(lab)
    assert list(runs) == [1]


def test_run_lengths_drop_censored_false() -> None:
    """All runs including edge runs when drop_censored=False."""
    lab = np.array([[1, -1, 1]])
    runs = run_lengths(lab, drop_censored=False)
    assert list(runs) == [1, 1, 1]


def test_run_lengths_all_zero_row_excluded() -> None:
    """Rows that never reach a firm label contribute nothing."""
    lab = np.array([[0, 0, 0]])
    runs = run_lengths(lab)
    assert runs.size == 0


def test_run_lengths_multiple_rows() -> None:
    """Run lengths are pooled across rows."""
    lab = np.array([
        [1, -1, 1],
        [1, 1, -1],
    ])
    runs = run_lengths(lab)
    # Row 0: interior run [- 1] len 1. Row 1: no interior runs (two runs, both edge).
    assert list(runs) == [1]


# ---------------------------------------------------------------------------
# Metric registration side-effect
# ---------------------------------------------------------------------------


def test_regime_registered() -> None:
    """Importing ddssm.eval populates METRIC_REGISTRY with 'regime'."""
    from ddssm.eval.metrics import METRIC_REGISTRY
    assert "regime" in METRIC_REGISTRY


# ---------------------------------------------------------------------------
# Smoke test of eval_regime with a tiny mock model + loader
# ---------------------------------------------------------------------------


class _ConstantForecastModel:
    """Tiny stand-in: always predicts zeros for all future steps."""

    j = 1
    emb_time_dim = 4

    def forecast(self, x_hist, x_mask, past_time, future_time, **kw):
        B = x_hist.shape[0]
        D = x_hist.shape[1]
        L2 = future_time.shape[1]
        S = kw.get("num_samples", 4)
        return {
            "pred_samples": torch.zeros(B, S, D, L2),
            "pred_mean": torch.zeros(B, D, L2),
        }


def _make_ctx(x_full: torch.Tensor, T_split: int, num_samples: int = 4):
    from ddssm.eval.metrics import EvalContext
    from torch.utils.data import DataLoader, TensorDataset

    T = x_full.shape[-1]
    mask = torch.ones_like(x_full)
    timepoints = (
        torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(x_full.shape[0], -1)
    )
    ds = TensorDataset(x_full, mask, timepoints)

    def collate(items):
        xs, ms, ts = zip(*items)
        return {
            "observed_data": torch.stack(xs),
            "observation_mask": torch.stack(ms),
            "timepoints": torch.stack(ts),
        }

    loader = DataLoader(ds, batch_size=x_full.shape[0], collate_fn=collate)
    return EvalContext(
        model=_ConstantForecastModel(),
        loader=loader,
        device=torch.device("cpu"),
        T_split=T_split,
        num_samples=num_samples,
    )


def test_regime_metric_returns_expected_keys() -> None:
    """eval_regime returns the core keys with a trivial (constant-zero) model."""
    from ddssm.eval.regime import eval_regime

    # Four sequences: two positive-lobe, two negative-lobe context endpoints
    T, L1 = 16, 8
    x = torch.cat([
        torch.ones(2, 1, T),    # always positive
        -torch.ones(2, 1, T),   # always negative
    ], dim=0)
    ctx = _make_ctx(x, T_split=L1)
    result = eval_regime(ctx, channel=0, deadband=0.1, k_steps=(2, 4))

    assert "regime_n_sequences" in result
    assert "regime_acc_per_t" in result
    assert len(result["regime_acc_per_t"]) == T - L1
    assert "regime_acc_at_2" in result
    assert "regime_acc_at_4" in result
    assert "regime_n_truth_switches" in result


def test_regime_persistence_accuracy_expected_value() -> None:
    """A constant-lobe ground truth gives perfect persistence accuracy."""
    from ddssm.eval.regime import eval_regime

    T, L1 = 8, 4
    x = torch.ones(4, 1, T)  # all +1; context boundary also +1
    ctx = _make_ctx(x, T_split=L1)
    result = eval_regime(ctx, channel=0, deadband=0.0)
    # Persistence always predicts +1, truth is always +1 -> acc = 1.0
    for k in (4, 8, 16):
        val = result.get(f"regime_persistence_acc_at_{k}", 1.0)
        assert val == pytest.approx(1.0), (
            f"persistence_acc_at_{k} should be 1.0 for constant-positive data"
        )
