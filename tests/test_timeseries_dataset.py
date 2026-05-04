import sys
from pathlib import Path

# ensure src is importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import pytest
from ddssm.dataset import MultiSeriesDataset, collate_fixed, compute_masked_mean_std


def make_synthetic_series(T=10, D=3, missing_frac=0.2, irregular=False, seed=0):
    rng = np.random.RandomState(seed)
    if irregular:
        base = np.arange(T, dtype=float)
        jitter = rng.uniform(-0.3, 0.3, size=T)
        timestamps = np.sort(base + jitter)
    else:
        timestamps = np.arange(T, dtype=float)

    values = rng.randn(T, D)
    mask = np.ones((T, D), dtype=float)
    num_missing = int(missing_frac * T * D)
    idx = rng.choice(T * D, num_missing, replace=False)
    values.reshape(-1)[idx] = 0.0
    mask.reshape(-1)[idx] = 0.0
    values_with_nan = values.copy()
    values_with_nan[0, 0] = np.nan
    mask[0, 0] = 0.0

    return {"timestamps": timestamps, "values": values_with_nan, "mask": mask}


def test_compute_masked_mean_std_simple():
    values = np.array([[1.0, 2.0], [3.0, np.nan], [np.nan, 4.0]])
    mask = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    mean, std = compute_masked_mean_std(values, mask)
    np.testing.assert_allclose(mean, np.array([2.0, 3.0]), atol=1e-5)
    np.testing.assert_allclose(std, np.array([1.0, 1.0]), atol=1e-5)


@pytest.mark.parametrize("history_length,pred_length", [(4, 2), (3, 3)])
def test_multiseries_basic_windowing(history_length, pred_length):
    series = make_synthetic_series(T=12, D=2, missing_frac=0.1, seed=42)
    dataset = MultiSeriesDataset(
        series_list=[series],
        history_length=history_length,
        pred_length=pred_length,
        normalize=True,
    )
    T = len(series["timestamps"])
    expected_windows = (T - (history_length + pred_length)) // pred_length + 1
    assert len(dataset) == expected_windows

    for i in range(len(dataset)):
        item = dataset[i]
        L = history_length + pred_length
        assert item["observed_data"].shape == (L, 2)
        assert item["observed_mask"].shape == (L, 2)
        assert item["gt_mask"].shape == (L, 2)
        assert item["timepoints"].shape == (L,)
        assert item["feature_id"].shape == (2,)
        assert item["raw_timestamps"].shape == (L,)
        assert torch.all(item["gt_mask"][-pred_length:] == 0)
        tp = item["timepoints"].numpy()
        assert tp[0] == 0.0
        assert np.all(np.diff(tp) >= -1e-6)

    if len(dataset) >= 2:
        batch = [dataset[0], dataset[1]]
        collated = collate_fixed(batch)
        for key, tensor in collated.items():
            assert tensor.shape[0] == 2


def test_global_mean_std_override():
    T, D = 8, 4
    timestamps = np.arange(T, dtype=float)
    values = np.ones((T, D)) * 5.0
    mask = np.ones((T, D), dtype=float)
    series = {"timestamps": timestamps, "values": values, "mask": mask}
    manual_mean = np.zeros(D, dtype=float)
    manual_std = np.ones(D, dtype=float) * 2.0
    dataset = MultiSeriesDataset(
        series_list=[series],
        history_length=3,
        pred_length=1,
        normalize=True,
        global_mean=manual_mean,
        global_std=manual_std,
    )
    item = dataset[0]
    obs = item["observed_data"]
    assert torch.allclose(obs, torch.full_like(obs, 2.5))


def test_error_on_short_series():
    series = make_synthetic_series(T=3, D=2)
    dataset = MultiSeriesDataset(
        series_list=[series],
        history_length=3,
        pred_length=2,
        normalize=False,
    )
    assert len(dataset) == 0


def test_multiple_series_combined():
    s1 = make_synthetic_series(T=10, D=3, seed=1)
    s2 = make_synthetic_series(T=10, D=3, seed=2)
    dataset = MultiSeriesDataset(
        series_list=[s1, s2],
        history_length=4,
        pred_length=2,
        normalize=True,
    )
    assert len(dataset) > 0
    _ = dataset[0]
    if len(dataset) > 1:
        _ = dataset[1]
