"""Tests for the ``bimodal_jsd`` eval metric and its NPZ side-file output."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ddssm.eval import EvalContext, METRIC_REGISTRY
from ddssm.eval.metrics import (
    _bimodal_truth_mass,
    _hist_mass,
    _jsd_discrete,
    eval_bimodal_jsd,
)


def test_bimodal_jsd_in_registry():
    assert "bimodal_jsd" in METRIC_REGISTRY


def test_jsd_self_is_zero():
    """JSD(p, p) = 0 (sanity check on the helper)."""
    edges = np.linspace(-5.0, 5.0, 51)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = _bimodal_truth_mass(centers, x_prev=1.0, a=0.9, step_size=4.0,
                            sigma=0.2, center_coef=0.9)
    assert abs(_jsd_discrete(p, p)) < 1e-12


def test_jsd_symmetric_and_bounded():
    """JSD is symmetric and lies in [0, log(2)]."""
    edges = np.linspace(-5.0, 5.0, 51)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = _bimodal_truth_mass(centers, x_prev=1.0, a=0.9, step_size=4.0,
                            sigma=0.2, center_coef=0.9)
    q = _bimodal_truth_mass(centers, x_prev=1.0, a=0.9, step_size=4.0,
                            sigma=0.2, center_coef=0.0)  # different shift
    pq = _jsd_discrete(p, q)
    qp = _jsd_discrete(q, p)
    assert abs(pq - qp) < 1e-12
    assert 0.0 <= pq <= math.log(2.0) + 1e-9


def test_hist_mass_uniform_for_empty():
    """Empty histogram falls back to uniform (avoids div-by-zero downstream)."""
    edges = np.linspace(-1.0, 1.0, 5)
    p = _hist_mass(np.array([], dtype=np.float32), edges)
    assert p.shape == (4,)
    assert np.allclose(p, np.full(4, 0.25))


# ---------------------------------------------------------------------------
# End-to-end eval_bimodal_jsd test with a stub model whose ``forecast`` returns
# samples drawn from the analytic truth — JSD should be small (≪ log 2).
# ---------------------------------------------------------------------------


class _StubBimodalForecaster(torch.nn.Module):
    """Returns samples drawn from the analytic bimodal one-step distribution."""

    def __init__(self, *, a: float = 0.9, step_size: float = 4.0, sigma: float = 0.2):
        super().__init__()
        self.a, self.step_size, self.sigma = a, step_size, sigma

    def forecast(self, *, x_hist, x_mask, past_time, future_time, num_samples, **_):
        B = x_hist.shape[0]
        x_prev = x_hist[:, 0, -1]  # (B,)
        # 50/50 mixture: shift = a*x_prev ± step_size, plus N(0, sigma)
        s = (torch.randint(0, 2, (B, num_samples)).float() * 2.0) - 1.0
        eps = torch.randn(B, num_samples) * self.sigma
        samples = self.a * x_prev[:, None] + self.step_size * s + eps  # (B, S)
        # Shape into (B, S, D=1, L2=1)
        pred = samples[:, :, None, None]
        return {"pred_samples": pred, "pred_mean": pred.mean(dim=1)}


def _make_bimodal_loader(B: int, T: int, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 1, T, generator=g) * 2.0
    mask = torch.ones_like(x)
    timepoints = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, -1)

    class _DictDataset(torch.utils.data.Dataset):
        def __len__(self): return B
        def __getitem__(self, i):
            return {
                "observed_data": x[i],
                "observation_mask": mask[i],
                "timepoints": timepoints[i],
            }

    return DataLoader(_DictDataset(), batch_size=B)


def test_bimodal_jsd_low_for_truth_drawn_samples(tmp_path):
    """Samples drawn from the analytic truth distribution → small JSD."""
    torch.manual_seed(0)
    np.random.seed(0)

    loader = _make_bimodal_loader(B=64, T=8)
    ctx = EvalContext(
        model=_StubBimodalForecaster(),
        loader=loader,
        device=torch.device("cpu"),
        T_split=7,         # one-step horizon
        num_samples=512,   # enough for stable histogram
        run_dir=str(tmp_path),
    )
    out = eval_bimodal_jsd(ctx)
    assert out["bimodal_jsd_n"] == 64
    # JSD should be very small for a well-calibrated forecast (truth-matched),
    # well below log(2) ≈ 0.693. Empirically sits around 0.01-0.05.
    assert out["bimodal_jsd_mean"] < 0.1, out


def test_bimodal_jsd_high_for_constant_predictor(tmp_path):
    """A predictor that always returns ``x_prev`` (LOCF) → larger JSD."""
    torch.manual_seed(0)

    class _LocfForecaster(torch.nn.Module):
        def forecast(self, *, x_hist, x_mask, past_time, future_time, num_samples, **_):
            B = x_hist.shape[0]
            x_prev = x_hist[:, 0, -1]
            samples = x_prev[:, None].expand(B, num_samples)  # (B, S)
            pred = samples[:, :, None, None]
            return {"pred_samples": pred, "pred_mean": pred.mean(dim=1)}

    loader = _make_bimodal_loader(B=64, T=8)
    ctx = EvalContext(
        model=_LocfForecaster(),
        loader=loader,
        device=torch.device("cpu"),
        T_split=7, num_samples=512,
        run_dir=str(tmp_path),
    )
    out = eval_bimodal_jsd(ctx)
    # LOCF concentrates all mass at one bin → high JSD vs the bimodal truth.
    # Empirically sits near log(2) ≈ 0.69.
    assert out["bimodal_jsd_mean"] > 0.4, out


def test_bimodal_jsd_writes_npz(tmp_path):
    """When ``npz_path`` is given, the metric writes the per-sample NPZ."""
    torch.manual_seed(0)
    loader = _make_bimodal_loader(B=8, T=8)
    ctx = EvalContext(
        model=_StubBimodalForecaster(),
        loader=loader,
        device=torch.device("cpu"),
        T_split=7, num_samples=64,
        run_dir=str(tmp_path),
    )
    out = eval_bimodal_jsd(ctx, npz_path="bimodal.npz")
    assert (tmp_path / "bimodal.npz").is_file()
    npz = np.load(tmp_path / "bimodal.npz")
    for key in ("sample_idx", "x_prev", "xhat_samples",
                "edges", "centers", "model_mass", "truth_mass"):
        assert key in npz.files, f"missing NPZ key: {key}"
    assert npz["xhat_samples"].shape == (8, 64)
    assert npz["model_mass"].shape == npz["truth_mass"].shape
    assert "bimodal_jsd_median_idx_1" in out
    assert "bimodal_jsd_median_idx_2" in out


def test_bimodal_jsd_npz_path_absolute(tmp_path):
    """Absolute ``npz_path`` is honoured even when ``run_dir`` is set."""
    target = tmp_path / "elsewhere" / "bimodal.npz"
    torch.manual_seed(0)
    loader = _make_bimodal_loader(B=4, T=8)
    ctx = EvalContext(
        model=_StubBimodalForecaster(), loader=loader,
        device=torch.device("cpu"), T_split=7, num_samples=32,
        run_dir=str(tmp_path / "run"),
    )
    eval_bimodal_jsd(ctx, npz_path=str(target))
    assert target.is_file()


def test_bimodal_jsd_requires_t_split():
    """Missing T_split must raise — the metric is forecasting-based."""
    ctx = EvalContext(model=_StubBimodalForecaster(), loader=None,
                      device=torch.device("cpu"), T_split=None)
    with pytest.raises(ValueError):
        eval_bimodal_jsd(ctx)
