"""Tests for the ``obs_space_jsd`` eval metric.

Covers:

- The deterministic reconstruction of the DGP obs-lift MLP
  (``_mv_lift_matrices``): shapes, and byte-identical repeated calls.
- The GT-sign trajectory surface (``gt_signs`` in each item of a
  ``nonlinear-bimodal-lift-mv`` dataset with ``expose_gt_latents=True``).
- End-to-end smoke of ``eval_obs_space_jsd`` against a stub model whose
  ``forecast`` returns random samples of the right shape.
- The metric skips cleanly on the wrong dataset mode.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from ddssm.eval import METRIC_REGISTRY, EvalContext
from ddssm.eval.metrics import eval_obs_space_jsd
from ddssm.data.datamodule import SyntheticDataModule
from ddssm.data.synthetic import (
    NLBL_MV_HIDDEN_DIM,
    NLBL_MV_LATENT_D,
    NLBL_MV_OBS_D,
    SyntheticDataset,
    _mv_lift_matrices,
)


# ---------------------------------------------------------------------------
# _mv_lift_matrices — deterministic MLP weight reconstruction.
# ---------------------------------------------------------------------------


def test_lift_matrices_deterministic() -> None:
    """Two calls yield equal arrays with the documented shapes."""
    W1a, b1a, W2a, b2a = _mv_lift_matrices()
    W1b, b1b, W2b, b2b = _mv_lift_matrices()
    assert W1a.shape == (NLBL_MV_HIDDEN_DIM, NLBL_MV_LATENT_D) == (16, 4)
    assert b1a.shape == (NLBL_MV_HIDDEN_DIM,) == (16,)
    assert W2a.shape == (NLBL_MV_OBS_D, NLBL_MV_HIDDEN_DIM) == (8, 16)
    assert b2a.shape == (NLBL_MV_OBS_D,) == (8,)
    assert np.array_equal(W1a, W1b)
    assert np.array_equal(b1a, b1b)
    assert np.array_equal(W2a, W2b)
    assert np.array_equal(b2a, b2b)


# ---------------------------------------------------------------------------
# gt_signs surface.
# ---------------------------------------------------------------------------


def test_gt_signs_exposed() -> None:
    """The nlblmv dataset exposes ``gt_signs`` of shape (d, T), values ±1."""
    T = 16
    N = 8
    ds = SyntheticDataset(
        mode="nonlinear-bimodal-lift-mv",
        split="val",
        N_per_split=N,
        T=T,
        D=NLBL_MV_OBS_D,
        dataset_seed=0,
        expose_gt_latents=True,
    )
    item = ds[0]
    assert "gt_signs" in item
    signs = item["gt_signs"]
    assert signs.shape == (NLBL_MV_LATENT_D, T) == (4, 16)
    # Values are exactly ±1.
    uniq = torch.unique(signs).tolist()
    assert set(uniq).issubset({-1.0, 1.0})


def test_gt_signs_batched_through_datamodule() -> None:
    """A SyntheticDataModule batch on nlblmv carries gt_signs of shape (B, d, T)."""
    T = 16
    N = 8
    B = 4
    dm = SyntheticDataModule(
        mode="nonlinear-bimodal-lift-mv",
        T=T,
        D=NLBL_MV_OBS_D,
        N_per_split=N,
        batch_size=B,
        dataset_seed=0,
        expose_gt_latents=True,
    )
    batch = next(iter(dm.val_loader()))
    assert "gt_signs" in batch
    assert batch["gt_signs"].shape == (B, NLBL_MV_LATENT_D, T) == (4, 4, 16)


# ---------------------------------------------------------------------------
# End-to-end smoke: stubbed forecast + a small nlblmv loader.
# ---------------------------------------------------------------------------


class _StubForecaster(torch.nn.Module):
    """Stub whose ``forecast`` returns random N(0, 1) samples of the right shape."""

    def forecast(self, *, x_hist, x_mask, past_time, future_time, num_samples, **_):
        B = x_hist.shape[0]
        D = x_hist.shape[1]
        L2 = int(future_time.size(1))
        samples = torch.randn(B, num_samples, D, L2)
        return {"pred_samples": samples, "pred_mean": samples.mean(dim=1)}


def test_obs_space_jsd_in_registry() -> None:
    assert "obs_space_jsd" in METRIC_REGISTRY


def test_obs_space_jsd_smoke() -> None:
    """End-to-end run on a stubbed forecaster + a small nlblmv val loader."""
    torch.manual_seed(0)
    dm = SyntheticDataModule(
        mode="nonlinear-bimodal-lift-mv",
        T=32,
        D=NLBL_MV_OBS_D,
        N_per_split=16,
        batch_size=8,
        dataset_seed=0,
        expose_gt_latents=True,
    )
    loader = dm.val_loader()
    ctx = EvalContext(
        model=_StubForecaster(),
        loader=loader,
        device=torch.device("cpu"),
        batch_transform=dm.batch_transform,
        num_samples=32,
    )
    out = eval_obs_space_jsd(ctx, max_batches=2)
    assert out["obs_space_jsd_available"] is True
    hd = out["obs_space_jsd_mean"]
    assert isinstance(hd, float) and math.isfinite(hd)
    # JSD is bounded above by log(2). Values should be in [0, log 2].
    assert 0.0 <= hd <= math.log(2.0) + 1e-9, hd
    per_origin = out["obs_space_jsd_per_origin"]
    assert set(per_origin.keys()) == {8, 16, 24}
    for v in per_origin.values():
        assert math.isfinite(v)
        assert 0.0 <= v <= math.log(2.0) + 1e-9
    per_dim = out["obs_space_jsd_per_dim"]
    assert len(per_dim) == NLBL_MV_OBS_D == 8
    for v in per_dim:
        assert math.isfinite(v)


def test_obs_space_jsd_skips_wrong_mode() -> None:
    """Dataset with a different mode → returns available=False cleanly."""
    dm = SyntheticDataModule(
        mode="lgssm",
        T=32,
        D=1,
        N_per_split=8,
        batch_size=4,
        dataset_seed=0,
    )
    ctx = EvalContext(
        model=_StubForecaster(),
        loader=dm.val_loader(),
        device=torch.device("cpu"),
        batch_transform=dm.batch_transform,
        num_samples=8,
    )
    out = eval_obs_space_jsd(ctx, max_batches=1)
    assert out["obs_space_jsd_available"] is False
    assert "reason" in out["obs_space_jsd_reason"].lower() or isinstance(
        out["obs_space_jsd_reason"], str
    )


def test_obs_space_jsd_skips_without_model_or_loader() -> None:
    """No model / no loader → available=False."""
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"))
    out = eval_obs_space_jsd(ctx)
    assert out["obs_space_jsd_available"] is False
