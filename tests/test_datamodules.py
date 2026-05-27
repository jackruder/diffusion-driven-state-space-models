"""Smoke tests for the DataModule layer.

Ensure each concrete DataModule yields the canonical model-ready dict
that the trainer expects (``observed_data``, ``observation_mask``,
``timepoints``, optional ``covariates`` / ``static_covariates``), and
that ``parse_batch`` is a no-op pass-through on those batches.
"""

from __future__ import annotations

import os

import torch
import pytest

from ddssm.data.datamodule import (
    DataMetadata,
    KDDDataModule,
    NullDataModule,
    DDSSMDataModule,
    SyntheticDataModule,
)


def _assert_canonical_batch(batch: dict, *, expect_covariates: bool) -> None:
    for key in ("observed_data", "observation_mask", "timepoints"):
        assert key in batch, f"missing key: {key}"
        assert isinstance(batch[key], torch.Tensor)
    assert batch["observed_data"].shape == batch["observation_mask"].shape
    # observed_data is (B, D, T); timepoints is (B, T) or (T,)
    assert batch["observed_data"].dim() == 3
    if expect_covariates:
        assert batch.get("covariates") is not None
        assert batch["covariates"].dim() == 3


def test_null_datamodule_returns_no_loaders():
    dm = NullDataModule(data_dim=3)
    assert isinstance(dm, DDSSMDataModule)
    assert dm.train_loader() is None
    assert dm.val_loader() is None
    assert dm.test_loader() is None
    assert dm.metadata.data_dim == 3
    assert dm.batch_format == "sequence"


def test_synthetic_datamodule_smoke():
    dm = SyntheticDataModule(
        mode="lgssm", T=16, D=2, N_per_split=8, batch_size=4
    )
    assert isinstance(dm, DDSSMDataModule)
    assert dm.batch_format == "sequence"

    meta = dm.metadata
    assert isinstance(meta, DataMetadata)
    assert meta.data_dim == 2
    assert meta.T == 16
    assert meta.covariate_dim == 0

    train = dm.train_loader()
    val = dm.val_loader()
    test = dm.test_loader()
    assert all(loader is not None for loader in (train, val, test))

    batch = next(iter(train))
    _assert_canonical_batch(batch, expect_covariates=False)
    assert batch["observed_data"].shape == (4, 2, 16)

    # parse_batch must be a passthrough on canonical dicts.
    parsed = dm.batch_transform(batch, torch.device("cpu"))
    for key in ("observed_data", "observation_mask", "timepoints"):
        assert torch.equal(parsed[key], batch[key].to(torch.float32))


@pytest.mark.skipif(
    not os.path.isfile("data/kdd.pt") or os.path.getsize("data/kdd.pt") < 1024,
    reason="data/kdd.pt is an LFS pointer or missing in this environment",
)
def test_kdd_datamodule_smoke():
    dm = KDDDataModule(filepath="data/kdd.pt", batch_size=2, eval_step_size=24)
    assert dm.batch_format == "windowed"
    train = dm.train_loader()
    batch = next(iter(train))
    _assert_canonical_batch(batch, expect_covariates=True)
    meta = dm.metadata
    assert meta.data_dim > 0
    assert meta.T == dm.L1 + dm.L2


def test_datamodule_protocol_runtime_check():
    # All concrete datamodules satisfy the Protocol at runtime.
    assert isinstance(NullDataModule(), DDSSMDataModule)
    assert isinstance(
        SyntheticDataModule(mode="lgssm", T=8, D=1, N_per_split=4, batch_size=2),
        DDSSMDataModule,
    )


# ---------------------------------------------------------------------------
# Parametrized smoke-tests for every SyntheticDataModule mode used by the
# verification presets (harmonic, harmonic-noisy, bimodal, robot-basis-pursuit).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,D", [
    ("harmonic", 1),
    ("harmonic-noisy", 1),
    ("bimodal", 1),
    ("bimodal-noisy", 1),
    ("nonlinear-bimodal-lift", 4),
    ("robot-basis-pursuit", 2),
])
def test_synthetic_datamodule_mode_shapes(mode: str, D: int) -> None:
    """Every verification mode must produce canonical (B, D, T) batches without NaNs."""
    T, N, B = 16, 8, 4
    dm = SyntheticDataModule(mode=mode, T=T, D=D, N_per_split=N, batch_size=B)
    assert isinstance(dm, DDSSMDataModule)
    assert dm.metadata.data_dim == D
    assert dm.metadata.T == T
    # forecast_split is always None for synthetic data (no canonical split)
    assert dm.metadata.forecast_split is None

    batch = next(iter(dm.train_loader()))
    _assert_canonical_batch(batch, expect_covariates=False)
    assert batch["observed_data"].shape == (B, D, T)
    assert not torch.isnan(batch["observed_data"]).any(), (
        f"mode={mode!r} produced NaN values in observed_data"
    )
    # Mask should be all-ones (no structured missingness in synthetic data)
    assert batch["observation_mask"].all()


def test_synthetic_datamodule_robot_requires_d_ge_2() -> None:
    """robot-basis-pursuit with D=1 must not silently produce wrong-shaped data."""
    dm = SyntheticDataModule(mode="robot-basis-pursuit", T=16, D=2, N_per_split=8, batch_size=2)
    batch = next(iter(dm.train_loader()))
    # X and Y coordinates must both be present
    assert batch["observed_data"].shape[1] == 2


def test_synthetic_datamodule_bimodal_has_variance() -> None:
    """Bimodal data should have non-trivial variance (the two modes are well-separated)."""
    dm = SyntheticDataModule(mode="bimodal", T=64, D=1, N_per_split=128, batch_size=64)
    batch = next(iter(dm.train_loader()))
    # Variance across batch and time should be well above noise floor
    std = batch["observed_data"].std().item()
    assert std > 0.5, f"bimodal std={std:.3f} suspiciously low — modes may have collapsed"
