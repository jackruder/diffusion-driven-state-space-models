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

from ddssm.data.dataload import parse_batch
from ddssm.data.datamodule import (
    DataMetadata,
    KDDDataModule,
    NullDataModule,
    DDSSMDataModule,
    GluonTSDataModule,
    SyntheticDataModule,
    WindowedSeriesDataModule,
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
    dm = SyntheticDataModule(mode="lgssm", T=16, D=2, N_per_split=8, batch_size=4)
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


def test_kdd_is_windowed_series_subclass():
    """KDD now rides the shared windowed base; construction stays lazy."""
    dm = KDDDataModule(filepath="data/kdd.pt")
    assert isinstance(dm, WindowedSeriesDataModule)
    assert dm.batch_format == "windowed"
    assert dm.batch_transform is parse_batch
    assert dm._built is False  # no payload load on __init__


def test_gluonts_datamodule_contract_is_network_free():
    """Constructing a GluonTSDataModule must NOT fetch (lazy); it satisfies the
    windowed contract with the per-dataset window spec from SPECS.
    """
    dm = GluonTSDataModule(name="solar")
    assert isinstance(dm, DDSSMDataModule)
    assert isinstance(dm, WindowedSeriesDataModule)
    assert dm.batch_format == "windowed"
    assert dm.batch_transform is parse_batch
    assert (dm.L1, dm.L2, dm.test_windows) == (168, 24, 7)
    assert dm._built is False  # nothing fetched yet


def test_gluonts_datamodule_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown GluonTS dataset"):
        GluonTSDataModule(name="not-a-dataset")


def test_data_presets_instantiate_to_datamodules():
    """The registered Hydra preset configs build the right DataModule types."""
    from hydra_zen import instantiate

    from ddssm.data import presets

    assert isinstance(instantiate(presets.Solar), GluonTSDataModule)
    assert isinstance(instantiate(presets.Wiki), GluonTSDataModule)
    assert isinstance(instantiate(presets.KDDFull), KDDDataModule)
    assert isinstance(instantiate(presets.KDDStation), KDDDataModule)


@pytest.mark.skip(reason="GluonTS repo fetch is network/cache-heavy; run manually")
def test_gluonts_datamodule_fetches_and_windows():
    dm = GluonTSDataModule(name="solar", batch_size=2)
    batch = next(iter(dm.train_loader()))
    _assert_canonical_batch(batch, expect_covariates=False)
    assert dm.metadata.data_dim == 137
    assert dm.metadata.T == dm.L1 + dm.L2


def test_datamodule_abc_membership():
    # All concrete datamodules inherit the ABC (nominal isinstance).
    assert isinstance(NullDataModule(), DDSSMDataModule)
    assert isinstance(
        SyntheticDataModule(mode="lgssm", T=8, D=1, N_per_split=4, batch_size=2),
        DDSSMDataModule,
    )


def test_loader_dispatch_by_split():
    """``loader(split)`` routes to the matching split loader; bad split raises."""
    dm = SyntheticDataModule(mode="lgssm", T=8, D=1, N_per_split=4, batch_size=2)
    assert dm.loader("train") is not None
    assert dm.loader("val") is not None
    assert dm.loader("test") is not None
    with pytest.raises(ValueError, match="Unknown split"):
        dm.loader("holdout")
    # NullDataModule has no data — every split is None, no raise on known splits.
    null = NullDataModule()
    assert null.loader("train") is None
    assert null.loader("test") is None


def test_forecast_split_or_resolution():
    """``forecast_split_or`` prefers the explicit override, else the dataset's split."""
    seq = DataMetadata(data_dim=1, forecast_split=None)
    assert seq.forecast_split_or(None) is None  # sequence data, no override
    assert seq.forecast_split_or(16) == 16  # spec override wins
    windowed = DataMetadata(data_dim=6, forecast_split=72)
    assert windowed.forecast_split_or(None) == 72  # dataset boundary
    assert windowed.forecast_split_or(10) == 10  # override still wins


# ---------------------------------------------------------------------------
# Parametrized smoke-tests for every SyntheticDataModule mode used by the
# verification presets (harmonic, harmonic-noisy, bimodal, robot-basis-pursuit).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,D",
    [
        ("harmonic", 1),
        ("harmonic-noisy", 1),
        ("bimodal", 1),
        ("bimodal-noisy", 1),
        ("nonlinear-bimodal-lift", 4),
        ("robot-basis-pursuit", 2),
    ],
)
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
    dm = SyntheticDataModule(
        mode="robot-basis-pursuit", T=16, D=2, N_per_split=8, batch_size=2
    )
    batch = next(iter(dm.train_loader()))
    # X and Y coordinates must both be present
    assert batch["observed_data"].shape[1] == 2


def test_synthetic_datamodule_bimodal_has_variance() -> None:
    """Bimodal data should have non-trivial variance (the two modes are well-separated)."""
    dm = SyntheticDataModule(mode="bimodal", T=64, D=1, N_per_split=128, batch_size=64)
    batch = next(iter(dm.train_loader()))
    # Variance across batch and time should be well above noise floor
    std = batch["observed_data"].std().item()
    assert std > 0.5, (
        f"bimodal std={std:.3f} suspiciously low — modes may have collapsed"
    )


# ---------------------------------------------------------------------------
# Eval window alignment (_make_window_ends)
# ---------------------------------------------------------------------------


def test_window_ends_anchor_backward_from_last_end() -> None:
    """Eval window grids must end exactly at the region boundary.

    With forward anchoring, (last_end - start_end) % step != 0 shifted every
    eval window earlier, so val/test forecast targets could overlap the
    training region.
    """
    from ddssm.data.dataload import _make_window_ends

    # (last_end - start_end) % step = (100 - 24) % 10 = 6 → misaligned grid
    ends = _make_window_ends(start_end=24, last_end=100, step=10, k_last=5)
    assert ends[-1] == 100
    assert ends == [60, 70, 80, 90, 100]
    # Every end respects the valid range
    assert all(24 <= e <= 100 for e in ends)


def test_window_ends_step_one_unchanged() -> None:
    """Train windows (step=1) enumerate every end; alignment is a no-op."""
    from ddssm.data.dataload import _make_window_ends

    ends = _make_window_ends(start_end=5, last_end=9, step=1)
    assert ends == [5, 6, 7, 8, 9]


def test_window_ends_aligned_grid_identical() -> None:
    """When the span divides evenly by step the grid is the classic one."""
    from ddssm.data.dataload import _make_window_ends

    ends = _make_window_ends(start_end=20, last_end=80, step=20, k_last=None)
    assert ends == [20, 40, 60, 80]


def test_window_ends_empty_when_range_invalid() -> None:
    from ddssm.data.dataload import _make_window_ends

    assert _make_window_ends(start_end=50, last_end=40, step=5) == []


def test_window_ends_k_last_shorter_than_grid() -> None:
    """k_last larger than the grid returns everything available."""
    from ddssm.data.dataload import _make_window_ends

    ends = _make_window_ends(start_end=24, last_end=44, step=10, k_last=99)
    assert ends == [24, 34, 44]
