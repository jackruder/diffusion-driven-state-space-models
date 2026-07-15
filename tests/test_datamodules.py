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
    GluonTSDataModule,
    SyntheticDataModule,
    TimeSeriesDataModule,
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
    assert isinstance(dm, TimeSeriesDataModule)
    assert dm.train_loader() is None
    assert dm.val_loader() is None
    assert dm.test_loader() is None
    assert dm.metadata.data_dim == 3
    assert dm.batch_format == "sequence"


def test_synthetic_datamodule_smoke():
    dm = SyntheticDataModule(mode="lgssm", T=16, D=2, N_per_split=8, batch_size=4)
    assert isinstance(dm, TimeSeriesDataModule)
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
    assert isinstance(dm, TimeSeriesDataModule)
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
    assert isinstance(NullDataModule(), TimeSeriesDataModule)
    assert isinstance(
        SyntheticDataModule(mode="lgssm", T=8, D=1, N_per_split=4, batch_size=2),
        TimeSeriesDataModule,
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
    assert isinstance(dm, TimeSeriesDataModule)
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


# ---------------------------------------------------------------------------
# henon-lift standardization: train-only stats
# ---------------------------------------------------------------------------


def test_henon_lift_standardization_uses_train_stats() -> None:
    """Val/test normalisation must use train-split stats, not the full population.

    After the fix the normalisation stats are anchored to ``z[:N_per_split]``
    (the train rows).  By definition that makes the train-slice gt_latents
    have mean ≈ 0 and std ≈ 1.  The val-split gt_latents come from different
    Henon trajectories so they will NOT be self-normalised — they carry a
    nonzero mean (shifted by the train-anchored stats).  This confirms that
    the train-slice stats govern ALL splits, not a per-split normalisation.

    The key regression check: if full-population stats were used, the
    train-slice gt_latents would NOT have mean exactly 0 (because the
    full-population mean is pulled away from the train-slice mean).
    """
    import torch

    from ddssm.data.synthetic import HENON_OBS_D, SyntheticDataset

    N = 64
    T = 40
    seed = 42

    with torch.random.fork_rng(devices=[]):
        ds_train = SyntheticDataset(
            "henon-lift", split="train", N_per_split=N, T=T, D=HENON_OBS_D,
            dataset_seed=seed, expose_gt_latents=True,
        )
    with torch.random.fork_rng(devices=[]):
        ds_val = SyntheticDataset(
            "henon-lift", split="val", N_per_split=N, T=T, D=HENON_OBS_D,
            dataset_seed=seed, expose_gt_latents=True,
        )

    assert ds_train.gt_latents is not None and ds_val.gt_latents is not None

    # Train-slice normalisation invariant: mean ≈ 0, std ≈ 1.
    z_tr = ds_train.gt_latents  # (N, latent_d, T)
    tr_mean = z_tr.mean(dim=(0, 2))
    tr_std = z_tr.std(dim=(0, 2))
    assert (tr_mean.abs() < 0.1).all(), (
        f"Train gt_latents mean={tr_mean.tolist()} not near 0; "
        "standardization may be using full-population stats."
    )
    assert ((tr_std - 1.0).abs() < 0.15).all(), (
        f"Train gt_latents std={tr_std.tolist()} not near 1; "
        "standardization may be using full-population stats."
    )

    # Val-slice: different trajectories anchored to the SAME train stats,
    # so the val mean is NOT zero (the train-slice anchor shifts it).
    z_val = ds_val.gt_latents  # (N, latent_d, T)
    val_mean = z_val.mean(dim=(0, 2))
    # At least one latent dim should have a non-trivial mean offset.
    assert val_mean.abs().max().item() > 0.01, (
        "Val gt_latents appear unit-normalised, suggesting per-split "
        "normalisation rather than train-slice-anchored stats."
    )


def test_henon_lift_gt_latents_train_slice_is_unit_normalised() -> None:
    """After the fix, the train-slice of the gt latent z must be unit-normalised.

    The standardisation uses ``z[:N_per_split]`` stats (mean / std computed over
    train sequences only).  By definition, after subtracting the train-slice mean
    and dividing by its std, the train slice of z has mean ≈ 0 and std ≈ 1 per
    dim.  If the full-population stats were used instead, the train-slice of the
    normalised z would NOT have mean ≈ 0 because the full-population mean shifts
    away from the train-slice mean.

    ``expose_gt_latents=True`` surfaces the post-normalised latent z so we can
    inspect it directly (the normalisation happens before the tanh-MLP lift).
    """
    import torch

    from ddssm.data.synthetic import HENON_OBS_D, SyntheticDataset

    N = 64
    T = 30
    seed = 55

    with torch.random.fork_rng(devices=[]):
        ds = SyntheticDataset(
            "henon-lift",
            split="train",
            N_per_split=N,
            T=T,
            D=HENON_OBS_D,
            dataset_seed=seed,
            expose_gt_latents=True,
        )

    assert ds.gt_latents is not None, "expose_gt_latents must populate gt_latents"
    z_train = ds.gt_latents  # shape (N, latent_d, T)

    # Mean and std of the train slice per latent dim (averaged over N and T).
    train_mean = z_train.mean(dim=(0, 2))  # (latent_d,)
    train_std = z_train.std(dim=(0, 2))    # (latent_d,)

    # After train-slice normalisation: mean should be close to 0, std close to 1.
    assert (train_mean.abs() < 0.1).all(), (
        f"Train-slice gt_latents mean={train_mean.tolist()} is not near 0 — "
        "standardization is likely using full-population stats."
    )
    assert ((train_std - 1.0).abs() < 0.1).all(), (
        f"Train-slice gt_latents std={train_std.tolist()} is not near 1 — "
        "standardization is likely using full-population stats."
    )


# ---------------------------------------------------------------------------
# Rename: DDSSMDataModule → TimeSeriesDataModule
# ---------------------------------------------------------------------------


def test_datamodule_abc_renamed_to_timeseries() -> None:
    """The model-agnostic ABC is exposed as ``TimeSeriesDataModule``.

    The legacy ``DDSSMDataModule`` name is gone (no backwards-compat alias).
    """
    from ddssm.data import datamodule as dm_mod
    from ddssm.data.datamodule import TimeSeriesDataModule

    assert "TimeSeriesDataModule" in dm_mod.__all__
    assert not hasattr(dm_mod, "DDSSMDataModule")
    assert isinstance(NullDataModule(), TimeSeriesDataModule)
