"""Smoke tests for :class:`MocapDataModule`.

Mirrors the shape assertions in :mod:`test_datamodules` — the mocap
DataModule advertises the sequence batch format and must yield the
canonical ``observed_data``/``observation_mask``/``timepoints`` dict.

The on-disk artifact is ``data/mocap35.mat`` (fetched on first
construction of a ``download=True`` DataModule). Tests skip cleanly
when it's absent so CI without the file still passes.
"""

from __future__ import annotations

import os

import torch
import pytest

from ddssm.data.mocap import (
    MocapDataset,
    MocapDataModule,
    _ensure_mocap35,
)
from ddssm.data.dataload import parse_batch
from ddssm.data.datamodule import DataMetadata, TimeSeriesDataModule

MOCAP_PATH = "data/mocap35.mat"
_HAS_MOCAP = os.path.isfile(MOCAP_PATH) and os.path.getsize(MOCAP_PATH) > 1024


def _assert_canonical_batch(batch: dict) -> None:
    for key in ("observed_data", "observation_mask", "timepoints"):
        assert key in batch, f"missing key: {key}"
        assert isinstance(batch[key], torch.Tensor)
    assert batch["observed_data"].shape == batch["observation_mask"].shape
    assert batch["observed_data"].dim() == 3  # (B, D, T)


def test_mocap_module_is_datamodule_and_lazy(tmp_path):
    """Construction touches no disk — no I/O until a loader / metadata is asked for."""
    # Point at a nonexistent path with download=False; init must still succeed.
    dm = MocapDataModule(
        filepath=str(tmp_path / "no-such.mat"),
        download=False,
        batch_size=2,
    )
    assert isinstance(dm, TimeSeriesDataModule)
    assert dm.batch_format == "sequence"
    assert dm.batch_transform is parse_batch
    assert dm._built is False


def test_ensure_mocap_noops_when_present(tmp_path):
    """When the file already exists, no network I/O is attempted."""
    target = tmp_path / "mocap35.mat"
    target.write_bytes(b"placeholder-not-a-real-mat")
    # A bogus URL — if the helper tries to download, this raises.
    _ensure_mocap35(str(target), url="http://invalid.invalid/never-hit")
    assert target.read_bytes() == b"placeholder-not-a-real-mat"


def test_loader_access_raises_when_missing_and_download_disabled(tmp_path):
    """First loader access surfaces a clear FileNotFoundError, not a scipy crash."""
    missing = tmp_path / "no-such.mat"
    dm = MocapDataModule(filepath=str(missing), download=False)
    with pytest.raises(FileNotFoundError):
        dm.train_loader()


@pytest.mark.skipif(
    not _HAS_MOCAP,
    reason=f"{MOCAP_PATH} missing (run the DataModule once to download).",
)
def test_mocap_datamodule_smoke():
    dm = MocapDataModule(filepath=MOCAP_PATH, download=False, batch_size=2)

    meta = dm.metadata
    assert isinstance(meta, DataMetadata)
    assert meta.data_dim == 50
    assert meta.T == 300
    assert meta.covariate_dim == 0
    assert meta.means is not None
    assert meta.stds is not None
    assert meta.means.shape == (50,)
    assert meta.stds.shape == (50,)
    assert torch.all(meta.stds > 0)

    train = dm.train_loader()
    val = dm.val_loader()
    test = dm.test_loader()
    assert train is not None and val is not None and test is not None

    # Split sizes are the canonical Wang-2007 / Gan-2015 benchmark split.
    assert len(train.dataset) == 16
    assert len(val.dataset) == 3
    assert len(test.dataset) == 4

    batch = next(iter(train))
    _assert_canonical_batch(batch)
    assert batch["observed_data"].shape == (2, 50, 300)
    # parse_batch passthrough on canonical dicts.
    parsed = dm.batch_transform(batch, torch.device("cpu"))
    for key in ("observed_data", "observation_mask", "timepoints"):
        assert torch.equal(parsed[key], batch[key].to(torch.float32))


@pytest.mark.skipif(not _HAS_MOCAP, reason=f"{MOCAP_PATH} missing.")
def test_mocap_normalization_zero_centered_train():
    """With ``normalize=True``, non-degenerate features are zero-mean unit-std
    on the train set. Constant joint axes (some mocap features never move) are
    left in raw units — their std stays at zero, mean stays at zero.
    """
    dm = MocapDataModule(filepath=MOCAP_PATH, download=False, normalize=True)
    train = dm.train_loader()
    # (50, 16*300) — per-feature stream over the whole train set.
    flat = torch.cat(
        [item["observed_data"] for item in train.dataset], dim=-1
    )
    means = flat.mean(dim=1)
    stds = flat.std(dim=1)

    non_deg = stds > 1e-3
    assert non_deg.sum() > 40, "expected most features to be non-degenerate"
    # Non-degenerate features are z-scored.
    assert torch.allclose(means[non_deg], torch.zeros_like(means[non_deg]), atol=1e-4)
    assert torch.allclose(stds[non_deg], torch.ones_like(stds[non_deg]), atol=1e-3)
    # Constant features stay centered at zero (mean subtracted, unit divisor).
    assert torch.allclose(means[~non_deg], torch.zeros_like(means[~non_deg]), atol=1e-4)


@pytest.mark.skipif(not _HAS_MOCAP, reason=f"{MOCAP_PATH} missing.")
def test_mocap_dataset_item_shapes():
    """Per-item shape is ``(D, T) = (50, 300)`` — the sequence-format contract."""
    dm = MocapDataModule(filepath=MOCAP_PATH, download=False)
    item = dm.train_loader().dataset[0]
    assert isinstance(item, dict)
    assert item["observed_data"].shape == (50, 300)
    assert item["observation_mask"].shape == (50, 300)
    assert torch.all(item["observation_mask"] == 1.0)
    assert item["timepoints"].shape == (300,)


@pytest.mark.skipif(not _HAS_MOCAP, reason=f"{MOCAP_PATH} missing.")
def test_mocap_preset_instantiates():
    """The registered Hydra preset builds the right DataModule type."""
    from hydra_zen import instantiate

    from ddssm.data import presets

    dm = instantiate(presets.Mocap35, download=False, filepath=MOCAP_PATH)
    assert isinstance(dm, MocapDataModule)


def test_mocap_dataset_direct_construction():
    """``MocapDataset`` yields the canonical dict from raw tensors."""
    X = torch.randn(4, 50, 300)
    tp = torch.arange(300, dtype=torch.float32) * 0.1
    ds = MocapDataset(X, tp)
    assert len(ds) == 4
    item = ds[0]
    assert item["observed_data"].shape == (50, 300)
    assert torch.all(item["observation_mask"] == 1.0)
    assert torch.equal(item["timepoints"], tp)
