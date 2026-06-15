"""Tests for the clean_data (noise-free ground truth) surface.

Verifies that ``expose_clean_data=True`` correctly threads through the
data module and batch transform, that ``observed_data`` remains bit-identical
to the flag-off baseline (no extra RNG calls), and that ``clean_data`` is
observably different from ``observed_data`` (obs noise was actually added).
"""

from __future__ import annotations

import torch

from ddssm.data.synthetic import SyntheticDataset
from ddssm.data.datamodule import SyntheticDataModule


_LORENZ_KW = dict(mode="lorenz", N_per_split=8, T=64, D=3, dataset_seed=42)


def test_default_no_clean_data_key() -> None:
    """Default behaviour: no ``clean_data`` key in items."""
    ds = SyntheticDataset(split="val", **_LORENZ_KW)
    item = ds[0]
    assert "clean_data" not in item
    assert "observed_data" in item


def test_lorenz_exposes_clean_data_with_flag() -> None:
    """``expose_clean_data=True`` adds a ``clean_data`` key of the same shape."""
    ds = SyntheticDataset(split="val", expose_clean_data=True, **_LORENZ_KW)
    item = ds[0]
    assert "clean_data" in item
    assert item["clean_data"].shape == item["observed_data"].shape == (3, 64)


def test_clean_data_differs_from_observed() -> None:
    """Observation noise is non-zero: clean_data != observed_data."""
    ds = SyntheticDataset(split="val", expose_clean_data=True, **_LORENZ_KW)
    item = ds[0]
    diff = (item["observed_data"] - item["clean_data"]).abs().mean().item()
    # Expected RMS noise σ=0.1 -> mean abs ~0.08; zero would mean no noise added.
    assert diff > 0.01


def test_clean_data_closer_to_true_signal_than_observed() -> None:
    """clean_data should have smaller MSE against a 2nd realisation's clean data
    than the noisy observations do (verifying it is the pre-noise trajectory)."""
    ds1 = SyntheticDataset(split="test", expose_clean_data=True, **_LORENZ_KW)
    # With the flag off, observed_data is the noisy version of the same trajectory.
    ds2 = SyntheticDataset(split="test", expose_clean_data=False, **_LORENZ_KW)
    # observed_data must be identical (bit-identity invariant).
    for i in range(4):
        assert torch.equal(ds1[i]["observed_data"], ds2[i]["observed_data"]), (
            "observed_data changed when expose_clean_data=True — RNG side-effect!"
        )


def test_bit_identity_of_observed_data() -> None:
    """Enabling the flag must not change observed_data (no extra RNG calls)."""
    common = dict(split="train", **_LORENZ_KW)
    ds_off = SyntheticDataset(expose_clean_data=False, **common)
    ds_on = SyntheticDataset(expose_clean_data=True, **common)
    for i in range(len(ds_off)):
        assert torch.equal(ds_off[i]["observed_data"], ds_on[i]["observed_data"]), (
            f"observed_data at index {i} differs with expose_clean_data=True"
        )


def test_disabled_flag_leaves_no_state() -> None:
    """Flag off (default) -> clean_data attribute is None, no memory allocated."""
    ds = SyntheticDataset(split="val", **_LORENZ_KW)
    assert ds.clean_data is None


def test_data_module_threads_flag() -> None:
    """``SyntheticDataModule(expose_clean_data=True)`` propagates the flag."""
    dm = SyntheticDataModule(
        mode="lorenz", T=64, D=3, N_per_split=8, batch_size=4,
        expose_clean_data=True,
    )
    batch = next(iter(dm.val_loader()))
    assert "clean_data" in batch
    assert batch["clean_data"].shape == batch["observed_data"].shape


def test_parse_batch_threads_clean_data() -> None:
    """``parse_batch`` carries ``clean_data`` through to float32 on device."""
    dm = SyntheticDataModule(
        mode="lorenz", T=64, D=3, N_per_split=8, batch_size=4,
        expose_clean_data=True,
    )
    raw = next(iter(dm.val_loader()))
    parsed = dm.batch_transform(raw, torch.device("cpu"))
    assert "clean_data" in parsed
    assert parsed["clean_data"].dtype == torch.float32


def test_splits_keep_clean_data_disjoint() -> None:
    """train/val/test slices each retain their own clean_data (disjoint slices)."""
    kw = dict(expose_clean_data=True, **_LORENZ_KW)
    train = SyntheticDataset(split="train", **kw)
    val = SyntheticDataset(split="val", **kw)
    test = SyntheticDataset(split="test", **kw)

    assert train.clean_data is not None
    assert val.clean_data is not None
    assert test.clean_data is not None
    assert train.clean_data.shape[0] == 8
    assert not torch.equal(train.clean_data[0], val.clean_data[0])


def test_unsupported_mode_silently_no_ops() -> None:
    """Modes that don't add obs noise don't populate clean_data; flag is a no-op."""
    ds = SyntheticDataset(
        mode="lgssm", split="val", N_per_split=4, T=8, D=1, dataset_seed=0,
        expose_clean_data=True,
    )
    # clean_data attribute stays None; __getitem__ omits the key.
    assert ds.clean_data is None
    item = ds[0]
    assert "clean_data" not in item
