"""Tests for the GT-latent surface on :class:`SyntheticDataset`.

Verifies that ``expose_gt_latents=True`` correctly threads through the
data module and exposes the underlying clean latent ``z`` to consumers
(used by ``crps_sum_latent`` and ``gt_latent_jsd`` metrics).
"""

from __future__ import annotations

import pytest
import torch

from ddssm.data.datamodule import SyntheticDataModule
from ddssm.data.synthetic import SyntheticDataset


def test_legacy_no_gt_latent_field_by_default() -> None:
    """Default behaviour: no ``gt_latent`` key in items."""
    ds = SyntheticDataset(
        mode="lgssm", split="val", N_per_split=4, T=8, D=1, dataset_seed=0,
    )
    item = ds[0]
    assert "gt_latent" not in item
    assert "observed_data" in item


def test_lgssm_exposes_gt_latents() -> None:
    """LGSSM with the flag on adds ``gt_latent`` of matching shape."""
    ds = SyntheticDataset(
        mode="lgssm",
        split="val",
        N_per_split=4,
        T=8,
        D=1,
        dataset_seed=0,
        expose_gt_latents=True,
    )
    item = ds[0]
    assert "gt_latent" in item
    assert item["gt_latent"].shape == item["observed_data"].shape
    # Under LGSSM data = z + noise, so the two should differ.
    diff = (item["observed_data"] - item["gt_latent"]).abs().mean().item()
    assert diff > 0.0


def test_other_modes_dont_expose_gt_latent_even_with_flag() -> None:
    """Non-LGSSM modes return no ``gt_latent`` even when the flag is on.

    Initial coverage is LGSSM-only.  Other modes opt-in over time.
    """
    ds = SyntheticDataset(
        mode="harmonic",
        split="val",
        N_per_split=4,
        T=8,
        D=1,
        dataset_seed=0,
        expose_gt_latents=True,
    )
    item = ds[0]
    assert "gt_latent" not in item


def test_data_module_threads_flag_to_dataset() -> None:
    """``SyntheticDataModule(expose_gt_latents=True)`` propagates the flag."""
    dm = SyntheticDataModule(
        mode="lgssm", T=8, D=1, N_per_split=4, batch_size=2, expose_gt_latents=True,
    )
    batch = next(iter(dm.val_loader()))
    assert "gt_latent" in batch
    assert batch["gt_latent"].shape == batch["observed_data"].shape


def test_parse_batch_threads_gt_latent_to_device() -> None:
    """``parse_batch`` (the data module's transform) carries ``gt_latent`` through."""
    dm = SyntheticDataModule(
        mode="lgssm", T=8, D=1, N_per_split=4, batch_size=2, expose_gt_latents=True,
    )
    raw_batch = next(iter(dm.val_loader()))
    parsed = dm.batch_transform(raw_batch, torch.device("cpu"))
    assert "gt_latent" in parsed
    assert parsed["gt_latent"].dtype == torch.float32


def test_dataset_splits_keep_gt_latents_disjoint() -> None:
    """train/val/test slices each retain their own gt_latents."""
    common = dict(mode="lgssm", N_per_split=4, T=8, D=1, dataset_seed=0,
                  expose_gt_latents=True)
    train = SyntheticDataset(split="train", **common)
    val = SyntheticDataset(split="val", **common)
    test = SyntheticDataset(split="test", **common)

    assert train.gt_latents is not None
    assert val.gt_latents is not None
    assert test.gt_latents is not None
    # Each split has N_per_split sequences.
    assert train.gt_latents.shape[0] == 4
    assert val.gt_latents.shape[0] == 4
    assert test.gt_latents.shape[0] == 4
    # They should be different sequences (drawn from different slice).
    assert not torch.equal(train.gt_latents[0], val.gt_latents[0])


def test_disabled_flag_preserves_legacy_behavior() -> None:
    """``expose_gt_latents=False`` (default) ⇒ no GT-latent state stored."""
    ds = SyntheticDataset(
        mode="lgssm", split="val", N_per_split=4, T=8, D=1, dataset_seed=0,
    )
    assert ds.gt_latents is None
    item = ds[0]
    assert "gt_latent" not in item
