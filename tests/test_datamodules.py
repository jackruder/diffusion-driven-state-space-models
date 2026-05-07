"""Smoke tests for the DataModule layer.

Ensure each concrete DataModule yields the canonical model-ready dict
that the trainer expects (``observed_data``, ``observation_mask``,
``timepoints``, optional ``covariates`` / ``static_covariates``), and
that ``parse_batch`` is a no-op pass-through on those batches.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from ddssm.data.datamodule import (
    DataMetadata,
    DDSSMDataModule,
    KDDDataModule,
    NullDataModule,
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
