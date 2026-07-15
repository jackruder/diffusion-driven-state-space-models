"""In-memory WINDOWED smoke data for the CSDI family.

The CSDI adapter (:class:`ddssm.adapters.csdi.CSDIAdapter`) requires a WINDOWED
data module: ``metadata.forecast_split`` must be an int (== L1), not ``None``.
The synthetic / null modules leave it unset, so the smoke cannot reuse them.

There is no offline GluonTS cache and the KDD ``.pt`` blobs are large + local, so
the smoke uses an in-memory synthetic windowed generator instead. The dataset
emits the canonical model-ready dict (the exact shape
:func:`ddssm.data.dataload.parse_batch` consumes) so raw batches flow through the
production ``batch_transform``. This mirrors the PROVEN template in
``tests/adapters/test_csdi.py`` (``_TinyWindowDataset`` /
``TinyWindowedDataModule``) but lives here as a real, importable data preset so
ALL CSDI experiment code stays inside ``experiments/csdi/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from hydra_zen import builds
from torch.utils.data import Dataset, DataLoader

from ddssm.data.dataload import parse_batch
from ddssm.data.datamodule import DataMetadata, TimeSeriesDataModule

if TYPE_CHECKING:
    pass


class _SmokeWindowDataset(Dataset):
    """Emits one ``(D, T)`` past+future window per item (the model-ready dict).

    Mirrors ``_GroupedWindowDataset`` / the tested ``_TinyWindowDataset``: an
    all-ones observation mask and local ``0..T-1`` timepoints. Deterministic
    given ``seed`` so runs are reproducible.
    """

    def __init__(self, n: int, d: int, t: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randn(n, d, t, generator=g)
        self.t = t

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {
            "observed_data": self.data[i],
            "observation_mask": torch.ones_like(self.data[i]),
            "timepoints": torch.arange(self.t, dtype=torch.float32),
            "covariates": None,
            "static_covariates": None,
        }


def _collate(batch: list[dict]) -> dict:
    keys = ["observed_data", "observation_mask", "timepoints"]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["covariates"] = None
    out["static_covariates"] = None
    return out


class SmokeWindowedDataModule(TimeSeriesDataModule):
    """Tiny in-memory windowed DataModule with ``metadata.forecast_split = L1``.

    The cheapest object satisfying the CSDI adapter's windowed requirement. Uses
    the shipped ``parse_batch`` as its ``batch_transform`` so raw batches flow
    through the exact production transform. ``batch_size`` is a mutable attribute
    (read by each loader at call time) so ``Experiment.train`` can reconcile it
    from ``hparams.batch_size`` — the loader source of truth.
    """

    batch_format = "windowed"
    batch_transform = staticmethod(parse_batch)

    def __init__(
        self,
        n: int = 48,
        d: int = 2,
        l1: int = 8,
        l2: int = 4,
        batch_size: int = 8,
        seed: int = 0,
    ) -> None:
        """Build a tiny windowed loader set with ``forecast_split = l1``."""
        self.n, self.d, self.l1, self.l2 = n, d, l1, l2
        self.t = l1 + l2
        self.batch_size = batch_size
        # Distinct seeds so train / val / test are not the identical windows.
        self._train_ds = _SmokeWindowDataset(n, d, self.t, seed=seed)
        self._val_ds = _SmokeWindowDataset(max(d * 4, 8), d, self.t, seed=seed + 1)
        self._test_ds = _SmokeWindowDataset(max(d * 4, 8), d, self.t, seed=seed + 2)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
        )

    def train_loader(self) -> DataLoader:
        """Shuffled train loader over the tiny window set."""
        return self._loader(self._train_ds, shuffle=True)

    def val_loader(self) -> DataLoader:
        """Deterministic val loader (held-out windows, no shuffle)."""
        return self._loader(self._val_ds, shuffle=False)

    def test_loader(self) -> DataLoader:
        """Deterministic test loader (held-out windows, no shuffle)."""
        return self._loader(self._test_ds, shuffle=False)

    @property
    def metadata(self) -> DataMetadata:
        """Windowed metadata whose ``forecast_split`` is ``L1`` (not None)."""
        return DataMetadata(
            data_dim=self.d,
            covariate_dim=0,
            T=self.t,
            use_observation_mask=True,
            forecast_split=self.l1,
        )


# hydra-zen config for the smoke data axis (dims tiny for a fast CPU fit).
SmokeData = builds(SmokeWindowedDataModule, populate_full_signature=True)


__all__ = ["SmokeData", "SmokeWindowedDataModule"]
