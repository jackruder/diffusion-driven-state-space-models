"""Per-dataset axis for the gluonts_forecast benchmark family.

Five NIPS GP-copula datasets (the CSDI / TimeGrad benchmark set). The library
data presets (:mod:`ddssm.data.presets`) own the actual fetch + windowing; the
``data_dim`` / ``L1`` / ``L2`` here feed the MODEL (its ``data_dim`` and
``T_max = L1 + L2``) and must mirror ``GluonTSDataModule.SPECS``. ``batch_size``
is a per-dataset starting default (swept); the wide datasets start smaller.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from ddssm.data.presets import Taxi, Wiki, Solar, Traffic, Electricity


@dataclass(frozen=True)
class GluonDataset:
    """One dataset point: the data preset + the dims the model needs."""

    label: str
    data_preset: Any
    data_dim: int
    L1: int
    L2: int
    batch_size: int

    @property
    def T_max(self) -> int:
        return self.L1 + self.L2


# data_dim / L1 / L2 mirror ``GluonTSDataModule.SPECS`` (loader = windowing
# source of truth). Series counts are the published GP-copula dims.
GLUONTS_DATASETS = [
    GluonDataset("solar", Solar, data_dim=137, L1=168, L2=24, batch_size=64),
    GluonDataset(
        "electricity", Electricity, data_dim=370, L1=168, L2=24, batch_size=64
    ),
    GluonDataset("traffic", Traffic, data_dim=963, L1=168, L2=24, batch_size=32),
    GluonDataset("taxi", Taxi, data_dim=1214, L1=48, L2=24, batch_size=32),
    GluonDataset("wiki", Wiki, data_dim=2000, L1=90, L2=30, batch_size=16),
]

GLUONTS_BY_LABEL = {d.label: d for d in GLUONTS_DATASETS}

__all__ = ["GLUONTS_BY_LABEL", "GLUONTS_DATASETS", "GluonDataset"]
