"""Parameterizable ``ModelAdapter`` contract harness.

Later family test modules (``test_ddssm.py``, ``test_csdi.py``) import
:class:`ModelAdapterContract`, subclass it, and fill the two abstract seams:

* :meth:`~ModelAdapterContract.make_adapter` — build a *fresh* concrete adapter.
* :meth:`~ModelAdapterContract.make_data` — build a fitted-against
  :class:`~ddssm.data.datamodule.TimeSeriesDataModule` (train + val splits).

The base class then runs the five family-agnostic contract checks below. Because
no concrete adapter exists yet at module 4, this file is **scaffolding only**:
it must import cleanly and expose the parameterization seam, but the two hooks
are abstract, so pytest collects the base class as producing no runnable tests
(there is nothing to instantiate it with). A subclass that implements both
hooks lights every check up automatically.

Contract checks
---------------
a. ``fit`` writes ``loss/total`` rows for BOTH ``train`` and ``val`` splits into
   ``csv_log_path`` (via ``MetricStore``).
b. ``save_checkpoint`` → ``load_checkpoint`` round-trips forecast-equivalent
   state (a reloaded adapter forecasts identically to the original).
c. Cross-format ``load_checkpoint`` raises ``ValueError``.
d. ``forecast`` returns ``{"pred_mean", "pred_samples"}`` with shapes
   ``(B, D, L2)`` / ``(B, S, D, L2)`` in NORMALIZED space.
e. ``data.train_loader() is None`` (``NullDataModule``) ⇒ ``fit`` no-ops.
"""

from __future__ import annotations

import os
import abc
import csv
from typing import TYPE_CHECKING

import torch
import pytest

if TYPE_CHECKING:  # annotation-only — keep the harness import-light / cycle-safe
    from pathlib import Path

    from ddssm.adapters import ModelAdapter
    from ddssm.data.datamodule import TimeSeriesDataModule
    from ddssm.experiment.experiment import TrainingScalars


def _read_csv_splits(csv_log_path: str) -> set[str]:
    """Return the distinct ``split`` values present in a MetricStore CSV.

    Small helper the ``fit`` checks share; kept module-level so a subclass can
    reuse it directly if it wants a bespoke assertion.
    """
    with open(csv_log_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "split" not in reader.fieldnames:
            return set()
        return {row["split"] for row in reader if row.get("split")}


def _has_metric_rows(csv_log_path: str, metric: str, split: str) -> bool:
    """Whether the CSV has ≥1 row for ``metric`` on ``split``."""
    with open(csv_log_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        for row in reader:
            if row.get("split") != split:
                continue
            # MetricStore may store metric-per-column or a (metric, value) pair;
            # accept either shape so the harness isn't coupled to one layout.
            if metric in fields and row.get(metric) not in (None, ""):
                return True
            if row.get("metric") == metric and row.get("value") not in (None, ""):
                return True
    return False


class ModelAdapterContract(abc.ABC):
    """Family-agnostic contract every concrete ``ModelAdapter`` must satisfy.

    Subclass in a family test module and implement :meth:`make_adapter` /
    :meth:`make_data`; the inherited ``test_*`` methods do the rest. The two
    hooks are the whole parameterization seam — no other coupling to a family.
    """

    # ---- parameterization seam (subclasses fill these) ----------------------

    @abc.abstractmethod
    def make_adapter(self) -> ModelAdapter:
        """Return a fresh, unfitted concrete adapter for this family."""
        raise NotImplementedError

    @abc.abstractmethod
    def make_data(self) -> TimeSeriesDataModule:
        """Return a small DataModule with real ``train``/``val`` splits."""
        raise NotImplementedError

    def make_null_data(self) -> TimeSeriesDataModule:
        """Return a ``NullDataModule`` (``train_loader() is None``) for check (e).

        Default uses the shipped ``NullDataModule``; a family may override if it
        needs a differently-shaped no-data module.
        """
        from ddssm.data.datamodule import NullDataModule

        return NullDataModule()

    # ---- shared fit config the checks reuse ---------------------------------

    def _training(self) -> TrainingScalars:
        """A tiny ``TrainingScalars`` so contract fits stay fast.

        Imported lazily (inside the method) to keep this harness free of any
        module-level ``ddssm.experiment`` import at collection time.
        """
        from ddssm.experiment.experiment import TrainingScalars

        return TrainingScalars(steps=2, log_every=1, validate_every=1)

    def _fit(
        self, adapter: ModelAdapter, data: TimeSeriesDataModule, tmp_path: Path
    ) -> str:
        """Run a short ``fit`` into ``tmp_path`` and return the CSV path."""
        csv_log_path = str(tmp_path / "metrics.csv")
        adapter.fit(
            data=data,
            training=self._training(),
            device=torch.device("cpu"),
            csv_log_path=csv_log_path,
            tensorboard_dir=str(tmp_path / "tb"),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        return csv_log_path

    # ---- contract checks (a)-(e) --------------------------------------------

    def test_fit_writes_loss_total_for_train_and_val(self, tmp_path: Path) -> None:
        """(a) ``fit`` logs ``loss/total`` rows for train AND val splits."""
        adapter = self.make_adapter()
        csv_log_path = self._fit(adapter, self.make_data(), tmp_path)
        assert _has_metric_rows(csv_log_path, "loss/total", "train")
        assert _has_metric_rows(csv_log_path, "loss/total", "val")

    def test_checkpoint_roundtrip_preserves_forecast(self, tmp_path: Path) -> None:
        """(b) save → load restores forecast-equivalent state."""
        adapter = self.make_adapter()
        data = self.make_data()
        self._fit(adapter, data, tmp_path)
        batch = self._forecast_inputs(data)
        before = adapter.forecast(**batch, num_samples=4)

        ckpt = str(tmp_path / "adapter.pth")
        adapter.save_checkpoint(ckpt)
        reloaded = self.make_adapter()
        reloaded.load_checkpoint(ckpt, device=torch.device("cpu"))
        after = reloaded.forecast(**batch, num_samples=4)

        torch.testing.assert_close(before["pred_mean"], after["pred_mean"])

    def test_cross_format_load_raises_value_error(self, tmp_path: Path) -> None:
        """(c) loading a foreign-format checkpoint raises ``ValueError``."""
        bogus = tmp_path / "foreign.pth"
        torch.save({"__foreign_format__": True}, bogus)
        adapter = self.make_adapter()
        with pytest.raises(ValueError):
            adapter.load_checkpoint(str(bogus), device=torch.device("cpu"))

    def test_forecast_shapes_and_normalized_space(self, tmp_path: Path) -> None:
        """(d) forecast returns the canonical dict with correct shapes."""
        adapter = self.make_adapter()
        data = self.make_data()
        self._fit(adapter, data, tmp_path)
        batch = self._forecast_inputs(data)
        num_samples = 4
        out = adapter.forecast(**batch, num_samples=num_samples)

        assert set(out) >= {"pred_mean", "pred_samples"}
        mean, samples = out["pred_mean"], out["pred_samples"]
        assert mean.ndim == 3  # (B, D, L2)
        assert samples.ndim == 4  # (B, S, D, L2)
        assert samples.shape[1] == num_samples
        assert samples.shape[0] == mean.shape[0]  # B
        assert samples.shape[2] == mean.shape[1]  # D
        assert samples.shape[3] == mean.shape[2]  # L2

    def test_null_data_fit_noops(self, tmp_path: Path) -> None:
        """(e) ``NullDataModule`` (``train_loader() is None``) ⇒ fit no-ops."""
        adapter = self.make_adapter()
        # Must not raise and must not write a metrics CSV with train rows.
        csv_log_path = self._fit(adapter, self.make_null_data(), tmp_path)
        assert (not os.path.exists(csv_log_path)) or not _has_metric_rows(
            csv_log_path, "loss/total", "train"
        )

    # ---- helper a subclass supplies inputs through --------------------------

    @abc.abstractmethod
    def _forecast_inputs(self, data: TimeSeriesDataModule) -> dict:
        """Build the kwargs dict passed to :meth:`ModelAdapter.forecast`.

        Returns everything except ``num_samples`` (the checks add that). Kept
        abstract because the exact history/mask/covariate tensors are
        family- and dataset-specific.
        """
        raise NotImplementedError
