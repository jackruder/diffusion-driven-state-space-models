"""Tests for CSV logging robustness and per-split meter specs."""

from __future__ import annotations

import csv
from pathlib import Path

from ddssm.loggers import CSVLogger, MetricStore, MetricSpec


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def test_csv_logger_writes_step_and_epoch_rows(tmp_path: Path):
    """Both on_step (train) and on_epoch (val) rows reach the file."""
    path = tmp_path / "metrics.csv"
    lg = CSVLogger(str(path))
    lg.on_step("train", 1, {"loss/total": 1.0})
    lg.on_epoch("val", 1, {"loss/total": 2.0})

    _fields, rows = _read(path)
    splits = {r["split"] for r in rows}
    assert splits == {"train", "val"}
    val_row = next(r for r in rows if r["split"] == "val")
    assert float(val_row["loss/total"]) == 2.0


def test_csv_logger_widens_header_for_new_keys(tmp_path: Path):
    """A row introducing a new column rewrites the header; old rows keep restval."""
    path = tmp_path / "metrics.csv"
    lg = CSVLogger(str(path))
    lg.on_step("train", 1, {"loss/total": 1.0})
    # New column appears (e.g. a stage boundary adds a transition sub-term).
    lg.on_step("train", 2, {"loss/total": 0.9, "loss/rate/trans/new": 0.1})

    fields, rows = _read(path)
    assert "loss/rate/trans/new" in fields
    # First row predates the column -> filled with restval ("").
    assert rows[0]["loss/rate/trans/new"] == ""
    assert float(rows[1]["loss/rate/trans/new"]) == 0.1
    # No misalignment: loss/total stays correct for both rows.
    assert [float(r["loss/total"]) for r in rows] == [1.0, 0.9]


def test_csv_logger_fills_missing_keys_with_restval(tmp_path: Path):
    """A later row missing an established column doesn't shift other columns."""
    path = tmp_path / "metrics.csv"
    lg = CSVLogger(str(path))
    lg.on_step("train", 1, {"loss/total": 1.0, "time/elapsed_s": 0.5})
    lg.on_epoch("val", 1, {"loss/total": 2.0})  # no time/elapsed_s

    _fields, rows = _read(path)
    val_row = next(r for r in rows if r["split"] == "val")
    assert val_row["time/elapsed_s"] == ""
    assert float(val_row["loss/total"]) == 2.0


def test_metric_store_counts_nonfinite(caplog):
    """NaN/Inf metrics are counted into nonfinite/total and warned (deduped)."""
    import logging

    store = MetricStore(spec=[MetricSpec("loss/*", "last")], loggers=[])
    with caplog.at_level(logging.WARNING, logger="ddssm.loggers"):
        store.update("train", {"loss/total": float("nan")})
        row = store.step_end("train", 1)
    assert row["nonfinite/total"] == 1.0
    assert "Non-finite" in caplog.text

    # A finite step doesn't increment, and a healthy row carries the 0-baseline.
    store2 = MetricStore(spec=[MetricSpec("loss/*", "last")], loggers=[])
    store2.update("train", {"loss/total": 1.0})
    assert store2.step_end("train", 1)["nonfinite/total"] == 0.0


def test_metric_store_val_uses_mean_meter_train_uses_last(tmp_path: Path):
    """split_spec gives val mean meters (set-average) while train stays 'last'."""
    store = MetricStore(
        spec=[MetricSpec("loss/*", "last")],
        split_spec={"val": [MetricSpec("loss/*", "mean")]},
        loggers=[],
    )
    # Two val batches with different losses, weighted by batch size.
    store.update("val", {"loss/total": 1.0}, weight=2.0)
    store.update("val", {"loss/total": 3.0}, weight=2.0)
    val_row = store.epoch_end("val", 1)
    assert val_row["loss/total"] == 2.0  # weighted mean, not last (3.0)

    # Train keeps last-value semantics.
    store.update("train", {"loss/total": 1.0})
    store.update("train", {"loss/total": 5.0})
    train_row = store.step_end("train", 1)
    assert train_row["loss/total"] == 5.0
