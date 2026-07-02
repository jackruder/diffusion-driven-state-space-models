"""Regression tests for :class:`ddssm.training.loggers.CSVLogger` schema-drift handling.

The model emits metric keys conditionally (stage-1 vs stage-2, train vs val,
λ-warmup vs steady-state). A naive append-with-fixed-header writer misaligns
columns when a new key appears or an existing key is omitted; downstream
``csv.DictReader`` consumers (Optuna objectives, eval metrics, plots) then
silently read the wrong column. These tests pin the superset-header behaviour.
"""

from __future__ import annotations

import csv
import os
import unittest.mock as mock

from ddssm.training.loggers import CSVLogger


def _read_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def test_new_keys_trigger_header_rewrite_without_misalignment(tmp_path):
    """A second row with a new key must not misalign existing columns."""
    path = str(tmp_path / "metrics.csv")
    logger = CSVLogger(path)

    logger.on_step("train", 1, {"A": 1.0, "B": 2.0, "C": 3.0})
    logger.on_step("train", 2, {"A": 10.0, "B": 20.0, "D": 40.0})  # D is new, C dropped

    fieldnames, rows = _read_rows(path)
    assert fieldnames == ["split", "step", "A", "B", "C", "D"], fieldnames
    assert len(rows) == 2
    # Row 1: original values present, D padded.
    assert rows[0]["A"] == "1.0"
    assert rows[0]["B"] == "2.0"
    assert rows[0]["C"] == "3.0"
    assert rows[0]["D"] == ""
    # Row 2: A/B/D present, C is missing -> empty.
    assert rows[1]["A"] == "10.0"
    assert rows[1]["B"] == "20.0"
    assert rows[1]["C"] == ""
    assert rows[1]["D"] == "40.0"


def test_split_misalignment_train_then_val(tmp_path):
    """Stage-1-style failure: train row carries ``optim/lambda``; val row doesn't.

    The val row's values must land under the correct keys, not under the
    ``optim/lambda`` column that train owns.
    """
    path = str(tmp_path / "metrics.csv")
    logger = CSVLogger(path)
    logger.on_step("train", 1, {"loss/total": 0.5, "optim/lambda": 0.1})
    logger.on_step("val", 1, {"loss/total": 0.6})

    fieldnames, rows = _read_rows(path)
    assert "loss/total" in fieldnames
    assert "optim/lambda" in fieldnames
    assert rows[0]["split"] == "train"
    assert rows[0]["loss/total"] == "0.5"
    assert rows[0]["optim/lambda"] == "0.1"
    assert rows[1]["split"] == "val"
    assert rows[1]["loss/total"] == "0.6"
    assert rows[1]["optim/lambda"] == ""


def test_stage2_new_diag_keys_do_not_corrupt_stage1_rows(tmp_path):
    """Stage 2 emits ``diag/sigma_data2/t=*`` keys never seen in stage 1.

    All stage-1 rows must retain their original values; the new columns
    must be appended with empty cells for the stage-1 history.
    """
    path = str(tmp_path / "metrics.csv")
    logger = CSVLogger(path)
    # Stage 1: a handful of training rows with the base schema.
    for step in range(1, 4):
        logger.on_step(
            "train",
            step,
            {"loss/total": 1.0 / step, "loss/rate/init/loss_init": 0.01 * step},
        )
    # Stage 2: same base keys plus per-t diagnostics.
    logger.on_step(
        "train",
        4,
        {
            "loss/total": 0.1,
            "loss/rate/init/loss_init": 0.05,
            "diag/sigma_data2/t=1": 0.7,
            "diag/sigma_data2/t=2": 0.8,
        },
    )

    fieldnames, rows = _read_rows(path)
    assert fieldnames == [
        "split",
        "step",
        "loss/total",
        "loss/rate/init/loss_init",
        "diag/sigma_data2/t=1",
        "diag/sigma_data2/t=2",
    ], fieldnames
    # Stage-1 rows preserved unchanged.
    for i, step in enumerate((1, 2, 3)):
        assert rows[i]["step"] == str(step)
        assert rows[i]["loss/total"] == str(1.0 / step)
        assert rows[i]["loss/rate/init/loss_init"] == str(0.01 * step)
        assert rows[i]["diag/sigma_data2/t=1"] == ""
        assert rows[i]["diag/sigma_data2/t=2"] == ""
    # Stage-2 row fully populated.
    assert rows[3]["loss/total"] == "0.1"
    assert rows[3]["loss/rate/init/loss_init"] == "0.05"
    assert rows[3]["diag/sigma_data2/t=1"] == "0.7"
    assert rows[3]["diag/sigma_data2/t=2"] == "0.8"


def test_resume_picks_up_existing_header(tmp_path):
    """A second logger instance pointed at the same file appends compatibly."""
    path = str(tmp_path / "metrics.csv")
    logger = CSVLogger(path)
    logger.on_step("train", 1, {"A": 1.0, "B": 2.0})

    # Fresh logger (simulating a resumed run) sees the existing header.
    logger2 = CSVLogger(path)
    logger2.on_step("train", 2, {"A": 10.0, "B": 20.0, "C": 30.0})

    fieldnames, rows = _read_rows(path)
    assert fieldnames == ["split", "step", "A", "B", "C"], fieldnames
    assert rows[0]["A"] == "1.0"
    assert rows[0]["B"] == "2.0"
    assert rows[0]["C"] == ""
    assert rows[1]["A"] == "10.0"
    assert rows[1]["B"] == "20.0"
    assert rows[1]["C"] == "30.0"


def test_no_rewrite_when_no_new_keys(tmp_path):
    """Steady-state cost is a plain append: file mtime/contents must reflect that.

    We can't easily assert "no rewrite" from outside, but we can verify that
    repeated identical-schema rows don't disturb earlier rows.
    """
    path = str(tmp_path / "metrics.csv")
    logger = CSVLogger(path)
    for step in range(1, 6):
        logger.on_step("train", step, {"A": float(step), "B": float(step) * 2})

    fieldnames, rows = _read_rows(path)
    assert fieldnames == ["split", "step", "A", "B"]
    assert len(rows) == 5
    for i, step in enumerate(range(1, 6)):
        assert rows[i]["A"] == str(float(step))
        assert rows[i]["B"] == str(float(step) * 2)


def test_header_rewrite_is_atomic(tmp_path):
    """A crash mid-rewrite must not corrupt the existing metrics.csv.

    Regression guard: the old rewrite opened the file for write directly, so
    any exception between open() and close() left a truncated or partial file.
    The fix writes to a sibling temp file then calls ``os.replace``, which is
    atomic on POSIX.
    """
    path = str(tmp_path / "metrics.csv")
    logger = CSVLogger(path)
    # Write an initial row so the file exists with content.
    logger.on_step("train", 1, {"A": 1.0, "B": 2.0})
    original_contents = open(path).read()

    # Patch os.replace to raise *after* the temp file is written, simulating a
    # crash at the swap step.  The original file must survive intact.
    real_replace = os.replace
    replace_called: list[tuple[str, str]] = []

    def _failing_replace(src, dst):
        replace_called.append((src, dst))
        raise OSError("simulated disk-full during os.replace")

    with mock.patch("ddssm.training.loggers.os.replace", side_effect=_failing_replace):
        try:
            # Trigger header rewrite by introducing a new key.
            logger.on_step("train", 2, {"A": 10.0, "C": 30.0})
        except OSError:
            pass  # expected from the patched os.replace

    assert replace_called, "os.replace was not called — rewrite path not exercised"
    # The original file must be intact (not truncated/overwritten).
    assert open(path).read() == original_contents, (
        "metrics.csv was corrupted by a failed mid-rewrite"
    )
    # Verify the temp file was cleaned up on failure.
    src_tmp = replace_called[0][0]
    assert not os.path.exists(src_tmp), (
        f"temp rewrite file {src_tmp!r} was not cleaned up after failure"
    )
