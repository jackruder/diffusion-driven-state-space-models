"""Tests for ddssm.report (run_summary + summarize_run)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ddssm.report import summarize_run, write_run_summary


def _write_metrics(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "step", "loss/total", "loss/total_unweighted", "optim/lambda",
        "stage/idx", "diag/sigma_data2/t=1", "nonfinite/total", "time/elapsed_s",
    ]
    rows = []
    for i in range(1, 11):  # stage 1: steps 1..5, stage 2: steps 6..10
        rows.append({
            "split": "train", "step": str(i),
            "loss/total": str(10.0 - i),          # decreasing
            "loss/total_unweighted": str(12.0 - i),
            "optim/lambda": str(min(1.0, i / 5.0)),  # reaches 1.0
            "stage/idx": "1" if i <= 5 else "2",
            "diag/sigma_data2/t=1": str(1.0 + 0.01 * i),
            "nonfinite/total": "0",
            "time/elapsed_s": str(float(i)),
        })
    rows.append({"split": "val", "step": "10", "loss/total": "3.5",
                 "loss/total_unweighted": "", "optim/lambda": "", "stage/idx": "",
                 "diag/sigma_data2/t=1": "", "nonfinite/total": "0",
                 "time/elapsed_s": ""})
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_summarize_run_reads_metrics(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "run")
    s = summarize_run(tmp_path / "run", tail_n=3)
    assert s["available"] is True
    assert s["final_step"] == 10
    assert s["loss_total"]["last"] == 0.0          # 10 - 10
    assert s["loss_total"]["tail"] < s["loss_total"]["head"]  # decreasing
    assert s["lambda"]["last"] == 1.0
    assert s["lambda"]["warmup_complete"] is True
    assert s["stages_run"] == [1, 2]
    assert s["val_loss_total_last"] == 3.5
    assert s["nonfinite_total"] == 0
    assert s["sigma_data2"]["drift"] > 0           # 1.10 - 1.01


def test_summarize_run_missing_csv(tmp_path: Path) -> None:
    s = summarize_run(tmp_path / "nope")
    assert s["available"] is False


def test_write_run_summary_roundtrips(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_metrics(run)
    s = write_run_summary(run)
    assert s is not None
    written = json.loads((run / "run_summary.json").read_text())
    assert written["final_step"] == 10
    assert written["stages_run"] == [1, 2]
