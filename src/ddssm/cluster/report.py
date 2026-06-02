"""Self-describing run health: ``run_summary.json`` + ``python -m ddssm.cluster.report``.

``summarize_run`` reduces a run's ``metrics.csv`` to a compact health dict
(final loss, λ-warmup state, σ_data² drift, val loss, non-finite count,
elapsed, stages). ``Experiment.train`` calls :func:`write_run_summary` at exit so
every run is self-describing without re-deriving from the CSV. The
``python -m ddssm.cluster.report <path>`` CLI prints a single run's summary, or — given a
parent directory of run dirs — a one-row-per-run comparison table.
"""

from __future__ import annotations

import csv
import sys
import json
from typing import Any
from pathlib import Path

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _safe_float(s: Any) -> float | None:
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    return f


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _col(rows: list[dict[str, str]], col: str) -> list[float]:
    """Finite values of ``col`` across rows, in order."""
    out: list[float] = []
    for r in rows:
        v = _safe_float(r.get(col, ""))
        if v is not None and v == v and v not in (float("inf"), float("-inf")):
            out.append(v)
    return out


def _head_tail(vals: list[float], n: int = 20) -> dict[str, float | None]:
    if not vals:
        return {"head": None, "tail": None, "last": None}
    head = sum(vals[: min(n, len(vals))]) / min(n, len(vals))
    tail = sum(vals[-min(n, len(vals)):]) / min(n, len(vals))
    return {"head": head, "tail": tail, "last": vals[-1]}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_run(run_dir: str | Path, *, tail_n: int = 20) -> dict[str, Any]:
    """Reduce ``<run_dir>/metrics.csv`` to a health-summary dict.

    Returns ``{"available": False, ...}`` when there is no CSV to read.
    """
    run_dir = Path(run_dir)
    csv_path = run_dir if run_dir.suffix == ".csv" else run_dir / "metrics.csv"
    if not csv_path.is_file():
        return {"available": False, "run_dir": str(run_dir)}

    fields, rows = _load_csv(csv_path)
    train = [r for r in rows if r.get("split", "train") == "train"]
    val = [r for r in rows if r.get("split") == "val"]

    summary: dict[str, Any] = {
        "available": True,
        "run_dir": str(run_dir),
        "rows": len(rows),
        "final_step": int(_col(train, "step")[-1]) if _col(train, "step") else None,
    }

    # Optimized objective + raw ELBO.
    summary["loss_total"] = _head_tail(_col(train, "loss/total"), tail_n)
    if "loss/total_unweighted" in fields:
        summary["loss_total_unweighted"] = _head_tail(
            _col(train, "loss/total_unweighted"), tail_n
        )

    # Validation loss (set-mean per validation event; report the last).
    vlt = _col(val, "loss/total")
    summary["val_loss_total_last"] = vlt[-1] if vlt else None

    # λ-warmup state.
    lam = _col(train, "optim/lambda")
    if lam:
        summary["lambda"] = {"last": lam[-1], "max": max(lam)}
        cross = next((i for i, v in enumerate(lam) if v >= 0.999), None)
        summary["lambda"]["warmup_complete"] = cross is not None

    # σ_data² across the per-t buffer slots (last row): mean + drift vs first row.
    sd_cols = sorted(c for c in fields if c.startswith("diag/sigma_data2/t="))
    if sd_cols and train:
        def _meanrow(r: dict[str, str]) -> float | None:
            vals = [_safe_float(r.get(c, "")) for c in sd_cols]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None
        first_m, last_m = _meanrow(train[0]), _meanrow(train[-1])
        summary["sigma_data2"] = {
            "mean_last": last_m,
            "drift": (last_m - first_m) if (first_m is not None and last_m is not None) else None,
        }

    # Stages observed + run health.
    stage_idx = _col(train, "stage/idx")
    summary["stages_run"] = sorted({int(s) for s in stage_idx}) if stage_idx else None
    nf = _col(train, "nonfinite/total")
    summary["nonfinite_total"] = int(nf[-1]) if nf else 0
    el = _col(train, "time/elapsed_s")
    summary["elapsed_s"] = el[-1] if el else None
    return summary


def write_run_summary(run_dir: str | Path) -> dict[str, Any] | None:
    """Write ``<run_dir>/run_summary.json`` from the run's CSV. No-op if absent."""
    summary = summarize_run(run_dir)
    if not summary.get("available"):
        return None
    out = Path(run_dir) / "run_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def _print_single(summary: dict[str, Any]) -> None:
    if not summary.get("available"):
        print(f"no metrics.csv under {summary['run_dir']}")
        return
    lt = summary["loss_total"]
    print(f"run: {summary['run_dir']}")
    print(f"  rows={summary['rows']}  final_step={_fmt(summary['final_step'])}  "
          f"stages={summary['stages_run']}  elapsed_s={_fmt(summary['elapsed_s'])}")
    print(f"  loss/total   head={_fmt(lt['head'])} tail={_fmt(lt['tail'])} last={_fmt(lt['last'])}")
    if "lambda" in summary:
        lam = summary["lambda"]
        print(f"  lambda       last={_fmt(lam['last'])} warmup_complete={lam['warmup_complete']}")
    if "sigma_data2" in summary:
        sd = summary["sigma_data2"]
        print(f"  sigma_data2  mean_last={_fmt(sd['mean_last'])} drift={_fmt(sd['drift'])}")
    print(f"  val_loss/total last={_fmt(summary['val_loss_total_last'])}  "
          f"nonfinite_total={summary['nonfinite_total']}")


def _print_parent(parent: Path) -> None:
    runs = sorted(d for d in parent.iterdir() if (d / "metrics.csv").is_file())
    if not runs:
        print(f"no run dirs (with metrics.csv) under {parent}")
        return
    print(f"{'run':<40} {'step':>7} {'L_tail':>10} {'val':>10} {'λ':>6} {'σd̄':>7} {'nf':>4}")
    for d in runs:
        s = summarize_run(d)
        lam = s.get("lambda", {}).get("last") if "lambda" in s else None
        sd = s.get("sigma_data2", {}).get("mean_last") if "sigma_data2" in s else None
        print(f"{d.name:<40} {_fmt(s['final_step']):>7} {_fmt(s['loss_total']['tail']):>10} "
              f"{_fmt(s['val_loss_total_last']):>10} {_fmt(lam):>6} {_fmt(sd):>7} "
              f"{s['nonfinite_total']:>4}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry: print one run's summary, or a table over a parent of run dirs."""
    import argparse

    ap = argparse.ArgumentParser(description="Summarize DDSSM run health from metrics.csv.")
    ap.add_argument("path", help="a run dir (metrics.csv) or a parent of run dirs")
    ap.add_argument("--json", action="store_true", help="print the summary dict as JSON")
    args = ap.parse_args(argv)

    p = Path(args.path)
    is_run = (p / "metrics.csv").is_file() or p.suffix == ".csv"
    if is_run:
        s = summarize_run(p)
        if args.json:
            print(json.dumps(s, indent=2, default=float))
        else:
            _print_single(s)
    else:
        _print_parent(p)


if __name__ == "__main__":
    main(sys.argv[1:])
