"""Auto-discover DDSSM training output and print structured diagnostics.

Designed for the `check-training-progress` skill. The assistant invokes this
script, then narrates the structured plain-text it prints back to the user.
Stdlib only — `optuna` and `sqlite3` are imported lazily so the script keeps
working in environments where they're unavailable.

Usage:
    python progress.py                       # auto-detect newest under runs/
    python progress.py <path>                # explicit run / sweep / parent
    python progress.py <path> --tail-rows 50 # widen the tail window
    python progress.py <path> --max-trials 8 # cap per-trial output in sweeps

Layout detection:
    1. <path>/metrics.csv exists                 → SINGLE_RUN
    2. <path>/{0,1,...}/metrics.csv exists       → SWEEP (Hydra multirun)
    3. <path>/<cell>/metrics.csv exists for >=2  → MULTI_CELL (headline_*)
    4. <path>/<sweep>/{0,...}/metrics.csv exists → MULTI_SWEEP (parent of sweeps)
    5. <path> has no metrics — list newest children with timestamps
"""

from __future__ import annotations

import os
import csv
import sys
import json
import math
import time
import argparse
from typing import Any
from pathlib import Path

# ----- Discovery ----------------------------------------------------------


def detect_layout(p: Path) -> str:
    """Classify a path as one of the recognised run-output layouts.

    Args:
        p: Directory to inspect.

    Returns:
        One of ``MISSING``, ``SINGLE_RUN``, ``SWEEP``, ``MULTI_CELL``,
        ``MULTI_SWEEP``, or ``EMPTY``.
    """
    if not p.exists():
        return "MISSING"
    if (p / "metrics.csv").is_file():
        return "SINGLE_RUN"
    children = [c for c in p.iterdir() if c.is_dir()]
    has_numeric_trial = any(c.name.isdigit() and (c / "metrics.csv").is_file() for c in children)
    if has_numeric_trial:
        return "SWEEP"
    has_named_cell = sum(1 for c in children if (c / "metrics.csv").is_file()) >= 2
    if has_named_cell:
        return "MULTI_CELL"
    has_sweep_child = any(
        any(g.name.isdigit() and (g / "metrics.csv").is_file() for g in c.iterdir() if c.is_dir())
        for c in children if c.is_dir()
    )
    if has_sweep_child:
        return "MULTI_SWEEP"
    return "EMPTY"


def auto_pick_root(repo_root: Path) -> Path | None:
    """Find the most recently modified plausible run target under `runs/`."""
    runs = repo_root / "runs"
    if not runs.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for c in runs.iterdir():
        if not c.is_dir():
            continue
        if c.name in {"optuna", "report", "sbatch"}:
            # Bookkeeping dirs — never the user's "where is training now" answer.
            continue
        candidates.append((c.stat().st_mtime, c))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# ----- CSV scanning -------------------------------------------------------


def _safe_float(s: str) -> float | None:
    """Parse a CSV cell to float, returning None for blank/non-finite/garbage."""
    if s == "" or s.lower() in {"nan", "inf", "-inf"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_csv(p: Path, limit: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    """Read a metrics CSV.

    Args:
        p: Path to the CSV file.
        limit: If given, stop after this many data rows.

    Returns:
        ``(fieldnames, rows)`` where rows are dicts keyed by column name.
    """
    with p.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if limit is not None and i + 1 >= limit:
                break
    return fields, rows


def group_columns(fields: list[str]) -> dict[str, list[str]]:
    """Bucket columns by prefix family: loss/, diag/, calib/, optim/, time/, etc."""
    groups: dict[str, list[str]] = {}
    for col in fields:
        if "/" not in col:
            family = "_scalar"
        else:
            family = col.split("/", 1)[0]
        groups.setdefault(family, []).append(col)
    return groups


def head_tail(rows: list[dict[str, str]], col: str, *, head_n: int = 20, tail_n: int = 20) -> dict[str, Any]:
    """Summarise one column's head vs tail to gauge training direction.

    Args:
        rows: Parsed CSV rows.
        col: Column name to summarise.
        head_n: Number of leading finite values to average.
        tail_n: Number of trailing finite values to average.

    Returns:
        Dict with finite-value count, NaN/Inf count, and (when any finite
        values exist) head/tail means, their delta and relative delta, min,
        max, and last value.
    """
    vals: list[float] = []
    nans = 0
    for r in rows:
        v = _safe_float(r.get(col, ""))
        if v is None:
            nans += 1
        else:
            if not math.isfinite(v):
                nans += 1
                continue
            vals.append(v)
    if not vals:
        return {"n": 0, "nans": nans}
    head = vals[: min(head_n, len(vals))]
    tail = vals[-min(tail_n, len(vals)):]
    h = sum(head) / len(head)
    t = sum(tail) / len(tail)
    return {
        "n": len(vals),
        "nans": nans,
        "head_mean": h,
        "tail_mean": t,
        "delta": t - h,
        "rel_delta": (t - h) / (abs(h) + 1e-12),
        "min": min(vals),
        "max": max(vals),
        "last": vals[-1],
    }


def lambda_state(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    """Summarise the λ-warmup schedule from the ``optim/lambda`` column.

    Args:
        rows: Parsed CSV rows.

    Returns:
        Dict with first/last λ, whether warmup is complete (λ ≥ 0.99), and the
        step at which λ first reached 0.99; or None if the column is absent or
        the last value is unreadable.
    """
    if not rows or "optim/lambda" not in rows[0]:
        return None
    last = _safe_float(rows[-1].get("optim/lambda", ""))
    first = _safe_float(rows[0].get("optim/lambda", ""))
    if last is None:
        return None
    cross_step = None
    for r in rows:
        v = _safe_float(r.get("optim/lambda", ""))
        if v is not None and v >= 0.99:
            cross_step = int(r["step"]) if r.get("step") else None
            break
    return {
        "first": first, "last": last,
        "warmup_complete": last >= 0.99,
        "warmup_cross_step": cross_step,
    }


def sigma_data_summary(rows: list[dict[str, str]], fields: list[str]) -> dict[str, Any] | None:
    """Summarise the last-row per-t ``diag/sigma_data2`` spread.

    Args:
        rows: Parsed CSV rows.
        fields: Column names, scanned for ``diag/sigma_data2/t=`` keys.

    Returns:
        Dict with the number of t-columns, mean/std/min/max over the final
        row, and the drift of the mean from 1.0; or None if no such columns or
        values exist.
    """
    sd_cols = [c for c in fields if c.startswith("diag/sigma_data2/t=")]
    if not sd_cols or not rows:
        return None
    last_row = rows[-1]
    vals = [v for c in sd_cols if (v := _safe_float(last_row.get(c, ""))) is not None]
    if not vals:
        return None
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)
    return {
        "n_t": len(sd_cols),
        "mean": mu,
        "std": var ** 0.5,
        "min": min(vals),
        "max": max(vals),
        "drift_from_1": abs(mu - 1.0),
    }


def is_actively_writing(p: Path, *, fresh_seconds: int = 300) -> bool:
    """Return True if ``p`` was modified within the last ``fresh_seconds``."""
    try:
        return (time.time() - p.stat().st_mtime) < fresh_seconds
    except OSError:
        return False


# ----- Single-run report --------------------------------------------------


# Columns we always summarise if present, in this display order.
PRIMARY_LOSS_COLS = (
    "loss/total",
    "loss/distortion/rec",
    "loss/rate/total",
    "loss/rate/init/tot",
    "loss/rate/trans/kl",
    "loss/rate/trans/r_sigma_p",
    "loss/rate/trans/r_mu_p",
    "calib/ratio_res2_to_sigma2",
)


def report_single_run(run_dir: Path, *, head_n: int = 20, tail_n: int = 20) -> None:
    """Print a single-run diagnostic block (losses, λ, σ_data², NaN scan)."""
    csv_path = run_dir / "metrics.csv"
    json_path = run_dir / "metrics.json"
    fields, rows = load_csv(csv_path)
    groups = group_columns(fields)
    last_step = rows[-1].get("step") if rows else None
    fresh = is_actively_writing(csv_path)

    print(f"== SINGLE_RUN  {run_dir}")
    print(f"   rows={len(rows)}  last_step={last_step}  csv_age={int(time.time() - csv_path.stat().st_mtime)}s  active={fresh}")
    if json_path.is_file():
        try:
            j = json.loads(json_path.read_text())
            scalar_keys = [k for k, v in j.items() if isinstance(v, (int, float)) and "buffer" not in k]
            picks = [k for k in scalar_keys if k.startswith("stage")] or scalar_keys[:6]
            for k in picks[:8]:
                print(f"   metrics.json :: {k} = {j[k]}")
        except Exception as exc:
            print(f"   metrics.json :: <unreadable: {exc}>")
    print()
    print("   families :", ", ".join(f"{k}({len(v)})" for k, v in groups.items()))
    print()

    # Lambda warmup
    lam = lambda_state(rows)
    if lam is not None:
        flag = "OK" if lam["warmup_complete"] else "WARMUP"
        cross = f"crossed @ step {lam['warmup_cross_step']}" if lam["warmup_cross_step"] is not None else "not yet @1.0"
        print(f"   lambda :: {flag}  last={lam['last']:.4g}  {cross}")

    # σ_data drift
    sd = sigma_data_summary(rows, fields)
    if sd is not None:
        verdict = "OK" if sd["drift_from_1"] < 0.15 else ("DRIFT" if sd["drift_from_1"] < 0.5 else "BAD_DRIFT")
        print(f"   sigma_data² :: {verdict}  mean={sd['mean']:.3f} ± {sd['std']:.3f}  range=[{sd['min']:.3f}, {sd['max']:.3f}]  drift|μ-1|={sd['drift_from_1']:.3f}")
    print()

    # Per-loss head/tail summary
    print(f"   {'column':<36} {'head':>10} {'tail':>10} {'Δ':>10} {'rel%':>8} {'last':>10}  flag")
    for col in PRIMARY_LOSS_COLS:
        if col not in fields:
            continue
        s = head_tail(rows, col, head_n=head_n, tail_n=tail_n)
        if s["n"] == 0:
            print(f"   {col:<36}  no values  (nans={s['nans']})")
            continue
        flag = ""
        if s["rel_delta"] < -0.05:
            flag = "descending"
        elif s["rel_delta"] > 0.05:
            flag = "INCREASING"
        else:
            flag = "flat"
        if s["nans"] > 0:
            flag += f" nans={s['nans']}"
        print(
            f"   {col:<36} {s['head_mean']:>10.3g} {s['tail_mean']:>10.3g} "
            f"{s['delta']:>+10.3g} {100*s['rel_delta']:>7.1f}% {s['last']:>10.3g}  {flag}"
        )

    # Any other column with NaNs we haven't surfaced
    nan_cols = []
    for col in fields:
        if col in PRIMARY_LOSS_COLS or col == "step" or col == "split":
            continue
        s = head_tail(rows, col, head_n=2, tail_n=2)
        if s.get("nans", 0) > 0:
            nan_cols.append((col, s["nans"]))
    if nan_cols:
        print()
        print("   NaN/Inf in other columns:", ", ".join(f"{c}({n})" for c, n in nan_cols[:8]))


# ----- Sweep report -------------------------------------------------------


def _optuna_best(db_path: Path) -> dict[str, Any] | None:
    """Read best-by-first-objective trial summary from an Optuna sqlite db."""
    if not db_path.is_file():
        return None
    try:
        import sqlite3
    except ImportError:
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None
    try:
        cur = conn.cursor()
        # Distinguish single- vs multi-objective layouts.
        try:
            row = cur.execute(
                "SELECT t.trial_id, t.number, t.state, v.value "
                "FROM trials t JOIN trial_values v ON v.trial_id=t.trial_id "
                "WHERE v.objective=0 AND t.state='COMPLETE' "
                "ORDER BY v.value ASC LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        n_complete = cur.execute(
            "SELECT COUNT(*) FROM trials WHERE state='COMPLETE'"
        ).fetchone()[0]
        n_running = cur.execute(
            "SELECT COUNT(*) FROM trials WHERE state='RUNNING'"
        ).fetchone()[0]
        n_failed = cur.execute(
            "SELECT COUNT(*) FROM trials WHERE state IN ('FAIL', 'PRUNED')"
        ).fetchone()[0]
        return {
            "n_complete": n_complete, "n_running": n_running, "n_failed": n_failed,
            "best": ({"trial": row[1], "value": row[3]} if row else None),
        }
    finally:
        conn.close()


def find_optuna_db_for(sweep_dir: Path, repo_root: Path) -> Path | None:
    """Best-effort: runs/sweeps/foo__mv → runs/optuna/foo__mv.db."""
    name = sweep_dir.name
    cand = repo_root / "runs" / "optuna" / f"{name}.db"
    if cand.is_file():
        return cand
    return None


def report_sweep(sweep_dir: Path, repo_root: Path, *, max_trials: int = 10, head_n: int = 20, tail_n: int = 20) -> None:
    """Print a per-trial table for a Hydra-multirun sweep plus Optuna db state.

    Joins the on-disk trial dirs with the matching Optuna sqlite db (if found)
    and emits sweep-level red flags (loss increasing, σ_data² drift, λ stalled).
    """
    trials = sorted(
        [c for c in sweep_dir.iterdir() if c.is_dir() and c.name.isdigit() and (c / "metrics.csv").is_file()],
        key=lambda p: int(p.name),
    )
    print(f"== SWEEP  {sweep_dir}")
    print(f"   trials_on_disk={len(trials)}")
    db = find_optuna_db_for(sweep_dir, repo_root)
    if db is not None:
        info = _optuna_best(db)
        if info is not None:
            print(f"   optuna db    : {db.name}")
            print(f"   optuna state : complete={info['n_complete']} running={info['n_running']} failed={info['n_failed']}")
            if info["best"] is not None:
                print(f"   optuna best  : trial #{info['best']['trial']}  value={info['best']['value']:.5g}")
    print()

    rows_out: list[tuple[str, dict[str, Any]]] = []
    for tdir in trials:
        try:
            fields, rows = load_csv(tdir / "metrics.csv")
        except Exception as exc:
            rows_out.append((tdir.name, {"error": str(exc)}))
            continue
        last_step = rows[-1].get("step") if rows else None
        elbo = None
        jp = tdir / "metrics.json"
        if jp.is_file():
            try:
                elbo = json.loads(jp.read_text()).get("stage2_elbo_surrogate")
            except Exception:
                elbo = None
        s_total = head_tail(rows, "loss/total", head_n=head_n, tail_n=tail_n) if "loss/total" in fields else {}
        lam = lambda_state(rows) or {}
        sd = sigma_data_summary(rows, fields) or {}
        rows_out.append((tdir.name, {
            "n_rows": len(rows),
            "last_step": last_step,
            "elbo": elbo,
            "active": is_actively_writing(tdir / "metrics.csv"),
            "loss_head": s_total.get("head_mean"),
            "loss_tail": s_total.get("tail_mean"),
            "loss_rel": s_total.get("rel_delta"),
            "lambda_last": lam.get("last"),
            "sd_mean": sd.get("mean"),
            "sd_drift": sd.get("drift_from_1"),
        }))

    print(f"   {'trial':<6} {'rows':>6} {'step':>6} {'elbo':>10} {'L_head':>10} {'L_tail':>10} {'L_rel%':>8} {'λ':>6} {'σd̄':>6} {'live':>5}")
    for name, r in rows_out[:max_trials]:
        if "error" in r:
            print(f"   {name:<6}  ERROR  {r['error']}")
            continue
        print(
            f"   {name:<6} {r['n_rows']:>6} {str(r['last_step']):>6} "
            f"{_fmt(r['elbo'], 4):>10} {_fmt(r['loss_head'], 3):>10} {_fmt(r['loss_tail'], 3):>10} "
            f"{(f'{100*r['loss_rel']:.1f}' if r['loss_rel'] is not None else '-'):>8} "
            f"{_fmt(r['lambda_last'], 3):>6} {_fmt(r['sd_mean'], 3):>6} {('Y' if r['active'] else '.'):>5}"
        )
    if len(rows_out) > max_trials:
        print(f"   ... {len(rows_out) - max_trials} more trials")

    # Sweep-level red flags
    print()
    actives = [r for _, r in rows_out if r.get("active")]
    incs = [n for n, r in rows_out if r.get("loss_rel") is not None and r["loss_rel"] > 0.05]
    bad_sd = [n for n, r in rows_out if r.get("sd_drift") is not None and r["sd_drift"] > 0.5]
    stuck_lam = [n for n, r in rows_out if r.get("lambda_last") is not None and r["lambda_last"] < 0.5 and r["n_rows"] > 200]
    print(f"   summary :: active={len(actives)} loss_increasing={len(incs)} sigma_data_bad_drift={len(bad_sd)} lambda_stuck<0.5={len(stuck_lam)}")
    if incs:
        print(f"   ⚠ loss_increasing trials: {incs[:8]}")
    if bad_sd:
        print(f"   ⚠ sigma_data drift trials: {bad_sd[:8]}")
    if stuck_lam:
        print(f"   ⚠ lambda warmup stalled: {stuck_lam[:8]}")


def _fmt(x: float | int | None, digits: int = 3) -> str:
    """Format a number for the table: ``-`` for None, ints as-is, else %g."""
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}g}"


# ----- Multi-cell report --------------------------------------------------


def report_multi_cell(parent: Path, *, head_n: int = 20, tail_n: int = 20) -> None:
    """Print one summary row per named cell under a multi-cell parent dir."""
    cells = sorted([c for c in parent.iterdir() if c.is_dir() and (c / "metrics.csv").is_file()])
    print(f"== MULTI_CELL  {parent}")
    print(f"   cells={len(cells)}")
    print(f"   {'cell':<40} {'rows':>6} {'step':>6} {'elbo':>10} {'L_head':>10} {'L_tail':>10} {'L_rel%':>8} {'λ':>6} {'σd̄':>6} {'live':>5}")
    for cdir in cells:
        try:
            fields, rows = load_csv(cdir / "metrics.csv")
        except Exception as exc:
            print(f"   {cdir.name:<40}  ERROR  {exc}")
            continue
        last_step = rows[-1].get("step") if rows else None
        elbo = None
        jp = cdir / "metrics.json"
        if jp.is_file():
            try:
                elbo = json.loads(jp.read_text()).get("stage2_elbo_surrogate")
            except Exception:
                elbo = None
        s = head_tail(rows, "loss/total", head_n=head_n, tail_n=tail_n) if "loss/total" in fields else {}
        lam = lambda_state(rows) or {}
        sd = sigma_data_summary(rows, fields) or {}
        active = is_actively_writing(cdir / "metrics.csv")
        rel = s.get("rel_delta")
        print(
            f"   {cdir.name:<40} {len(rows):>6} {str(last_step):>6} "
            f"{_fmt(elbo, 4):>10} {_fmt(s.get('head_mean'), 3):>10} {_fmt(s.get('tail_mean'), 3):>10} "
            f"{(f'{100*rel:.1f}' if rel is not None else '-'):>8} "
            f"{_fmt(lam.get('last'), 3):>6} {_fmt(sd.get('mean'), 3):>6} {('Y' if active else '.'):>5}"
        )


# ----- Multi-sweep (parent of sweep dirs) ---------------------------------


def report_multi_sweep(parent: Path, repo_root: Path) -> None:
    """Print one row per child sweep dir (newest first) with Optuna db state."""
    children = sorted(
        [c for c in parent.iterdir() if c.is_dir() and any(
            g.name.isdigit() and (g / "metrics.csv").is_file() for g in c.iterdir() if c.is_dir()
        )],
        key=lambda p: -p.stat().st_mtime,
    )
    print(f"== MULTI_SWEEP  {parent}")
    print(f"   sweep_dirs={len(children)}  (showing newest first)")
    for sd in children[:12]:
        n_trials = sum(1 for g in sd.iterdir() if g.is_dir() and g.name.isdigit() and (g / "metrics.csv").is_file())
        age = int(time.time() - sd.stat().st_mtime)
        db = find_optuna_db_for(sd, repo_root)
        opt = _optuna_best(db) if db else None
        best = (f"  best#{opt['best']['trial']}={opt['best']['value']:.4g}" if opt and opt["best"] else "")
        state = (f"  complete={opt['n_complete']} running={opt['n_running']}" if opt else "")
        print(f"   {sd.name:<60}  trials={n_trials:>3}  age={age}s{state}{best}")


# ----- SLURM / process scan ----------------------------------------------


def slurm_and_process_scan(repo_root: Path) -> None:
    """Print SLURM queue, GPU usage, and local ddssm processes if tools exist.

    Each probe (``squeue``/``nvidia-smi``/``pgrep``) is best-effort and silently
    skipped when the tool is missing or times out.
    """
    import subprocess

    # squeue (only if available + user has a job)
    try:
        out = subprocess.run(
            ["squeue", "--me", "--format=%i %j %T %M %R", "--noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            print("== SLURM (squeue --me)")
            for line in out.stdout.strip().splitlines():
                print(f"   {line}")
        elif out.returncode == 0:
            print("== SLURM :: no active jobs for current user")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # nvidia-smi
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            print("== GPU (nvidia-smi)")
            for line in out.stdout.strip().splitlines():
                print(f"   gpu{line}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # ddssm processes
    try:
        out = subprocess.run(["pgrep", "-af", "ddssm"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            print("== Local ddssm processes")
            for line in out.stdout.strip().splitlines()[:8]:
                # Trim long arg strings.
                pid, _, rest = line.partition(" ")
                print(f"   pid={pid}  {rest[:140]}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


# ----- Main ---------------------------------------------------------------


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the dir holding ``pyproject.toml`` + ``src/ddssm``.

    Falls back to the resolved ``start`` if no such ancestor is found.
    """
    p = start.resolve()
    while p != p.parent:
        if (p / "pyproject.toml").is_file() and (p / "src" / "ddssm").is_dir():
            return p
        p = p.parent
    return start.resolve()


def main(argv: list[str] | None = None) -> int:
    """Detect the layout of the target path and print the matching report.

    Returns:
        Process exit code: 0 on success, 1 for empty/unrecognised layouts, 2
        for a missing path or no discoverable target.
    """
    ap = argparse.ArgumentParser(description="Auto-discover DDSSM run output and print structured diagnostics.")
    ap.add_argument("path", nargs="?", default=None, help="Run / sweep / parent dir (default: newest under runs/)")
    ap.add_argument("--head-rows", type=int, default=20)
    ap.add_argument("--tail-rows", type=int, default=20)
    ap.add_argument("--max-trials", type=int, default=10)
    ap.add_argument("--no-slurm", action="store_true", help="Skip squeue/nvidia-smi/pgrep scans")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve()
    repo_root = find_repo_root(here.parent)

    if args.path is None:
        picked = auto_pick_root(repo_root)
        if picked is None:
            print("No path supplied and no runs/ subdir found.")
            return 2
        target = picked
        print(f"# auto-picked newest under runs/: {target.relative_to(repo_root)}")
    else:
        target = Path(args.path).resolve()

    layout = detect_layout(target)
    print(f"# layout: {layout}")
    print()

    if layout == "MISSING":
        print(f"path does not exist: {target}")
        return 2
    if layout == "EMPTY":
        print(f"path has no recognisable metrics.csv layout: {target}")
        children = sorted([c for c in target.iterdir() if c.is_dir()], key=lambda p: -p.stat().st_mtime)[:10]
        print("  newest subdirs:")
        for c in children:
            age = int(time.time() - c.stat().st_mtime)
            print(f"    {c.name:<60}  age={age}s")
        return 1
    if layout == "SINGLE_RUN":
        report_single_run(target, head_n=args.head_rows, tail_n=args.tail_rows)
    elif layout == "SWEEP":
        report_sweep(target, repo_root, max_trials=args.max_trials, head_n=args.head_rows, tail_n=args.tail_rows)
    elif layout == "MULTI_CELL":
        report_multi_cell(target, head_n=args.head_rows, tail_n=args.tail_rows)
    elif layout == "MULTI_SWEEP":
        report_multi_sweep(target, repo_root)

    if not args.no_slurm:
        print()
        slurm_and_process_scan(repo_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
