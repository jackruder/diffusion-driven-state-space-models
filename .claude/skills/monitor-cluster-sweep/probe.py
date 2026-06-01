#!/usr/bin/env python3
"""Gather per-cell stats for a DDSSM Optuna experiment, driven by a context.

Runs ON the cluster. Reads the context JSON produced by ``build_context.py``
(objectives + their directions, derived columns, suffix) and computes, per cell,
ONLY what the context says is relevant — no hardcoded objective/metric names:

  * trial-state counts; best value per objective (min if MINIMIZE else max);
  * each context-declared derived column (currently ``hit_rate``: hit/miss +
    crossing-step percentiles from the declared ``*_to_target_seconds`` key);
  * median per-trial wall-clock; SLURM queue (incl. ``TU_`` top-ups); and a
    projected final trial count vs ``--target``.

Folds any parallel ``*_topup`` sweep dir into each cell's stats.

Emits a human table to stderr and one ``__JSON__`` line to stdout.

Usage (on the cluster):
    python probe.py --remote-dir <dir> --context /tmp/ctx.json --target 128
"""
from __future__ import annotations

import os
import sys
import csv
import glob
import json
import argparse
import statistics
import subprocess
from collections import Counter


def _q(xs, p):
    """Return the ``p``-quantile (0..1) of ``xs`` by nearest-rank, or None."""
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def squeue_cell(job_name):
    """(running, pending, sum_running_timeleft_s) matching first-wave + TU_ jobs."""
    names = f"{job_name},TU_{job_name}"

    def _count(state):
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-t", state,
             "-n", names, "-O", "TimeLeft"], capture_output=True, text=True).stdout
        lines = [l for l in out.splitlines() if l.strip()]
        secs = 0
        for l in lines:
            p = l.strip().split(":")
            try:
                if len(p) == 3:
                    secs += int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
                elif len(p) == 2:
                    secs += int(p[0]) * 60 + int(p[1])
            except ValueError:
                pass
        return len(lines), secs
    nr, rs = _count("RUNNING")
    npd, _ = _count("PENDING")
    return nr, npd, rs


def trial_dirs(remote, study):
    """Trial subdirs for a cell, including its *_topup sibling."""
    for sd in (study, study + "_topup"):
        yield from glob.glob(os.path.join(remote, "sweeps", sd, "*"))


def main():
    """Compute per-cell stats per the context and emit the table + JSON.

    Prints a human table to stderr and one ``__JSON__``-prefixed line carrying
    the per-cell rows (state counts, best-per-objective, derived columns,
    median trial duration, queue state, projected final count) to stdout.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-dir", default="~/diffusion-driven-state-space-models")
    ap.add_argument("--context", required=True, help="context JSON from build_context.py")
    ap.add_argument("--target", type=int, default=128)
    args = ap.parse_args()
    remote = os.path.expanduser(args.remote_dir)
    ctx = json.load(open(args.context))
    prefix, suffix = ctx["study_prefix"], ctx["suffix"]
    objectives = ctx["objectives"]
    derived = ctx.get("derived", [])

    import optuna
    from optuna.trial import TrialState as TS
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dbs = sorted(glob.glob(os.path.join(
        remote, "optuna", f"{prefix}_*{suffix}.db")))
    rows = []
    for db in dbs:
        base = os.path.basename(db)[:-3]
        cell = base[len(prefix) + 1:]
        if suffix and cell.endswith(suffix):
            cell = cell[: -len(suffix)]
        try:
            st = optuna.load_study(study_name=base, storage=f"sqlite:///{db}")
        except Exception as e:
            rows.append({"cell": cell, "error": f"{type(e).__name__}: {e}"})
            continue
        ts = st.get_trials(deepcopy=False)
        cnt = Counter(t.state.name for t in ts)
        comp = [t for t in ts if t.state == TS.COMPLETE and t.values]

        best = []
        for o in objectives:
            i = o["idx"]
            vals = [t.values[i] for t in comp
                    if i < len(t.values) and t.values[i] is not None]
            if not vals:
                best.append(None)
            else:
                best.append(min(vals) if o["direction"] == "MINIMIZE" else max(vals))

        # derived columns + per-trial duration, scanned once over trial dirs
        dvals = {d["label"]: {"hit": [], "miss": 0, "steps": []} for d in derived}
        durs = []
        for td in trial_dirs(remote, base):
            mj = os.path.join(td, "metrics.json")
            if os.path.isfile(mj):
                try:
                    d = json.load(open(mj))
                except Exception:
                    d = {}
                for dc in derived:
                    if dc["type"] != "hit_rate":
                        continue
                    bucket = dvals[dc["label"]]
                    sec = d.get(dc["sec_key"], "__absent__")
                    if sec == "__absent__":
                        continue
                    if sec is None:
                        bucket["miss"] += 1
                    else:
                        bucket["hit"].append(float(sec))
                        sk = dc.get("step_key")
                        if sk and d.get(sk) is not None:
                            bucket["steps"].append(float(d[sk]))
            mc = os.path.join(td, "metrics.csv")
            if os.path.isfile(mc):
                last = None
                try:
                    with open(mc) as f:
                        for r in csv.DictReader(f):
                            last = r.get("time/elapsed_s")
                except Exception:
                    last = None
                if last:
                    try:
                        v = float(last)
                        if v > 30:
                            durs.append(v)
                    except ValueError:
                        pass

        derived_out = {}
        for dc in derived:
            b = dvals[dc["label"]]
            n = len(b["hit"]) + b["miss"]
            derived_out[dc["label"]] = {
                "hit": len(b["hit"]), "miss": b["miss"],
                "pct": (100 * len(b["hit"]) / n) if n else None,
                "step_p50": _q(b["steps"], .5), "step_p90": _q(b["steps"], .9),
            }

        trial_min = (statistics.median(durs) / 60) if durs else None
        nrun, npend, run_secs = squeue_cell(cell + suffix)
        proj = None
        if trial_min:
            proj = round(cnt.get("COMPLETE", 0)
                         + (run_secs / 60) / trial_min
                         + npend * 0.5 * (16 * 60 / trial_min))
        rows.append({
            "cell": cell, "complete": cnt.get("COMPLETE", 0),
            "running": cnt.get("RUNNING", 0), "fail": cnt.get("FAILED", 0),
            "pruned": cnt.get("PRUNED", 0), "best": best, "derived": derived_out,
            "trial_min": round(trial_min, 1) if trial_min else None,
            "q_running": nrun, "q_pending": npend,
            "q_run_timeleft_h": round(run_secs / 3600, 1),
            "proj_final": proj, "target": args.target,
        })

    # human table
    obj_hdr = " ".join(f"best_{o['short']:>9s}"[:14] for o in objectives)
    der_hdr = " ".join(f"{d['label']:>5s}" for d in derived)
    print(f"{'cell':30s} {'COMP':>4s} {'RUN':>3s} {obj_hdr} {der_hdr} "
          f"{'trl_m':>5s} {'q r/p':>6s} {'tl_h':>5s} {'proj':>4s}/{args.target}",
          file=sys.stderr)
    for r in rows:
        if "error" in r:
            print(f"{r['cell']:30s} ERR {r['error']}", file=sys.stderr)
            continue
        bs = " ".join(f"{(b if b is not None else float('nan')):>14.1f}" for b in r["best"])
        ds = " ".join((f"{r['derived'][d['label']]['pct']:.0f}%"
                       if r['derived'][d['label']]['pct'] is not None else "-").rjust(5)
                      for d in derived)
        print(f"{r['cell']:30s} {r['complete']:4d} {r['running']:3d} {bs} {ds} "
              f"{(r['trial_min'] or 0):5.0f} {r['q_running']}/{r['q_pending']:<4d} "
              f"{r['q_run_timeleft_h']:5.1f} {str(r['proj_final'] or '-'):>4s}",
              file=sys.stderr)
    print("__JSON__" + json.dumps({"study_prefix": prefix, "target": args.target,
                                   "context": ctx, "cells": rows}))


if __name__ == "__main__":
    main()
