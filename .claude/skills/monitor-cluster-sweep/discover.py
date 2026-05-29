#!/usr/bin/env python3
"""Discover DDSSM Optuna experiments on the cluster and rank them by activity.

Runs ON the cluster login node. Groups the per-cell Optuna DBs into
*experiments* (one study-prefix = one experiment) and reports which are live,
so the skill can suggest what to display when the user didn't name one.

Grouping strategy, most-robust first:
  1. SLURM jobs: each running/pending job name is ``<cell>__<suffix>``. Find the
     Optuna DB whose basename ENDS WITH that job name; the study prefix is the
     basename minus ``_<cell>__<suffix>``. Group studies by prefix.
  2. Idle fallback: any DB modified in the last ``--days`` days that no job
     matched, grouped by longest-common-prefix.

Emits one JSON line (``__JSON__`` prefixed) plus a human ranking to stderr.

Usage (on the cluster):
    python discover.py --remote-dir ~/diffusion-driven-state-space-models --days 3
"""
from __future__ import annotations

import os
import sys
import glob
import json
import time
import argparse
import subprocess
from collections import defaultdict


def squeue_jobs(user):
    """[(jobid, name, state)] for the user, names with any TU_ prefix stripped."""
    out = subprocess.run(
        ["squeue", "-u", user, "-h", "-O", "JobID:18,Name:60,StateCompact:8"],
        capture_output=True, text=True).stdout
    jobs = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        jid, name, st = parts[0], parts[1], parts[2]
        name = name[3:] if name.startswith("TU_") else name
        jobs.append((jid, name, st))
    return jobs


def split_jobname(name):
    """``<cell>__<suffix>`` -> (cell, '__suffix'). Project uses '__' delimiter."""
    if "__" in name:
        cell, suf = name.rsplit("__", 1)
        return cell, "__" + suf
    return name, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-dir", default="~/diffusion-driven-state-space-models")
    ap.add_argument("--days", type=int, default=3)
    args = ap.parse_args()
    remote = os.path.expanduser(args.remote_dir)
    user = os.environ.get("USER", "")
    db_dir = os.path.join(remote, "optuna")
    dbs = {os.path.basename(p)[:-3]: p for p in glob.glob(os.path.join(db_dir, "*.db"))}

    exps = defaultdict(lambda: {"cells": set(), "running": 0, "pending": 0,
                                "suffix": "", "mtime": 0.0})
    matched_dbs = set()

    # 1. squeue-driven grouping
    for jid, name, st in squeue_jobs(user):
        cell, suf = split_jobname(name)
        # find the DB ending with the (cell+suffix) job name
        cand = [s for s in dbs if s.endswith(name)]
        if not cand:
            continue
        study = max(cand, key=len)
        matched_dbs.add(study)
        prefix = study[: len(study) - len(name)].rstrip("_")
        e = exps[prefix]
        e["suffix"] = suf
        e["cells"].add(cell)
        e["mtime"] = max(e["mtime"], os.path.getmtime(dbs[study]))
        if st == "R":
            e["running"] += 1
        elif st == "PD":
            e["pending"] += 1

    # 2. idle fallback: recent DBs not matched to a running job
    cutoff = time.time() - args.days * 86400
    leftover = [(s, p) for s, p in dbs.items()
                if s not in matched_dbs and os.path.getmtime(p) > cutoff]
    if leftover:
        names = [s for s, _ in leftover]
        lcp = os.path.commonprefix(names).rstrip("_")
        if lcp:
            e = exps[lcp + " (idle)"]
            for s, p in leftover:
                e["cells"].add(s)
                e["mtime"] = max(e["mtime"], os.path.getmtime(p))

    out = []
    for prefix, e in exps.items():
        out.append({
            "study_prefix": prefix, "suffix": e["suffix"],
            "n_cells": len(e["cells"]), "cells": sorted(e["cells"]),
            "running": e["running"], "pending": e["pending"],
            "age_min": round((time.time() - e["mtime"]) / 60, 1) if e["mtime"] else None,
        })
    out.sort(key=lambda r: (r["running"], -(r["age_min"] or 1e9)), reverse=True)

    print("Experiments found (ranked by live activity):", file=sys.stderr)
    for r in out:
        print(f"  {r['study_prefix']:40s} cells={r['n_cells']:2d} "
              f"run={r['running']:2d} pend={r['pending']:2d} "
              f"suffix={r['suffix'] or '-':6s} age={r['age_min']}m", file=sys.stderr)
    if not out:
        print("  (none — no DBs and no jobs)", file=sys.stderr)
    print("__JSON__" + json.dumps({"experiments": out}))


if __name__ == "__main__":
    main()
