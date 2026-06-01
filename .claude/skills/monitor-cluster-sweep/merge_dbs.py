#!/usr/bin/env python3
"""Merge per-cell Optuna sqlite DBs into one combined DB for optuna-dashboard.

optuna-dashboard serves a single storage; this folds every per-cell study into
one file so all cells appear in the dashboard's study dropdown. Idempotent —
re-run after each pull to refresh. Run locally via ``uv run --with optuna``.

Usage:
    uv run --with optuna python merge_dbs.py <pull_dir> <combined.db> <prefix>
"""
from __future__ import annotations

import os
import sys
import glob


def main():
    """Copy each per-cell study from ``pull_dir`` into one combined DB.

    Reads ``pull_dir``, ``combined.db`` path, and study ``prefix`` from argv.
    Each cell study is stored under its prefix-stripped short name, replacing
    any prior copy so re-runs refresh idempotently.
    """
    pull_dir, combined_path, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    combined = f"sqlite:///{combined_path}"
    dbs = sorted(glob.glob(os.path.join(pull_dir, f"{prefix}_*.db")))
    if not dbs:
        print(f"no DBs matching {prefix}_*.db in {pull_dir}")
        return
    for db in dbs:
        if os.path.abspath(db) == os.path.abspath(combined_path):
            continue
        sname = os.path.basename(db)[:-3]
        short = sname[len(prefix) + 1:]
        try:
            try:
                optuna.delete_study(study_name=short, storage=combined)
            except Exception:
                pass
            optuna.copy_study(from_study_name=sname, from_storage=f"sqlite:///{db}",
                              to_storage=combined, to_study_name=short)
            st = optuna.load_study(study_name=short, storage=combined)
            print(f"merged {short:34s} trials={len(st.get_trials(deepcopy=False))}")
        except Exception as e:
            print(f"FAIL {short}: {type(e).__name__} {e}")
    print("combined ->", os.path.abspath(combined_path))


if __name__ == "__main__":
    main()
