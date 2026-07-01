"""Collect head-to-head campaign results into the REPORT.md tables.

Reads ``runs/h2h/phase1/<cell>/<N>/metrics.json`` (recon_mse per base_lr) and
``runs/h2h/phase2/<cell>/study.db`` (Optuna best CRPS-sum + params), and prints
markdown table fragments + the best Phase-2 config per cell (the finalist to
seed-replicate and test-eval). Robust to a partially-finished campaign — missing
cells render as ``--``.

Run::

    .venv/bin/python experiments/arflow_headtohead/collect_results.py
"""

import torch  # preload before numpy on NixOS  # noqa: F401
import os
import csv
import json
import glob

ROOT = os.path.join("runs", "h2h")
ENCS = ("gaussian", "iaf", "det")
DSS = ("lgssm", "nlblmv")
_BL = "experiment.training.stages.base_lr"


def _overrides(run_dir: str) -> dict:
    path = os.path.join(run_dir, ".hydra", "overrides.yaml")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip().lstrip("- ").strip()
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _json(run_dir: str) -> dict:
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _steps_run(run_dir: str) -> int:
    """Last train `step` in metrics.csv (Phase-1 budget actually consumed)."""
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(path):
        return 0
    last = 0
    with open(path) as f:
        for row in csv.DictReader(f):
            s = row.get("step", "")
            if s not in ("", None):
                try:
                    last = max(last, int(float(s)))
                except ValueError:
                    pass
    return last


def phase1() -> None:
    print("## Phase 1 — encoder capacity (recon_mse on mu_x, pure AE)\n")
    print("| dataset | encoder | best base_lr | recon_mse | steps | per-LR recon_mse |")
    print("|---------|---------|-------------|-----------|-------|------------------|")
    for ds in DSS:
        for enc in ENCS:
            cell = f"{enc}_{ds}"
            cdir = os.path.join(ROOT, "phase1", cell)
            rows = []
            for sub in sorted(glob.glob(os.path.join(cdir, "[0-9]*"))):
                lr = _overrides(sub).get(_BL)
                mse = _json(sub).get("recon_mse")
                if lr is not None and mse is not None:
                    rows.append((float(lr), float(mse), _steps_run(sub)))
            if not rows:
                print(f"| {ds} | {enc} | -- | -- | -- | _pending_ |")
                continue
            best = min(rows, key=lambda r: r[1])
            grid = " ".join(f"{lr:.0e}={mse:.4f}" for lr, mse, _ in sorted(rows))
            print(f"| {ds} | {enc} | {best[0]:.0e} | {best[1]:.4f} | {best[2]} | {grid} |")
    print()


def phase2() -> None:
    import optuna

    print("## Phase 2 — sweep best (val CRPS-sum, finalist per cell)\n")
    print("| dataset | encoder | best val CRPS | n_trials | best config |")
    print("|---------|---------|--------------|----------|-------------|")
    finalists = {}
    for ds in DSS:
        for enc in ENCS:
            cell = f"{enc}_{ds}"
            db = os.path.join(ROOT, "phase2", cell, "study.db")
            if not os.path.isfile(db):
                print(f"| {ds} | {enc} | -- | 0 | _pending_ |")
                continue
            try:
                st = optuna.load_study(study_name=cell, storage=f"sqlite:///{db}")
            except Exception as e:  # noqa: BLE001
                print(f"| {ds} | {enc} | ERR | -- | {e} |")
                continue
            done = [t for t in st.trials if t.value is not None]
            if not done:
                print(f"| {ds} | {enc} | -- | {len(st.trials)} | _running_ |")
                continue
            bt = st.best_trial
            finalists[cell] = bt.params
            cfg = ", ".join(
                f"{k.split('.')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
                for k, v in bt.params.items()
            )
            print(f"| {ds} | {enc} | {bt.value:.4f} | {len(done)} | {cfg} |")
    print()
    if finalists:
        print("### Finalist configs (to seed-replicate + test-eval)\n```")
        for cell, params in finalists.items():
            ov = " ".join(f"{k}={v}" for k, v in params.items())
            print(f"# {cell}\n{ov}\n")
        print("```")


if __name__ == "__main__":
    phase1()
    phase2()
