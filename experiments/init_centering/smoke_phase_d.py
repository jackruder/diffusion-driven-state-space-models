"""End-to-end Phase-D smoke — run every cell locally with shrunk knobs.

This is the recommended sanity check *before* submitting the real
20-job SLURM batch.  It:

1. Iterates every cell of the ablation grid + 2 controls.
2. Runs each as a single local ``python -m ddssm.app`` job (no
   Optuna, no SLURM, no multirun) with tiny step counts.
3. Pins each cell's ``hydra.run.dir`` to
   ``{sweeps_root}/smoke_phase_d_{cell}/`` so Phase E's
   :mod:`.report` can pick the runs up unchanged.
4. Optionally chains the reporter at the end (``--report``).

The whole sweep runs on CPU in a few minutes.

Run::

    python -m experiments.init_centering.smoke_phase_d --report

Just one cell to debug a regression::

    python -m experiments.init_centering.smoke_phase_d \\
        --cell init_zero_pinned_fixed --report

Custom step count + report::

    python -m experiments.init_centering.smoke_phase_d \\
        --steps 3 --report --out-dir runs/smoke_$(date +%s)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Iterable
import argparse
import subprocess

from experiments.init_centering.launch_phase_d import all_phase_d_cells


# Conservative shrink-set: matches the slow ``test_pilot_end_to_end_...``
# fixture (5 + 5 stage steps).  Keeps NN dims at defaults to avoid running
# into schedule / nhead-divides-channels constraints; the smoke is about
# pipeline plumbing, not throughput.
def _shrink_overrides(steps: int) -> list[str]:
    return [
        f"experiment.training.stages.n_pretrain={steps}",
        f"experiment.training.stages.n_stage2={steps}",
        "experiment.training.stages.log_every=1",
        "experiment.training.stages.checkpoint_every=100",
    ]


def _iter_cells(only: str | None) -> Iterable[str]:
    if only is None:
        yield from all_phase_d_cells()
        return
    if only not in all_phase_d_cells():
        raise SystemExit(
            f"Unknown cell {only!r}.  Choices: {', '.join(all_phase_d_cells())}"
        )
    yield only


def run_one_cell(
    cell: str, *, sweeps_root: str, study_prefix: str, steps: int,
    extra_overrides: list[str] | None = None,
) -> tuple[str, int, float]:
    """Run a single cell locally; return (run_dir, return_code, elapsed_sec)."""
    run_dir = os.path.join(sweeps_root, f"{study_prefix}_{cell}")
    os.makedirs(run_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "ddssm.app",
        f"experiment={cell}",
        f"hydra.run.dir={run_dir}",
        *_shrink_overrides(steps),
        *(extra_overrides or []),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    return run_dir, proc.returncode, elapsed


def run_report(out_dir: str, *, sweeps_root: str, study_prefix: str) -> int:
    """Chain the Phase-E reporter against the smoke outputs."""
    cmd = [
        sys.executable, "-m", "experiments.init_centering.report",
        "all",
        "--sweeps-root", sweeps_root,
        "--optuna-dir", "runs/optuna",  # ignored: smoke runs have no Optuna DB
        "--study-prefix", study_prefix,
        "--out", out_dir,
    ]
    return subprocess.run(cmd, check=False).returncode


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.init_centering.smoke_phase_d",
        description=(
            "End-to-end Phase-D smoke: run every cell locally with "
            "tiny step counts, then optionally chain the Phase-E report."
        ),
    )
    p.add_argument(
        "--cell", default=None,
        help="Only run one cell (e.g. 'init_mlp_pinned_per_t').",
    )
    p.add_argument(
        "--steps", type=int, default=5,
        help="Steps per stage (default 5 — matches the slow pilot test).",
    )
    p.add_argument(
        "--out-dir", default="runs/smoke_phase_d",
        help=(
            "Root for the smoke run.  Sweeps land at "
            "{out_dir}/sweeps/{study_prefix}_{cell}/; report at "
            "{out_dir}/report/  (default runs/smoke_phase_d)."
        ),
    )
    p.add_argument(
        "--study-prefix", default="smoke_phase_d",
        help="Prefix joining the cells to the reporter (default smoke_phase_d).",
    )
    p.add_argument(
        "--report", action="store_true",
        help="Chain the Phase-E reporter after the cells finish.",
    )
    p.add_argument(
        "--no-stop-on-error", action="store_true",
        help="Continue running remaining cells even if one fails.",
    )
    p.add_argument(
        "override", nargs=argparse.REMAINDER,
        help="Extra Hydra overrides (forwarded verbatim).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    sweeps_root = os.path.join(args.out_dir, "sweeps")
    report_dir = os.path.join(args.out_dir, "report")
    os.makedirs(sweeps_root, exist_ok=True)

    targets = list(_iter_cells(args.cell))
    print(f"Smoke: {len(targets)} cell(s) × {args.steps} steps/stage")
    print(f"Outputs: {sweeps_root}")
    print()

    failures: list[tuple[str, int]] = []
    for i, cell in enumerate(targets, 1):
        print(f"[{i:>2}/{len(targets)}] {cell} ...", flush=True)
        run_dir, rc, elapsed = run_one_cell(
            cell,
            sweeps_root=sweeps_root,
            study_prefix=args.study_prefix,
            steps=args.steps,
            extra_overrides=args.override or None,
        )
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"           {status}  {elapsed:6.1f}s  {run_dir}")
        if rc != 0:
            failures.append((cell, rc))
            if not args.no_stop_on_error:
                print(f"\nStopped on first failure ({cell}).  "
                      f"Pass --no-stop-on-error to continue past failures.")
                return 1

    print()
    if failures:
        print(f"Smoke FAILED on {len(failures)} cell(s):")
        for cell, rc in failures:
            print(f"  - {cell} (rc={rc})")
        return 1
    print(f"Smoke OK on all {len(targets)} cells.")

    if args.report:
        print()
        print(f"Running Phase-E report → {report_dir}")
        rc = run_report(
            report_dir,
            sweeps_root=sweeps_root,
            study_prefix=args.study_prefix,
        )
        if rc != 0:
            print(f"Report exited with rc={rc}")
            return rc
        print()
        print("Smoke + report complete.  Inspect:")
        print(f"  {report_dir}/summary.csv")
        print(f"  {report_dir}/plots/sigma_data_drift.png")
        print(f"  {report_dir}/plots/wallclock_to_target.png")
        print(f"  {report_dir}/plots/headline_table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
