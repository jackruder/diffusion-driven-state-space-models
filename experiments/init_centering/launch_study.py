"""Generic sbatch launcher for the init-centering :class:`Study`.

One launcher, two modes:

- ``--mode tiny``  — every study point at the tiny size (the full ablation grid,
  12 cells × 2 datasets). Filter with ``--baseline-forms/-modes/-tracking``,
  ``--datasets``, or ``--cell``.
- ``--mode paper`` — the chosen ``--top-cells`` × datasets at the paper-headline
  size (``latent_dim`` ×2).

Data + dims are baked into each registered ``init_<cell>__<dataset>`` preset, so
the only per-job overrides are the Optuna sweep wiring + (paper) the size
override. Dry-run by default; ``--write-dir`` writes scripts; ``--submit`` (with
``--write-dir``) shells out to ``sbatch``.

Run::

    python -m experiments.init_centering.launch_study --mode tiny --dry-run
    python -m experiments.init_centering.launch_study --mode tiny \\
        --write-dir runs/sbatch/tiny_$(date +%Y%m%d) --submit
    python -m experiments.init_centering.launch_study --mode paper \\
        --top-cells init_mlp_pinned_per_t init_linear_learnable_fixed --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

from experiments._launch import (
    add_output_args,
    add_resource_args,
    cli_overrides_from_args,
    run_launcher,
    sweep_overrides,
)
from experiments._sbatch import render_sbatch
from experiments._study import StudyPoint
from experiments.init_centering.cells import (
    BASELINE_FORMS,
    BASELINE_MODES,
    TRACKING_MODES,
)
from experiments.init_centering.datasets import ABLATION_DATASETS
from experiments.init_centering.study import INIT_CENTERING_STUDY

_STUDY = INIT_CENTERING_STUDY
_DATASET_LABELS = [ds.label for ds in ABLATION_DATASETS]
_KNOWN_CELLS = sorted({p.tags["cell"] for p in _STUDY.points})


def _select_points(args: argparse.Namespace) -> list[StudyPoint]:
    if args.mode == "paper":
        pts = _STUDY.select(cell=set(args.top_cells))
        if args.datasets:
            pts = [p for p in pts if p.tags["dataset"] in set(args.datasets)]
        return pts
    # tiny
    filters: dict[str, object] = {}
    if args.cell:
        filters["cell"] = args.cell
    if args.baseline_forms:
        filters["baseline_form"] = set(args.baseline_forms)
    if args.baseline_modes:
        filters["baseline_mode"] = set(args.baseline_modes)
    if args.tracking_modes:
        filters["tracking_mode"] = set(args.tracking_modes)
    if args.datasets:
        filters["dataset"] = set(args.datasets)
    return _STUDY.select(**filters)


def _render(point: StudyPoint, args: argparse.Namespace, size: str) -> str:
    overrides = sweep_overrides(
        point.name,
        sweep_group=args.sweep_group,
        study_prefix=args.study_prefix,
        n_trials=args.n_trials,
        storage_dir=args.storage_dir,
        sweeps_root=args.sweeps_root,
        n_jobs=args.n_jobs,
    )
    overrides += point.size_overrides(size)
    if args.wallclock_target is not None:
        overrides.append(
            f"experiment.eval.kwargs.wallclock_to_target.target_value={args.wallclock_target}"
        )
    # Study points don't carry a per-point SBatch spec; fall back to the
    # project default (resolved inside render_sbatch from cli_overrides).
    return render_sbatch(
        point.name,
        exp_sbatch=None,
        hydra_overrides=overrides,
        cli_overrides=cli_overrides_from_args(args),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.init_centering.launch_study",
        description=(
            "Render Optuna-multirun sbatch scripts for the init-centering study. "
            "--mode tiny = the full grid at tiny size; --mode paper = --top-cells "
            "× datasets at the paper-headline (2× latent) size. Dry-run by default."
        ),
    )
    p.add_argument(
        "--mode", choices=["tiny", "paper"], default="tiny",
        help="tiny: the full ablation grid; paper: confirmation runs on --top-cells.",
    )
    add_output_args(p)
    # Selection.
    p.add_argument(
        "--cell", default=None,
        help="tiny: render a single cell (both datasets).",
    )
    p.add_argument(
        "--top-cells", nargs="+", default=None, metavar="CELL",
        help="paper: the top-N cells to confirm (e.g. init_mlp_pinned_per_t).",
    )
    p.add_argument(
        "--baseline-forms", nargs="+", default=None, choices=list(BASELINE_FORMS),
        help="Restrict to a subset of baseline forms (default: all).",
    )
    p.add_argument(
        "--baseline-modes", nargs="+", default=None, choices=list(BASELINE_MODES),
        help="Restrict to a subset of baseline modes (default: all).",
    )
    p.add_argument(
        "--tracking-modes", nargs="+", default=None, choices=list(TRACKING_MODES),
        help="Restrict to a subset of σ_data tracking modes (default: all).",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None, choices=_DATASET_LABELS,
        help="Restrict to a subset of datasets by label (default: all).",
    )
    # Sweep.
    p.add_argument(
        "--sweep-group", default=_STUDY.sweep,
        choices=["init_ablation", "init_ablation_moo"],
        help=(
            "Sweep config. 'init_ablation_moo' (default) is NSGA-II multi-objective "
            "matching the cells' PilotMOObjective; 'init_ablation' is single-objective."
        ),
    )
    p.add_argument("--study-prefix", default="ablation",
                   help="Prefix for Optuna study_name + SQLite filename (default 'ablation').")
    p.add_argument("--n-trials", type=int, default=40,
                   help="Optuna trials per (cell, dataset) job (default 40).")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Concurrent Optuna trials per study (default 1).")
    p.add_argument("--storage-dir", default="runs/optuna",
                   help="Directory for the per-job SQLite databases (default runs/optuna).")
    p.add_argument("--sweeps-root", default="runs/sweeps",
                   help="Root for the per-job sweep dirs (default runs/sweeps).")
    p.add_argument("--wallclock-target", type=float, default=None,
                   help="Override experiment.eval.kwargs.wallclock_to_target.target_value.")
    add_resource_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.submit and args.write_dir is None:
        parser.error("--submit requires --write-dir (it submits the written scripts).")
    if args.mode == "paper" and not args.top_cells:
        parser.error("--mode paper requires --top-cells.")
    if args.top_cells:
        bad = [c for c in args.top_cells if c not in _KNOWN_CELLS]
        if bad:
            parser.error(f"unknown cell(s): {', '.join(bad)}. Known: {', '.join(_KNOWN_CELLS)}")

    os.makedirs(args.storage_dir, exist_ok=True)
    os.makedirs(args.sweeps_root, exist_ok=True)

    size = "tiny" if args.mode == "tiny" else "paper"
    jobs = [(p.name, _render(p, args, size)) for p in _select_points(args)]
    return run_launcher(jobs, args=args)


if __name__ == "__main__":
    sys.exit(main())
