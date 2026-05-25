"""Phase-D sbatch launcher — one SLURM job per cell of the 18-cell grid.

Each cell becomes one Optuna multirun (20 trials by default) wired to
its own SQLite study so the 18 sweeps run concurrently without
trampling each other.  The two Phase-D control cells
(``sigma_pert=0``, ``n_pretrain=0``) submit as single (non-multirun)
jobs because they have nothing to sweep over.

Reuses :func:`experiments._sbatch.render_sbatch` for the resource
block — ``--partition``, ``--time``, ``--gpus``, etc. fall through
the same default → experiment-level → CLI-override merge as
``python -m experiments sbatch <name>``.

Usage
-----

Print all 20 sbatch scripts to stdout (default — never submits)::

    python -m experiments.init_centering.launch_phase_d --dry-run

Write scripts to a directory::

    python -m experiments.init_centering.launch_phase_d \\
        --write-dir runs/sbatch/phase_d_$(date +%Y%m%d)

Just one cell::

    python -m experiments.init_centering.launch_phase_d \\
        --cell init_mlp_pinned_per_t --dry-run

Custom resources or sweep size::

    python -m experiments.init_centering.launch_phase_d \\
        --write-dir runs/sbatch/phase_d_overnight \\
        --n-trials 40 --time 12:00:00

The launcher *does not call ``sbatch``*.  Submission is the user's job.
After ``--write-dir`` finishes, the printed footer tells you exactly
the loop to run::

    for f in runs/sbatch/phase_d_*/*.sbatch; do sbatch "$f"; done
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

from hydra_zen import instantiate

from ddssm._experiment_registry import register_experiments

from experiments._sbatch import render_sbatch
from experiments.init_centering.cells import cell_name, iter_cells


CONTROL_CELLS: tuple[str, ...] = (
    "init_canonical_ctrl_sigma0",
    "init_canonical_ctrl_npretrain0",
)


def all_phase_d_cells() -> list[str]:
    """The 20 named presets Phase D submits: 18 grid cells + 2 controls."""
    return [cell_name(*c) for c in iter_cells()] + list(CONTROL_CELLS)


def _overrides_for_cell(
    name: str,
    *,
    study_prefix: str,
    n_trials: int,
    storage_dir: str,
    sweeps_root: str,
) -> tuple[list[str], bool]:
    """Build the Hydra overrides + multirun flag for a cell or control.

    Controls run as single jobs (no Optuna sweep); cells run as
    multirun + ``+sweep=init_pilot``.  Each gets a cell-scoped
    ``study_name`` and SQLite path so the sweeps don't share state.

    The cell's sweep dir is pinned at
    ``{sweeps_root}/{study_prefix}_{cell}/`` so Phase-E aggregation
    (see :mod:`.report`) can deterministically discover trial
    metrics.json paths.  Control runs reuse the same dir for the
    single-job output.
    """
    sweep_dir = os.path.join(sweeps_root, f"{study_prefix}_{name}")
    if name in CONTROL_CELLS:
        # Single-job (non-multirun) control: ``hydra.run.dir`` is the path
        # Hydra uses for non-multirun jobs.
        return [f"hydra.run.dir={sweep_dir}"], False

    db_path = os.path.join(storage_dir, f"{study_prefix}_{name}.db")
    overrides = [
        "--multirun",
        "+sweep=init_pilot",
        f"hydra.sweeper.n_trials={n_trials}",
        f"hydra.sweeper.study_name={study_prefix}_{name}",
        f"hydra.sweeper.storage=sqlite:///{db_path}",
        f"hydra.sweep.dir={sweep_dir}",
    ]
    return overrides, True


def _resolve_exp_sbatch(name: str):
    """Look up the named experiment's optional ``SBatch`` resource spec."""
    register_experiments()
    from hydra_zen import store

    node = store["experiment"][("experiment", name)]
    exp = instantiate(node)
    return exp.sbatch


def render_phase_d_sbatch(
    name: str,
    *,
    study_prefix: str,
    n_trials: int,
    storage_dir: str,
    sweeps_root: str,
    cli_overrides: dict[str, object] | None = None,
) -> str:
    """Render a single sbatch script for one Phase-D cell (or control)."""
    overrides, _is_multirun = _overrides_for_cell(
        name,
        study_prefix=study_prefix,
        n_trials=n_trials,
        storage_dir=storage_dir,
        sweeps_root=sweeps_root,
    )
    return render_sbatch(
        name,
        exp_sbatch=_resolve_exp_sbatch(name),
        hydra_overrides=overrides,
        cli_overrides=cli_overrides or {},
    )


def _iter_targets(only: str | None) -> Iterable[str]:
    if only is None:
        yield from all_phase_d_cells()
        return
    if only not in all_phase_d_cells():
        raise SystemExit(
            f"Unknown Phase-D cell {only!r}.  Known: "
            f"{', '.join(all_phase_d_cells())}"
        )
    yield only


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.init_centering.launch_phase_d",
        description=(
            "Render sbatch scripts for the Phase-D 18-cell grid + 2 controls. "
            "Default is --dry-run: prints scripts to stdout; nothing is submitted."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Print rendered sbatch scripts to stdout (default).",
    )
    mode.add_argument(
        "--write-dir", default=None,
        help="Write one '<name>.sbatch' file per cell into this directory.",
    )
    p.add_argument(
        "--cell", default=None,
        help="Render just one cell (e.g. 'init_mlp_pinned_per_t').",
    )
    p.add_argument(
        "--study-prefix", default="phase_d",
        help="Prefix for Optuna study_name + SQLite filename (default 'phase_d').",
    )
    p.add_argument(
        "--n-trials", type=int, default=20,
        help="Optuna trials per cell (default 20, ignored for controls).",
    )
    p.add_argument(
        "--storage-dir", default="runs/optuna",
        help="Directory for the per-cell SQLite databases (default runs/optuna).",
    )
    p.add_argument(
        "--sweeps-root", default="runs/sweeps",
        help=(
            "Root for the per-cell sweep dirs.  Each cell's output lands at "
            "{sweeps_root}/{study_prefix}_{cell}/{trial}/ — matching the "
            "layout Phase-E ``report.py`` aggregates from (default runs/sweeps)."
        ),
    )
    # CLI passthrough for the sbatch resource block.
    p.add_argument("--partition", default=None)
    p.add_argument("--time", default=None)
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument("--cpus", type=int, default=None)
    p.add_argument("--mem", default=None)
    p.add_argument("--nodes", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cli_overrides = {
        "partition": args.partition,
        "time": args.time,
        "gpus": args.gpus,
        "cpus": args.cpus,
        "mem": args.mem,
        "nodes": args.nodes,
    }

    # Ensure the storage + sweeps dirs exist so the jobs can open their files.
    os.makedirs(args.storage_dir, exist_ok=True)
    os.makedirs(args.sweeps_root, exist_ok=True)

    write_dir = args.write_dir
    if write_dir is not None:
        os.makedirs(write_dir, exist_ok=True)

    for name in _iter_targets(args.cell):
        script = render_phase_d_sbatch(
            name,
            study_prefix=args.study_prefix,
            n_trials=args.n_trials,
            storage_dir=args.storage_dir,
            sweeps_root=args.sweeps_root,
            cli_overrides=cli_overrides,
        )
        if write_dir is None:
            sys.stdout.write(f"# --- {name} ---\n{script}\n")
        else:
            path = os.path.join(write_dir, f"{name}.sbatch")
            with open(path, "w") as f:
                f.write(script)
            print(path)

    if write_dir is not None and args.cell is None:
        print(
            f"\n# Submit all with: for f in {write_dir}/*.sbatch; "
            f'do sbatch "$f"; done',
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
