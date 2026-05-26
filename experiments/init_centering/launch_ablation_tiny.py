"""Render sbatch scripts for the tiny-size init-centering ablation grid.

The "tiny" ablation grid is the cell ranking phase: every cell of the
18-cell grid runs at the data's true latent dim (size matrix per
CONTEXT.md § Size axis) on both the 1D and MV synthetic datasets,
under the 7-dim ``init_ablation`` Optuna sweep with 40 trials each.

Per (cell, dataset) tuple → one sbatch job. Total: 18 × 2 = 36 jobs.

Run (dry-run by default, prints to stdout — nothing submitted)::

    python -m experiments.init_centering.launch_ablation_tiny --dry-run

Write to disk::

    python -m experiments.init_centering.launch_ablation_tiny \\
        --write-dir runs/sbatch/ablation_$(date +%Y%m%d) \\
        --study-prefix ablation_$(date +%Y%m%d)
    for f in runs/sbatch/ablation_*/*.sbatch; do sbatch "$f"; done

After all jobs complete, aggregate with::

    python -m experiments.init_centering.report all \\
        --sweeps-root runs/sweeps --optuna-dir runs/optuna \\
        --study-prefix ablation_$(date +%Y%m%d) --out runs/report/ablation

For the paper-headline confirmation runs on the top-N cells from the
ablation, see :mod:`.launch_paper_headline`.
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


# (dataset_preset_name, data_dim, latent_dim_at_tiny, label) — the
# ablation-grid runs every cell at both. Add new datasets here to
# extend the grid; CONTEXT.md § "Init-experiment datasets" is the
# source of truth.
TINY_DATASETS: tuple[tuple[str, int, int, str], ...] = (
    ("nonlin_bimodal_lift_1d", 1, 1, "1d"),
    ("nonlin_bimodal_lift_mv", 8, 4, "mv"),
)


def all_tiny_jobs() -> list[tuple[str, str, int, int, str]]:
    """Cross-product of the 18 cells × the two ablation datasets.

    Returns ``[(cell_name, dataset_preset, data_dim, latent_dim, label), ...]``.
    """
    out: list[tuple[str, str, int, int, str]] = []
    for f, m, t in iter_cells():
        cell = cell_name(f, m, t)
        for ds_name, data_dim, latent_dim, ds_label in TINY_DATASETS:
            out.append((cell, ds_name, data_dim, latent_dim, ds_label))
    return out


def _job_name(cell: str, ds_label: str) -> str:
    return f"{cell}__{ds_label}"


def _overrides_for_job(
    cell: str,
    ds_name: str,
    data_dim: int,
    latent_dim: int,
    ds_label: str,
    *,
    study_prefix: str,
    n_trials: int,
    storage_dir: str,
    sweeps_root: str,
    n_jobs: int = 1,
) -> list[str]:
    job = _job_name(cell, ds_label)
    sweep_dir = os.path.join(sweeps_root, f"{study_prefix}_{job}")
    db_path = os.path.join(storage_dir, f"{study_prefix}_{job}.db")
    overrides = [
        "--multirun",
        "+sweep=init_ablation",
        f"hydra.sweeper.n_trials={n_trials}",
        f"hydra.sweeper.study_name={study_prefix}_{job}",
        f"hydra.sweeper.storage=sqlite:///{db_path}",
        f"hydra.sweep.dir={sweep_dir}",
        f"experiment.data={ds_name}",
        f"experiment.model.data_dim={data_dim}",
        f"experiment.model.latent_dim={latent_dim}",
    ]
    if n_jobs > 1:
        overrides.append(f"hydra.sweeper.n_jobs={n_jobs}")
    return overrides


def _resolve_exp_sbatch(name: str):
    """Look up the named cell's optional ``SBatch`` resource spec."""
    register_experiments()
    from hydra_zen import store

    node = store["experiment"][("experiment", name)]
    exp = instantiate(node)
    return exp.sbatch


def render_tiny_sbatch(
    cell: str,
    ds_name: str,
    data_dim: int,
    latent_dim: int,
    ds_label: str,
    *,
    study_prefix: str,
    n_trials: int,
    storage_dir: str,
    sweeps_root: str,
    cli_overrides: dict[str, object] | None = None,
    n_jobs: int = 1,
) -> str:
    overrides = _overrides_for_job(
        cell, ds_name, data_dim, latent_dim, ds_label,
        study_prefix=study_prefix,
        n_trials=n_trials,
        storage_dir=storage_dir,
        sweeps_root=sweeps_root,
        n_jobs=n_jobs,
    )
    return render_sbatch(
        cell,
        exp_sbatch=_resolve_exp_sbatch(cell),
        hydra_overrides=overrides,
        cli_overrides=cli_overrides or {},
    )


def _iter_targets(
    only_cell: str | None,
    datasets: list[str] | None = None,
) -> Iterable[tuple[str, str, int, int, str]]:
    jobs = all_tiny_jobs()
    if datasets is not None:
        jobs = [j for j in jobs if j[4] in datasets]
    if only_cell is None:
        yield from jobs
        return
    matches = [j for j in jobs if j[0] == only_cell]
    if not matches:
        cells_listed = sorted({j[0] for j in jobs})
        raise SystemExit(
            f"Unknown cell {only_cell!r}. Known cells: {', '.join(cells_listed)}"
        )
    yield from matches


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.init_centering.launch_ablation_tiny",
        description=(
            "Render sbatch scripts for the tiny init-centering ablation grid "
            "(18 cells × 2 datasets = 36 jobs). Dry-run by default."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Print rendered sbatch scripts to stdout (default).",
    )
    mode.add_argument(
        "--write-dir", default=None,
        help="Write one '<cell>__<ds_label>.sbatch' per job into this directory.",
    )
    p.add_argument(
        "--cell", default=None,
        help="Render just one cell (both datasets).",
    )
    p.add_argument(
        "--study-prefix", default="ablation",
        help="Prefix for Optuna study_name + SQLite filename (default 'ablation').",
    )
    p.add_argument(
        "--n-trials", type=int, default=40,
        help="Optuna trials per (cell, dataset) job (default 40).",
    )
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help=(
            "Optuna trials run concurrently per study (default 1). "
            "Concurrent trials share the GPU; tune against memory + "
            "the per-trial footprint you observed locally."
        ),
    )
    p.add_argument(
        "--datasets", nargs="+", default=None,
        choices=[label for _, _, _, label in TINY_DATASETS],
        help=(
            "Restrict to a subset of datasets by label (default: all). "
            f"Choices: {[label for _, _, _, label in TINY_DATASETS]}."
        ),
    )
    p.add_argument(
        "--storage-dir", default="runs/optuna",
        help="Directory for the per-job SQLite databases (default runs/optuna).",
    )
    p.add_argument(
        "--sweeps-root", default="runs/sweeps",
        help=(
            "Root for the per-job sweep dirs. Each job lands at "
            "{sweeps_root}/{study_prefix}_{cell}__{ds_label}/{trial}/."
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

    os.makedirs(args.storage_dir, exist_ok=True)
    os.makedirs(args.sweeps_root, exist_ok=True)

    write_dir = args.write_dir
    if write_dir is not None:
        os.makedirs(write_dir, exist_ok=True)

    for cell, ds_name, data_dim, latent_dim, ds_label in _iter_targets(
        args.cell, datasets=args.datasets,
    ):
        script = render_tiny_sbatch(
            cell, ds_name, data_dim, latent_dim, ds_label,
            study_prefix=args.study_prefix,
            n_trials=args.n_trials,
            storage_dir=args.storage_dir,
            sweeps_root=args.sweeps_root,
            cli_overrides=cli_overrides,
            n_jobs=args.n_jobs,
        )
        job = _job_name(cell, ds_label)
        if write_dir is None:
            sys.stdout.write(f"# --- {job} ---\n{script}\n")
        else:
            path = os.path.join(write_dir, f"{job}.sbatch")
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
