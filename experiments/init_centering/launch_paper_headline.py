"""Render sbatch scripts for the paper-headline confirmation runs.

After the tiny ablation grid (:mod:`.launch_ablation_tiny`) finishes
and the Phase-E report picks the top-N cells per dataset, this
launcher re-runs *only those cells* at the **paper-headline** size
(latent_dim doubled vs tiny per CONTEXT.md § Size axis) with a bigger
Optuna trial budget (default 80) for the confirmation study.

Per (cell, dataset) tuple → one sbatch job; total = N × 2 (one per
dataset). N is whatever the user picks via ``--top-cells``.

Run (dry-run by default)::

    python -m experiments.init_centering.launch_paper_headline --dry-run \\
        --top-cells init_mlp_pinned_per_t init_mlp_learnable_per_t init_linear_pinned_global_ema

Write to disk::

    python -m experiments.init_centering.launch_paper_headline \\
        --top-cells init_mlp_pinned_per_t init_mlp_learnable_per_t \\
        --write-dir runs/sbatch/paper_$(date +%Y%m%d) \\
        --study-prefix paper_$(date +%Y%m%d)
    for f in runs/sbatch/paper_*/*.sbatch; do sbatch "$f"; done
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

from hydra_zen import instantiate

from ddssm._experiment_registry import register_experiments
from experiments._sbatch import render_sbatch
from experiments.init_centering.cells import cell_name as _cell_name
from experiments.init_centering.cells import iter_cells


# Mirror of TINY_DATASETS in launch_ablation_tiny, but with the
# paper-headline latent_dim (2× the data's true latent dim) and the
# observation dim unchanged. Same tuple shape as TINY_DATASETS:
# (preset_name, data_dim, latent_dim, label, mode, expose_gt_latents).
PAPER_DATASETS: tuple[tuple[str, int, int, str, str, bool], ...] = (
    ("nonlin_bimodal_lift_1d", 1, 2, "1d", "nonlinear-bimodal-lift", True),
    ("nonlin_bimodal_lift_mv", 8, 8, "mv", "nonlinear-bimodal-lift-mv", True),
)


def _validate_cells(cells: list[str]) -> None:
    known = {_cell_name(*c) for c in iter_cells()}
    bad = [c for c in cells if c not in known]
    if bad:
        raise SystemExit(
            f"Unknown cell(s): {', '.join(bad)}. Known cells: "
            f"{', '.join(sorted(known))}"
        )


def all_paper_jobs(
    top_cells: list[str],
) -> list[tuple[str, str, int, int, str, str, bool]]:
    """Cross-product of selected top cells × the two ablation datasets."""
    out: list[tuple[str, str, int, int, str, str, bool]] = []
    for cell in top_cells:
        for ds_name, data_dim, latent_dim, ds_label, mode, expose_gt in PAPER_DATASETS:
            out.append((cell, ds_name, data_dim, latent_dim, ds_label, mode, expose_gt))
    return out


def _job_name(cell: str, ds_label: str) -> str:
    return f"{cell}__{ds_label}"


def _overrides_for_job(
    cell: str,
    ds_name: str,
    data_dim: int,
    latent_dim: int,
    ds_label: str,
    mode: str,
    expose_gt: bool,
    *,
    study_prefix: str,
    n_trials: int,
    storage_dir: str,
    sweeps_root: str,
) -> list[str]:
    job = _job_name(cell, ds_label)
    sweep_dir = os.path.join(sweeps_root, f"{study_prefix}_{job}")
    db_path = os.path.join(storage_dir, f"{study_prefix}_{job}.db")
    return [
        "--multirun",
        "+sweep=init_ablation",
        f"hydra.sweeper.n_trials={n_trials}",
        f"hydra.sweeper.study_name={study_prefix}_{job}",
        f"hydra.sweeper.storage=sqlite:///{db_path}",
        f"hydra.sweep.dir={sweep_dir}",
        # Per-field data override (cell presets bake in data=Harmonic).
        f"experiment.data.mode={mode}",
        f"experiment.data.D={data_dim}",
        f"experiment.data.expose_gt_latents={'true' if expose_gt else 'false'}",
        f"experiment.model.data_dim={data_dim}",
        f"experiment.model.latent_dim={latent_dim}",
    ]


def _resolve_exp_sbatch(name: str):
    register_experiments()
    from hydra_zen import store

    node = store["experiment"][("experiment", name)]
    exp = instantiate(node)
    return exp.sbatch


def render_paper_sbatch(
    cell: str,
    ds_name: str,
    data_dim: int,
    latent_dim: int,
    ds_label: str,
    mode: str = "nonlinear-bimodal-lift",
    expose_gt: bool = True,
    *,
    study_prefix: str,
    n_trials: int,
    storage_dir: str,
    sweeps_root: str,
    cli_overrides: dict[str, object] | None = None,
) -> str:
    overrides = _overrides_for_job(
        cell, ds_name, data_dim, latent_dim, ds_label, mode, expose_gt,
        study_prefix=study_prefix,
        n_trials=n_trials,
        storage_dir=storage_dir,
        sweeps_root=sweeps_root,
    )
    return render_sbatch(
        cell,
        exp_sbatch=_resolve_exp_sbatch(cell),
        hydra_overrides=overrides,
        cli_overrides=cli_overrides or {},
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.init_centering.launch_paper_headline",
        description=(
            "Render sbatch scripts for the paper-headline confirmation runs "
            "on the user-selected top-N cells. Dry-run by default."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print to stdout (default).")
    mode.add_argument(
        "--write-dir", default=None,
        help="Write one '<cell>__<ds_label>.sbatch' per job into this directory.",
    )
    p.add_argument(
        "--top-cells", nargs="+", required=True, metavar="CELL",
        help="The N top cells from the ablation (variable arity). Example: "
             "--top-cells init_mlp_pinned_per_t init_linear_learnable_global_ema",
    )
    p.add_argument(
        "--study-prefix", default="paper",
        help="Prefix for Optuna study_name + SQLite filename (default 'paper').",
    )
    p.add_argument(
        "--n-trials", type=int, default=80,
        help="Optuna trials per (cell, dataset) job (default 80).",
    )
    p.add_argument(
        "--storage-dir", default="runs/optuna",
        help="Directory for the per-job SQLite databases (default runs/optuna).",
    )
    p.add_argument(
        "--sweeps-root", default="runs/sweeps",
        help="Root for the per-job sweep dirs.",
    )
    p.add_argument("--partition", default=None)
    p.add_argument("--time", default=None)
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument("--cpus", type=int, default=None)
    p.add_argument("--mem", default=None)
    p.add_argument("--nodes", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_cells(args.top_cells)

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

    for cell, ds_name, data_dim, latent_dim, ds_label, mode, expose_gt in all_paper_jobs(args.top_cells):
        script = render_paper_sbatch(
            cell, ds_name, data_dim, latent_dim, ds_label, mode, expose_gt,
            study_prefix=args.study_prefix,
            n_trials=args.n_trials,
            storage_dir=args.storage_dir,
            sweeps_root=args.sweeps_root,
            cli_overrides=cli_overrides,
        )
        job = _job_name(cell, ds_label)
        if write_dir is None:
            sys.stdout.write(f"# --- {job} ---\n{script}\n")
        else:
            path = os.path.join(write_dir, f"{job}.sbatch")
            with open(path, "w") as f:
                f.write(script)
            print(path)

    if write_dir is not None:
        print(
            f"\n# Submit all with: for f in {write_dir}/*.sbatch; "
            f'do sbatch "$f"; done',
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
