"""Render sbatch scripts for the tiny-size init-centering ablation grid.

The "tiny" ablation grid is the cell ranking phase: every cell of the
ablation grid runs at the data's true latent dim (size matrix per
CONTEXT.md § Size axis) on both the 1D and MV synthetic datasets,
under the 7-dim ``init_ablation`` Optuna sweep with 40 trials each.

Per (cell, dataset) tuple → one sbatch job.

Axis filters (``--baseline-forms``, ``--baseline-modes``,
``--tracking-modes``) intersect with :func:`iter_cells` so a round can
be gated to a subset — e.g. ``--baseline-modes pinned`` runs only the
parameter-free (zero/identity) and pinned-μ_p cells.

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

import os
import sys
from typing import Iterable
import argparse

from hydra_zen import instantiate

from experiments._sbatch import render_sbatch
from ddssm._experiment_registry import register_experiments
from experiments.init_centering.cells import (
    BASELINE_FORMS,
    BASELINE_MODES,
    TRACKING_MODES,
    cell_name,
    iter_cells,
)

# (preset_name, data_dim, latent_dim_at_tiny, label, mode_string,
# expose_gt_latents). The grid runs every cell at every dataset
# entry. ``mode_string`` is the underlying SyntheticDataset mode
# (CLI override `experiment.data.mode=...`). The cell presets bake in
# ``data=Harmonic``; we swap to the ablation mode by per-field
# override since the cell preset doesn't use a defaults list.
# Add new datasets here to extend the grid; CONTEXT.md § "Init-experiment
# datasets" is the source of truth.
TINY_DATASETS: tuple[tuple[str, int, int, str, str, bool], ...] = (
    ("nonlin_bimodal_lift_1d", 1, 1, "1d", "nonlinear-bimodal-lift", True),
    ("nonlin_bimodal_lift_mv", 8, 4, "mv", "nonlinear-bimodal-lift-mv", True),
)


def all_tiny_jobs(
    *,
    baseline_forms: Iterable[str] | None = None,
    baseline_modes: Iterable[str] | None = None,
    tracking_modes: Iterable[str] | None = None,
) -> list[tuple[str, str, int, int, str, str, bool]]:
    """Cross-product of (filtered) cells × the two ablation datasets.

    ``baseline_forms`` / ``baseline_modes`` / ``tracking_modes``
    intersect with :func:`iter_cells`; ``None`` on any axis means
    "keep every value on that axis".

    Returns:
    ``[(cell_name, dataset_preset, data_dim, latent_dim, label, mode, expose_gt), ...]``.
    """
    bf = None if baseline_forms is None else frozenset(baseline_forms)
    bm = None if baseline_modes is None else frozenset(baseline_modes)
    tm = None if tracking_modes is None else frozenset(tracking_modes)

    out: list[tuple[str, str, int, int, str, str, bool]] = []
    for f, m, t in iter_cells():
        if bf is not None and f not in bf:
            continue
        if bm is not None and m not in bm:
            continue
        if tm is not None and t not in tm:
            continue
        cell = cell_name(f, m, t)
        for ds_name, data_dim, latent_dim, ds_label, mode, expose_gt in TINY_DATASETS:
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
    n_jobs: int = 1,
    sweep_group: str = "init_ablation",
    wallclock_target: float | None = None,
) -> list[str]:
    job = _job_name(cell, ds_label)
    sweep_dir = os.path.join(sweeps_root, f"{study_prefix}_{job}")
    db_path = os.path.join(storage_dir, f"{study_prefix}_{job}.db")
    overrides = [
        "--multirun",
        f"+sweep={sweep_group}",
        f"hydra.sweeper.n_trials={n_trials}",
        f"hydra.sweeper.study_name={study_prefix}_{job}",
        f"hydra.sweeper.storage=sqlite:///{db_path}",
        f"hydra.sweep.dir={sweep_dir}",
        # Per-field data override (cell presets bake in ``data=Harmonic``;
        # we mutate the fields rather than swap the whole subtree by name).
        f"experiment.data.mode={mode}",
        f"experiment.data.D={data_dim}",
        f"experiment.data.expose_gt_latents={'true' if expose_gt else 'false'}",
        f"experiment.model.data_dim={data_dim}",
        f"experiment.model.latent_dim={latent_dim}",
    ]
    if n_jobs > 1:
        overrides.append(f"hydra.sweeper.n_jobs={n_jobs}")
    if wallclock_target is not None:
        # The init_centering eval already lists ``wallclock_to_target``;
        # this override changes its target value for the round-1 sweep.
        overrides.append(
            f"experiment.eval.kwargs.wallclock_to_target.target_value={wallclock_target}"
        )
    return overrides


def _resolve_exp_sbatch(name: str):
    """Look up the named cell's optional ``SBatch`` resource spec."""
    register_experiments()
    from hydra_zen import store

    node = store["experiment"]["experiment", name]
    exp = instantiate(node)
    return exp.sbatch


def render_tiny_sbatch(
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
    n_jobs: int = 1,
    sweep_group: str = "init_ablation",
    wallclock_target: float | None = None,
) -> str:
    overrides = _overrides_for_job(
        cell, ds_name, data_dim, latent_dim, ds_label, mode, expose_gt,
        study_prefix=study_prefix,
        n_trials=n_trials,
        storage_dir=storage_dir,
        sweeps_root=sweeps_root,
        n_jobs=n_jobs,
        sweep_group=sweep_group,
        wallclock_target=wallclock_target,
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
    *,
    baseline_forms: list[str] | None = None,
    baseline_modes: list[str] | None = None,
    tracking_modes: list[str] | None = None,
) -> Iterable[tuple[str, str, int, int, str, str, bool]]:
    jobs = all_tiny_jobs(
        baseline_forms=baseline_forms,
        baseline_modes=baseline_modes,
        tracking_modes=tracking_modes,
    )
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
            "(every cell × every dataset). Subset via --baseline-forms / "
            "--baseline-modes / --tracking-modes. Dry-run by default."
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
        "--baseline-forms", nargs="+", default=None,
        choices=list(BASELINE_FORMS),
        help=(
            "Restrict to a subset of baseline forms (default: all). "
            f"Choices: {list(BASELINE_FORMS)}."
        ),
    )
    p.add_argument(
        "--baseline-modes", nargs="+", default=None,
        choices=list(BASELINE_MODES),
        help=(
            "Restrict to a subset of baseline modes (default: all). "
            f"Choices: {list(BASELINE_MODES)}."
        ),
    )
    p.add_argument(
        "--tracking-modes", nargs="+", default=None,
        choices=list(TRACKING_MODES),
        help=(
            "Restrict to a subset of σ_data tracking modes (default: all). "
            f"Choices: {list(TRACKING_MODES)}."
        ),
    )
    p.add_argument(
        "--sweep-group", default="init_ablation",
        choices=["init_ablation", "init_ablation_moo"],
        help=(
            "Which registered sweep config to use. "
            "'init_ablation' = single-objective TPE on stage2_elbo_surrogate; "
            "'init_ablation_moo' = NSGA-II multi-objective on "
            "(wallclock_to_target_seconds, stage2_elbo_surrogate). "
            "MOO requires the cell experiments to use PilotMOObjective."
        ),
    )
    p.add_argument(
        "--wallclock-target", type=float, default=None,
        help=(
            "Override the ELBO target for the wallclock_to_target eval "
            "metric. Sets "
            "experiment.eval.kwargs.wallclock_to_target.target_value=<float>. "
            "Defaults to the cell preset's bundled value "
            "(PILOT_WALLCLOCK_TARGET = -30 at the time of writing)."
        ),
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
        choices=[entry[3] for entry in TINY_DATASETS],
        help=(
            "Restrict to a subset of datasets by label (default: all). "
            f"Choices: {[entry[3] for entry in TINY_DATASETS]}."
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

    for cell, ds_name, data_dim, latent_dim, ds_label, mode, expose_gt in _iter_targets(
        args.cell,
        datasets=args.datasets,
        baseline_forms=args.baseline_forms,
        baseline_modes=args.baseline_modes,
        tracking_modes=args.tracking_modes,
    ):
        # Per-call extras so the existing render_tiny_sbatch signature
        # stays back-compat — we add MOO + target overrides only when set.
        _extra_kwargs: dict = {
            "sweep_group": args.sweep_group,
        }
        if args.wallclock_target is not None:
            _extra_kwargs["wallclock_target"] = args.wallclock_target
        script = render_tiny_sbatch(
            cell, ds_name, data_dim, latent_dim, ds_label, mode, expose_gt,
            study_prefix=args.study_prefix,
            n_trials=args.n_trials,
            storage_dir=args.storage_dir,
            sweeps_root=args.sweeps_root,
            cli_overrides=cli_overrides,
            n_jobs=args.n_jobs,
            **_extra_kwargs,
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
