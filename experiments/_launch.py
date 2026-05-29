"""Shared launcher machinery: argparse helpers + the render/write/submit loop.

The per-study launchers (``experiments/<family>/launch_*.py``) reuse this so the
resource block, sweep-override block, and the dry-run / ``--write-dir`` /
``--submit`` loop live in one place. Sbatch rendering + submission delegate to
:mod:`experiments._sbatch`.
"""

from __future__ import annotations

import argparse
import os
import sys

from experiments._sbatch import submit_sbatch


def add_resource_args(p: argparse.ArgumentParser) -> None:
    """The SBatch resource overrides shared by every launcher."""
    p.add_argument("--partition", default=None)
    p.add_argument("--time", default=None)
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument("--cpus", type=int, default=None)
    p.add_argument("--mem", default=None)
    p.add_argument("--nodes", type=int, default=None)


def cli_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        k: getattr(args, k)
        for k in ("partition", "time", "gpus", "cpus", "mem", "nodes")
    }


def add_output_args(p: argparse.ArgumentParser) -> None:
    """``--dry-run`` (default) vs ``--write-dir``, plus ``--submit``."""
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Print rendered sbatch scripts to stdout (default).",
    )
    mode.add_argument(
        "--write-dir", default=None,
        help="Write one '<job>.sbatch' per job into this directory.",
    )
    p.add_argument(
        "--submit", action="store_true",
        help="After writing, submit each script via 'sbatch' (requires --write-dir).",
    )


def sweep_overrides(
    job: str,
    *,
    sweep_group: str,
    study_prefix: str,
    n_trials: int,
    storage_dir: str,
    sweeps_root: str,
    n_jobs: int = 1,
) -> list[str]:
    """The Optuna multirun + study/storage/sweep-dir override block for one job."""
    sweep_dir = os.path.join(sweeps_root, f"{study_prefix}_{job}")
    db_path = os.path.join(storage_dir, f"{study_prefix}_{job}.db")
    overrides = [
        "--multirun",
        f"+sweep={sweep_group}",
        f"hydra.sweeper.n_trials={n_trials}",
        f"hydra.sweeper.study_name={study_prefix}_{job}",
        f"hydra.sweeper.storage=sqlite:///{db_path}",
        f"hydra.sweep.dir={sweep_dir}",
    ]
    if n_jobs > 1:
        overrides.append(f"hydra.sweeper.n_jobs={n_jobs}")
    return overrides


def run_launcher(jobs: list[tuple[str, str]], *, args: argparse.Namespace) -> int:
    """Emit ``jobs`` (``(job_name, rendered_script)`` pairs) per the output mode.

    Default (``--dry-run``) prints to stdout; ``--write-dir`` writes one file
    per job; ``--submit`` additionally shells out to ``sbatch`` per file.
    """
    write_dir = args.write_dir
    if write_dir is not None:
        os.makedirs(write_dir, exist_ok=True)

    submitted = 0
    for job, script in jobs:
        if write_dir is None:
            sys.stdout.write(f"# --- {job} ---\n{script}\n")
        else:
            path = os.path.join(write_dir, f"{job}.sbatch")
            with open(path, "w") as f:
                f.write(script)
            print(path)
            if args.submit:
                print(submit_sbatch(path))
                submitted += 1

    if write_dir is not None:
        if args.submit:
            print(f"\n# Submitted {submitted} job(s).", file=sys.stderr)
        else:
            print(
                f"\n# Submit all with: for f in {write_dir}/*.sbatch; "
                f'do sbatch "$f"; done',
                file=sys.stderr,
            )
    return 0


__all__ = [
    "add_resource_args",
    "add_output_args",
    "cli_overrides_from_args",
    "sweep_overrides",
    "run_launcher",
]
