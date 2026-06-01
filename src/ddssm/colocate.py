"""Co-located multi-cell Optuna launcher — ``python -m ddssm.colocate``.

Where ``ddssm.launch`` emits one job per cell (a cell's workers own a GPU), this
packs EVERY selected cell onto each of ``--n-gpus`` GPUs with ``--workers-per-cell``
workers apiece. A cell's total concurrency is therefore
``--workers-per-cell × --n-gpus`` against one shared study, while each GPU hosts
``cells × --workers-per-cell`` processes.

Its reason to exist: ADD trials to an *existing* shared-DB study at LOW per-cell
concurrency so NSGA-II can actually evolve generation-to-generation, without
dedicating a whole GPU to each cell. Point it at the same ``--storage-url`` and
``--study-prefix`` the study was launched with; ``--target`` is the per-cell TOTAL
budget (existing COMPLETE trials count toward it via each cell's preempt preamble),
so it converges to ``target`` instead of over-running on requeue.

The b6000 (or other) resource ask is inherited from a TEMPLATE cell's
``study.launch(point).resources`` (``--resources-from``) — so Tempest specifics
(setup, account, gres) stay defined in the study, not duplicated here — with
``--cpus`` / ``--mem`` / ``--time`` overridable. Pick a template cell whose
launch lands on the GPU type you want (e.g. a non-headline b6000 cell, NOT the
A100-routed headline).

Example (add to round2, 3 b6000 GPUs, 2 workers/cell/GPU = 6 concurrent/cell)::

    python -m ddssm.colocate init_centering \\
        --select dataset=mv tracking_mode=per_t \\
        --n-gpus 3 --workers-per-cell 2 --target 96 \\
        --sweep init_ablation_moo_r2 \\
        --storage-url postgresql://ddssm@epyc001:5432/round1 \\
        --study-prefix round2_20260531 --sweeps-root sweeps \\
        --resources-from init_linear_learnable_per_t__mv --cpus 32 \\
        --write-dir runs/sbatch/colo --submit --stagger 60
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import dataclasses

from ddssm.launch import (
    StudyOrchestrator,
    _load_studies,
    _parse_select,
)
from ddssm.sbatch import DEFAULT_SBATCH, submit_sbatch


def _resolve_template(study, points, resources_from: str | None):
    """The resource template: ``--resources-from`` cell's launch resources, else
    the first selected point's. Falls back to the project default if the point's
    launch carries no ``resources`` (so a study without per-point resources still
    renders).
    """
    point = points[0]
    if resources_from is not None:
        point = next((p for p in points if p.name == resources_from), None)
        if point is None:
            raise SystemExit(
                f"--resources-from {resources_from!r} is not among the selected "
                f"points: {', '.join(p.name for p in points)}"
            )
    return study.launch(point).resources or DEFAULT_SBATCH


def main(argv: list[str] | None = None) -> int:
    """CLI entry: render one co-located sbatch per GPU for a study's cells.

    Resolves the study, picks a resource template, then renders/writes/submits
    the per-GPU packed jobs. Returns the process exit code (0 on success).
    """
    p = argparse.ArgumentParser(prog="python -m ddssm.colocate")
    p.add_argument("study", help="registered study name (e.g. init_centering)")
    p.add_argument(
        "--select",
        nargs="+",
        default=None,
        metavar="K=V",
        help="filter cells by tag, e.g. --select dataset=mv tracking_mode=per_t",
    )
    p.add_argument(
        "--n-gpus",
        type=int,
        required=True,
        help="number of GPUs; every cell runs on each one",
    )
    p.add_argument(
        "--workers-per-cell",
        type=int,
        default=2,
        help="workers per (cell, GPU); per-cell concurrency = this x --n-gpus",
    )
    p.add_argument(
        "--target",
        type=int,
        required=True,
        help="per-cell TOTAL trial budget (existing trials count toward it)",
    )
    p.add_argument(
        "--sweep", default="init_ablation_moo_r2", help="registered sweep preset"
    )
    p.add_argument(
        "--storage-url",
        required=True,
        help="shared Optuna storage URL (e.g. postgresql://host/db)",
    )
    p.add_argument(
        "--study-prefix",
        required=True,
        help="MUST match the existing study's prefix to add to it",
    )
    p.add_argument("--storage-dir", default="runs/optuna")
    p.add_argument("--sweeps-root", default="runs/sweeps")
    p.add_argument(
        "--resources-from",
        default=None,
        metavar="CELL",
        help="cell whose launch resources are the template (default: first selected)",
    )
    p.add_argument(
        "--cpus", type=int, default=32, help="cpus-per-task for each GPU job"
    )
    p.add_argument("--mem", default=None, help="override the template mem (e.g. 80G)")
    p.add_argument(
        "--time", default=None, help="override the template walltime (e.g. 48:00:00)"
    )
    p.add_argument(
        "--no-preempt",
        action="store_true",
        help="render without requeue/launch_remaining (one-shot)",
    )
    p.add_argument(
        "--grace", type=int, default=180, help="preempt signal lead-time (s)"
    )
    p.add_argument("--size", default=None, help="variant to apply (e.g. smoke)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="print sbatch to stdout (default)"
    )
    mode.add_argument(
        "--write-dir", default=None, help="write one .sbatch per GPU here"
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="submit written scripts (needs --write-dir)",
    )
    p.add_argument(
        "--stagger",
        type=int,
        default=0,
        help="seconds to sleep between submits (lets the DB settle between waves)",
    )
    args = p.parse_args(argv)

    if args.submit and args.write_dir is None:
        p.error("--submit requires --write-dir")

    _load_studies()
    from ddssm.launch import STUDY_REGISTRY as registry

    if args.study not in registry:
        p.error(f"unknown study {args.study!r}; known: {', '.join(sorted(registry))}")
    study = registry[args.study]

    points = (
        study.select(**_parse_select(args.select))
        if args.select
        else list(study.points)
    )
    if not points:
        p.error("selection matched no cells")

    template = _resolve_template(study, points, args.resources_from)
    resources = dataclasses.replace(
        template,
        cpus=args.cpus,
        mem=args.mem or template.mem,
        time=args.time or template.time,
        job_name=None,
    )

    orch = StudyOrchestrator(
        study,
        study_prefix=args.study_prefix,
        storage_dir=args.storage_dir,
        sweeps_root=args.sweeps_root,
        storage_url=args.storage_url,
    )
    jobs = orch.render_colocated(
        points,
        n_gpus=args.n_gpus,
        workers_per_cell_per_gpu=args.workers_per_cell,
        target=args.target,
        sweep=args.sweep,
        resources=resources,
        preemptive=not args.no_preempt,
        grace_seconds=args.grace,
        size=args.size,
    )

    if args.write_dir is None:
        for job, script in jobs:
            sys.stdout.write(f"# --- {job} ---\n{script}\n")
        return 0

    # Init the shared schema once before any job's launch_remaining / worker
    # touches it (mirrors StudyOrchestrator._precreate_storage for shared URLs).
    if args.submit:
        from optuna.storages import RDBStorage

        RDBStorage(args.storage_url)

    os.makedirs(args.write_dir, exist_ok=True)
    paths: list[str] = []
    for job, script in jobs:
        path = os.path.join(args.write_dir, f"{job}.sbatch")
        with open(path, "w") as f:
            f.write(script)
        print(path)
        paths.append(path)

    if args.submit:
        for idx, path in enumerate(paths):
            if idx and args.stagger:
                time.sleep(args.stagger)
            print(submit_sbatch(path))
    else:
        print(
            f"\n# Submit (staggered) with: "
            f'for f in {args.write_dir}/*.sbatch; do sbatch "$f"; sleep 60; done',
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
