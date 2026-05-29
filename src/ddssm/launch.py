"""Run a :class:`ddssm.study.Study` — per-point launch strategies + the orchestrator.

Where ``StageOrchestrator`` (``stages.py``) runs the *stages of one experiment*,
:class:`StudyOrchestrator` runs the *points of one study*. The launch **shape** is
per-point — a function of the point via ``Study.launch(point) -> PointLaunch`` —
so a 2-model compare emits single-GPU jobs while a ``j=1..16`` sweep can ask for
many nodes per cell. **Cross-point scheduling** (node-pool allocation, deadlines)
is intentionally NOT here; that is the ``plan-campaign`` skill, which *drives*
this class.

Backends: the sbatch path (dry-run / write / submit) renders one script per point
via the point's :class:`LaunchStrategy`; the local path runs each point as a
single subprocess (replacing the old ``smoke_phase_d``). Replication is a
``seeds`` knob, not a study axis.
"""

from __future__ import annotations

import abc
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field

from ddssm.experiment import SBatch
from ddssm.sbatch import render_sbatch, submit_sbatch
from ddssm.study import Study, StudyPoint

# A study point's resource ask reuses the per-experiment SBatch dataclass.
ResourceSpec = SBatch


# ---------------------------------------------------------------------------
# Per-point launch intent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointLaunch:
    """How to run ONE study point (returned by ``Study.launch(point)``).

    ``strategy`` names a registered :class:`LaunchStrategy` (the sbatch shape).
    ``resources`` (an ``SBatch``) supersedes the experiment's own ``sbatch`` for
    study launches; ``None`` falls back to the project default.
    """

    strategy: str = "optuna_single_node"
    sweep: str | None = None
    n_trials: int = 40
    n_jobs: int = 1
    resources: ResourceSpec | None = None
    extra_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchContext:
    study_prefix: str
    storage_dir: str
    sweeps_root: str
    seed: int | None = None


def _job_name(point: StudyPoint, ctx: LaunchContext) -> str:
    return point.name if ctx.seed is None else f"{point.name}__seed{ctx.seed}"


# ---------------------------------------------------------------------------
# Launch strategies — the sbatch shape per point
# ---------------------------------------------------------------------------


class LaunchStrategy(abc.ABC):
    """Produces the Hydra overrides for one point's sbatch (the sweep shape)."""

    name: str

    @abc.abstractmethod
    def hydra_overrides(self, point: StudyPoint, pl: PointLaunch, ctx: LaunchContext) -> list[str]:
        ...


class SingleJob(LaunchStrategy):
    """One job, one trial — no Optuna multirun."""

    name = "single_job"

    def hydra_overrides(self, point, pl, ctx):
        return []


class OptunaSingleNode(LaunchStrategy):
    """One Optuna multirun on a single node (its own SQLite study)."""

    name = "optuna_single_node"

    def hydra_overrides(self, point, pl, ctx):
        if not pl.sweep:
            raise ValueError(f"strategy {self.name!r} needs PointLaunch.sweep set")
        job = _job_name(point, ctx)
        sweep_dir = os.path.join(ctx.sweeps_root, f"{ctx.study_prefix}_{job}")
        db_path = os.path.join(ctx.storage_dir, f"{ctx.study_prefix}_{job}.db")
        overrides = [
            "--multirun",
            f"+sweep={pl.sweep}",
            f"hydra.sweeper.n_trials={pl.n_trials}",
            f"hydra.sweeper.study_name={ctx.study_prefix}_{job}",
            f"hydra.sweeper.storage=sqlite:///{db_path}",
            f"hydra.sweep.dir={sweep_dir}",
        ]
        if pl.n_jobs > 1:
            overrides.append(f"hydra.sweeper.n_jobs={pl.n_jobs}")
        return overrides


class _Stub(LaunchStrategy):
    def hydra_overrides(self, point, pl, ctx):
        raise NotImplementedError(
            f"the {self.name!r} launch strategy is a documented extension point "
            f"(ADR-0008) and is not implemented yet"
        )


class OptunaMultiNode(_Stub):
    """Optuna multirun across N nodes sharing an NFS-hosted DB. (stub)"""

    name = "optuna_multi_node"


class SlurmArray(_Stub):
    """One SLURM array task per point. (stub)"""

    name = "slurm_array"


_STRATEGIES: dict[str, LaunchStrategy] = {
    s.name: s for s in (SingleJob(), OptunaSingleNode(), OptunaMultiNode(), SlurmArray())
}


# ---------------------------------------------------------------------------
# Study registry (so the CLI can resolve a study by name)
# ---------------------------------------------------------------------------


STUDY_REGISTRY: dict[str, Study] = {}


def register_study(study: Study) -> Study:
    """Register a study so ``python -m ddssm.launch <name>`` can find it."""
    STUDY_REGISTRY[study.name] = study
    return study


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class StudyOrchestrator:
    """Runs the points of a study: render/submit sbatch, or run locally.

    Per-point launch intent comes from ``study.launch(point)``; pass
    ``launch_override`` to swap it at run time (e.g. force ``single_job`` for a
    local smoke). ``size`` selects a variant (``study.variants[size]``) whose
    overrides are applied to every point.
    """

    study: Study
    study_prefix: str = "study"
    storage_dir: str = "runs/optuna"
    sweeps_root: str = "runs/sweeps"

    def _variant_overrides(self, point: StudyPoint, size: str | None) -> list[str]:
        if size is None:
            return []
        if size not in self.study.variants:
            raise KeyError(
                f"study {self.study.name!r} has no variant {size!r}; "
                f"known: {sorted(self.study.variants)}"
            )
        return list(self.study.variants[size](point))

    def _point_launch(self, point, launch_override):
        return launch_override(point) if launch_override else self.study.launch(point)

    def render(self, points, *, size=None, seed=None, launch_override=None):
        """Return ``[(job_name, sbatch_text), ...]`` for the given points."""
        ctx = LaunchContext(self.study_prefix, self.storage_dir, self.sweeps_root, seed=seed)
        out: list[tuple[str, str]] = []
        for point in points:
            pl = self._point_launch(point, launch_override)
            strategy = _STRATEGIES[pl.strategy]
            overrides = strategy.hydra_overrides(point, pl, ctx)
            overrides += self._variant_overrides(point, size)
            overrides += list(pl.extra_overrides)
            if seed is not None:
                overrides.append(f"experiment.seed={seed}")
            script = render_sbatch(
                point.name, exp_sbatch=pl.resources, hydra_overrides=overrides
            )
            out.append((_job_name(point, ctx), script))
        return out

    def launch(self, points, *, size=None, seeds=(None,), write_dir=None,
               submit=False, launch_override=None) -> int:
        """Render (and optionally write/submit) sbatch for points × seeds."""
        if submit and write_dir is None:
            raise ValueError("submit=True requires write_dir")
        if write_dir is not None:
            os.makedirs(write_dir, exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.sweeps_root, exist_ok=True)
        submitted = 0
        for seed in seeds:
            for job, script in self.render(
                points, size=size, seed=seed, launch_override=launch_override
            ):
                if write_dir is None:
                    sys.stdout.write(f"# --- {job} ---\n{script}\n")
                    continue
                path = os.path.join(write_dir, f"{job}.sbatch")
                with open(path, "w") as f:
                    f.write(script)
                print(path)
                if submit:
                    print(submit_sbatch(path))
                    submitted += 1
        if write_dir is not None:
            tail = (f"# Submitted {submitted} job(s)." if submit
                    else f'# Submit all with: for f in {write_dir}/*.sbatch; do sbatch "$f"; done')
            print(f"\n{tail}", file=sys.stderr)
        return 0

    def run_local(self, points, *, size=None, seeds=(None,), out_dir="runs/local") -> int:
        """Run each (point × seed) as a single local subprocess (smoke/debug)."""
        failures: list[tuple[str, int]] = []
        for seed in seeds:
            ctx = LaunchContext(self.study_prefix, self.storage_dir, self.sweeps_root, seed=seed)
            for point in points:
                job = _job_name(point, ctx)
                run_dir = os.path.join(out_dir, f"{self.study_prefix}_{job}")
                os.makedirs(run_dir, exist_ok=True)
                cmd = [
                    sys.executable, "-m", "ddssm.app",
                    f"experiment={point.name}",
                    f"hydra.run.dir={run_dir}",
                    *self._variant_overrides(point, size),
                ]
                if seed is not None:
                    cmd.append(f"experiment.seed={seed}")
                print(f"[local] {job} ...", flush=True)
                rc = subprocess.run(cmd, check=False).returncode
                if rc != 0:
                    failures.append((job, rc))
        if failures:
            for job, rc in failures:
                print(f"  FAIL {job} (rc={rc})", file=sys.stderr)
            return 1
        return 0


# ---------------------------------------------------------------------------
# CLI: python -m ddssm.launch <study> [...]
# ---------------------------------------------------------------------------


def _load_studies() -> None:
    # Importing the experiment families triggers register_study(...) side effects.
    from ddssm._experiment_registry import register_experiments

    register_experiments()


def _parse_select(items: list[str] | None) -> dict[str, set[str]]:
    filters: dict[str, set[str]] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--select expects key=val, got {item!r}")
        k, v = item.split("=", 1)
        filters.setdefault(k, set()).add(v)
    return filters


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ddssm.launch")
    p.add_argument("study", help="registered study name (e.g. init_centering)")
    p.add_argument("--select", nargs="+", default=None, metavar="K=V",
                   help="filter points by tag, e.g. --select baseline_form=mlp dataset=mv")
    p.add_argument("--size", default=None, help="variant to apply (e.g. tiny/paper/smoke)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="replicate each point with these experiment.seed values")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print sbatch to stdout (default)")
    mode.add_argument("--write-dir", default=None, help="write one .sbatch per job here")
    mode.add_argument("--local", action="store_true", help="run each point locally (smoke/debug)")
    p.add_argument("--submit", action="store_true", help="submit written scripts (needs --write-dir)")
    p.add_argument("--study-prefix", default="study")
    p.add_argument("--storage-dir", default="runs/optuna")
    p.add_argument("--sweeps-root", default="runs/sweeps")
    p.add_argument("--out-dir", default="runs/local", help="--local run-dir root")
    args = p.parse_args(argv)

    if args.submit and args.write_dir is None:
        p.error("--submit requires --write-dir")

    _load_studies()
    # Under ``python -m ddssm.launch`` this file runs as ``__main__`` while the
    # families' ``register_study`` writes into the *imported* ``ddssm.launch``
    # module — a distinct copy. Reference the canonical module's registry.
    from ddssm.launch import STUDY_REGISTRY as registry

    if args.study not in registry:
        p.error(f"unknown study {args.study!r}; known: {', '.join(sorted(registry))}")
    study = registry[args.study]

    points = study.select(**_parse_select(args.select)) if args.select else list(study.points)
    seeds = tuple(args.seeds) if args.seeds else (None,)
    orch = StudyOrchestrator(
        study, study_prefix=args.study_prefix,
        storage_dir=args.storage_dir, sweeps_root=args.sweeps_root,
    )

    if args.local:
        # Local backend runs single jobs (no Optuna multirun) — for smoke/debug.
        return orch.run_local(points, size=args.size, seeds=seeds, out_dir=args.out_dir)
    return orch.launch(points, size=args.size, seeds=seeds,
                       write_dir=args.write_dir, submit=args.submit)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ResourceSpec", "PointLaunch", "LaunchStrategy", "StudyOrchestrator",
    "register_study", "STUDY_REGISTRY",
]
