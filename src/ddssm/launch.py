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

import os
import abc
import sys
import math
import argparse
import subprocess
import dataclasses
from dataclasses import dataclass

from ddssm.study import Study, StudyPoint
from ddssm.sbatch import (
    CellWorker,
    PreemptSpec,
    render_sbatch,
    submit_sbatch,
    render_packed_sbatch,
    render_multicell_packed_sbatch,
)
from ddssm.experiment import SBatch

# Placeholder emitted by preemptive strategies; the orchestrator substitutes
# it per-backend (shell ``$N_PER_WORKER`` for sbatch; literal
# ``ceil(n_trials/n_workers)`` for the local backend).
_N_PER_WORKER_PLACEHOLDER = "__N_PER_WORKER__"

# Per-worker NSGA-II sampler seed. Without this, every packed/multi worker builds
# its sampler from the SAME ``hydra.sweeper.sampler.seed`` (42 in the yaml) and so
# draws the IDENTICAL trial sequence — a cell's "N trials" collapse to a handful
# of repeated configs, because at small per-worker budgets the gen-0 random
# population is never escaped (observed: 30 COMPLETE trials = 2 distinct configs).
# The value is a PLACEHOLDER the renderers replace with a distinct seed per worker:
# the sbatch paths substitute a bash arithmetic expansion over ``$SLURM_JOB_ID``
# (so the int reaches Hydra directly — env-var interpolation like
# ``${oc.env:...}`` does NOT parse through Hydra's override grammar), and
# ``run_local`` substitutes ``idx+1``. See ``_SAMPLER_SEED_PLACEHOLDER`` in sbatch.py.
_SAMPLER_SEED_PLACEHOLDER = "__SAMPLER_SEED__"
_SAMPLER_SEED_OVERRIDE = f"hydra.sweeper.sampler.seed={_SAMPLER_SEED_PLACEHOLDER}"

# A study point's resource ask reuses the per-experiment SBatch dataclass.
ResourceSpec = SBatch


# ---------------------------------------------------------------------------
# Per-point launch intent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointLaunch:
    """How to run ONE study point (returned by ``Study.launch(point)``).

    ``strategy`` names a registered :class:`LaunchStrategy` (the sweep shape).
    ``resources`` (an ``SBatch``) supersedes the experiment's own ``sbatch`` for
    study launches; ``None`` falls back to the project default. ``n_workers``
    is the per-point worker count for multi-worker strategies (multi-node sbatch
    sharing an NFS DB, or local subprocesses sharing a local SQLite DB); ignored
    by single-worker strategies.

    ``n_trials`` is the TOTAL trial budget for the point, split across its
    ``n_workers`` (~1/n_workers each) — the same meaning whether or not the run
    is ``preemptive``. ``workers_per_gpu`` packs that many workers onto one GPU
    (one sbatch per pack) for the ``optuna_packed_node`` strategy; ``1`` (default)
    keeps one GPU per worker.
    """

    strategy: str = "optuna_single_node"
    sweep: str | None = None
    n_trials: int = 40
    n_jobs: int = 1
    n_workers: int = 1
    workers_per_gpu: int = 1
    resources: ResourceSpec | None = None
    extra_overrides: tuple[str, ...] = ()
    preemptive: bool = False
    preempt_grace_seconds: int = 180


@dataclass(frozen=True)
class LaunchContext:
    study_prefix: str
    storage_dir: str
    sweeps_root: str
    seed: int | None = None
    # When set, ALL cells use this exact Optuna storage URL (e.g. a shared
    # Postgres DB) with a per-cell ``study_name``; when None, each cell gets its
    # own ``sqlite:///{storage_dir}/{prefix}_{job}.db``.
    storage_url: str | None = None


def _job_name(point: StudyPoint, ctx: LaunchContext) -> str:
    return point.name if ctx.seed is None else f"{point.name}__seed{ctx.seed}"


def _storage_for(ctx: LaunchContext, job: str) -> str:
    """The Optuna storage URL for ``job`` — the shared ``ctx.storage_url`` if set,
    else a per-cell ``sqlite:///{storage_dir}/{prefix}_{job}.db``. The strategy
    overrides and ``_make_preempt_spec`` MUST agree, so both go through here.
    """
    if ctx.storage_url:
        return ctx.storage_url
    return f"sqlite:///{os.path.join(ctx.storage_dir, f'{ctx.study_prefix}_{job}.db')}"


def _preempt_env(pl: PointLaunch, *, worker_idx: int) -> dict[str, str] | None:
    """Env dict for a preemptive local subprocess (else ``None`` → inherit parent env).

    When ``pl.preemptive`` is set, the trainer's signal handler (and ``ddssm.app``'s
    trial lookup) require ``DDSSM_PREEMPTIVE=1`` to activate; ``DDSSM_WORKER_ID``
    identifies the worker for the orchestrator-side bookkeeping. We merge with
    the parent environment so PATH / LD_LIBRARY_PATH / venv hooks survive.
    """
    if not pl.preemptive:
        return None
    return {**os.environ, "DDSSM_PREEMPTIVE": "1", "DDSSM_WORKER_ID": str(worker_idx)}


def _validate_preempt_compat(pl: PointLaunch, point: StudyPoint) -> None:
    """Raise ``ValueError`` if ``pl.preemptive=True`` is paired with an incompatible config.

    Only one rejection remains: ``strategy='single_job'`` has no trials concept,
    so preempt+resume cannot work. Multi-stage experiments ARE supported now
    that ``StageOrchestrator`` is stage-prefix-aware on resume (ADR-0009).
    """
    if not pl.preemptive:
        return
    if pl.strategy == "single_job":
        raise ValueError(
            "strategy 'single_job' cannot be preemptive "
            "(no trials concept; preemptive=True requires an Optuna sweep)"
        )


# ---------------------------------------------------------------------------
# Launch strategies — the sbatch shape per point
# ---------------------------------------------------------------------------


class LaunchStrategy(abc.ABC):
    """Produces the Hydra overrides for one point's worker (the sweep shape).

    ``supports_sbatch`` / ``supports_local`` gate which orchestrator backend is
    allowed for this strategy. ``n_workers_per_point`` reports how many parallel
    workers a multi-worker strategy emits per point (the orchestrator iterates
    ``range(n_workers_per_point(pl))`` and passes ``worker_idx`` to
    :meth:`hydra_overrides`). ``workers_per_job`` reports how many of those
    workers share ONE sbatch (and thus one GPU): the default ``1`` keeps the
    historical one-sbatch-per-worker shape; a packed strategy returns >1 to
    co-locate that many workers on a single GPU.
    """

    name: str
    supports_sbatch: bool = True
    supports_local: bool = True

    def n_workers_per_point(self, pl: PointLaunch) -> int:
        return 1

    def workers_per_job(self, pl: PointLaunch) -> int:
        return 1

    @abc.abstractmethod
    def hydra_overrides(
        self,
        point: StudyPoint,
        pl: PointLaunch,
        ctx: LaunchContext,
        *,
        worker_idx: int = 0,
    ) -> list[str]:
        ...


class SingleJob(LaunchStrategy):
    """One job, one trial — no Optuna multirun."""

    name = "single_job"

    def hydra_overrides(self, point, pl, ctx, *, worker_idx=0):
        return []


class OptunaSingleNode(LaunchStrategy):
    """One Optuna multirun on a single node (its own SQLite study)."""

    name = "optuna_single_node"

    def hydra_overrides(self, point, pl, ctx, *, worker_idx=0):
        if not pl.sweep:
            raise ValueError(f"strategy {self.name!r} needs PointLaunch.sweep set")
        job = _job_name(point, ctx)
        sweep_dir = os.path.join(ctx.sweeps_root, f"{ctx.study_prefix}_{job}")
        storage = _storage_for(ctx, job)
        # Under preemptive runs, n_trials is computed at sbatch-start time by
        # ddssm.launch_remaining and bound to $N_PER_WORKER; the orchestrator
        # substitutes the placeholder per-backend (shell var for sbatch,
        # literal ceil(n_trials/n_workers) for local).
        n_trials_value = "__N_PER_WORKER__" if pl.preemptive else str(pl.n_trials)
        overrides = [
            "--multirun",
            f"+sweep={pl.sweep}",
            f"hydra.sweeper.n_trials={n_trials_value}",
            f"hydra.sweeper.study_name={ctx.study_prefix}_{job}",
            f"hydra.sweeper.storage={storage}",
            f"hydra.sweep.dir={sweep_dir}",
        ]
        if pl.n_jobs > 1:
            overrides.append(f"hydra.sweeper.n_jobs={pl.n_jobs}")
        return overrides


class _MultiWorkerOptunaBase(LaunchStrategy):
    """Shared shape for ``optuna_multi_node`` + ``local_parallel``.

    Each worker runs an independent ``ddssm.app --multirun`` against the SAME
    ``study_name`` + ``storage`` + ``hydra.sweep.dir`` parent. Optuna's per-trial
    locking handles concurrent trial selection; ``hydra.sweep.subdir`` is the
    per-worker namespace that prevents trial-dir collisions.

    The DB path is just ``storage_dir/<study>_<point>.db``; the caller is
    responsible for pointing ``storage_dir`` at a *shared* filesystem (NFS)
    when using the multi-node backend, and at a local filesystem when using the
    local-parallel backend. The plan-campaign skill table caps NFS-SQLite at
    ~8 workers per DB; beyond that, shard the points or switch to Postgres.
    """

    def n_workers_per_point(self, pl):
        return max(1, pl.n_workers)

    def hydra_overrides(self, point, pl, ctx, *, worker_idx=0):
        if not pl.sweep:
            raise ValueError(f"strategy {self.name!r} needs PointLaunch.sweep set")
        job = _job_name(point, ctx)
        sweep_dir = os.path.join(ctx.sweeps_root, f"{ctx.study_prefix}_{job}")
        storage = _storage_for(ctx, job)
        # ``hydra.job.num`` resets per multirun invocation (per upstream
        # ``OptunaSweeperImpl.setup``), so under preemptive runs we add a
        # ``$DDSSM_INVOC`` stamp (set in the sbatch preamble) to keep retry
        # trial sub-dirs from colliding across requeues.
        #
        # ``pl.n_trials`` is the TOTAL trial budget for the cell, divided across
        # its ``n_workers``. Preemptive runs split it dynamically from the
        # remaining budget (``$N_PER_WORKER``, computed in the sbatch preamble);
        # non-preemptive runs split it statically here. Either way each worker
        # gets a ~1/n_workers share, so ``n_trials`` means the same thing
        # regardless of ``preemptive`` (it is NOT per-worker).
        n_trials_value = (
            "__N_PER_WORKER__" if pl.preemptive
            else str(math.ceil(pl.n_trials / max(1, pl.n_workers)))
        )
        subdir = (
            f"hydra.sweep.subdir=w{worker_idx}_${{oc.env:DDSSM_INVOC}}_${{hydra.job.num}}"
            if pl.preemptive
            else f"hydra.sweep.subdir=w{worker_idx}_${{hydra.job.num}}"
        )
        return [
            "--multirun",
            f"+sweep={pl.sweep}",
            f"hydra.sweeper.n_trials={n_trials_value}",
            f"hydra.sweeper.study_name={ctx.study_prefix}_{job}",
            f"hydra.sweeper.storage={storage}",
            f"hydra.sweep.dir={sweep_dir}",
            subdir,
            _SAMPLER_SEED_OVERRIDE,
        ]


class OptunaMultiNode(_MultiWorkerOptunaBase):
    """Optuna multirun across N SLURM jobs sharing an NFS-hosted SQLite DB.

    Render emits ``pl.n_workers`` independent sbatch scripts per point; each
    is one ``ddssm.app --multirun`` worker pulling trials from the shared DB.
    ``storage_dir`` must point at a shared filesystem before submission.

    Local execution is rejected — use :class:`LocalParallel` for the on-machine
    multi-worker shape (the two strategies share the override layout, but differ
    in which orchestrator backend may run them).
    """

    name = "optuna_multi_node"
    supports_local = False


class OptunaPackedNode(_MultiWorkerOptunaBase):
    """Optuna multirun packing ``workers_per_gpu`` workers onto ONE GPU per sbatch.

    Like :class:`OptunaMultiNode` (workers share one DB via the multi-worker
    override layout — per-worker ``hydra.sweep.subdir``, shared ``study_name`` /
    ``storage`` / ``sweep.dir``), but instead of one single-GPU sbatch *per
    worker*, the orchestrator emits one sbatch per group of ``pl.workers_per_gpu``
    workers that all run on the job's single GPU. Each worker is CPU-thread-pinned
    to ``resources.cpus // workers_per_gpu`` so K procs do not oversubscribe the
    allocation. Use when a trial's GPU-memory footprint is small (<<GPU) so many
    trials fit on one card and GPU parallelism beats one-GPU-per-trial waste.

    Local execution is rejected — the orchestrator renders this for sbatch only.
    """

    name = "optuna_packed_node"
    supports_local = False

    def workers_per_job(self, pl):
        return max(1, pl.workers_per_gpu)


class LocalParallel(_MultiWorkerOptunaBase):
    """Optuna multirun across ``n_workers`` local subprocesses sharing a SQLite DB.

    The orchestrator's ``run_local`` spawns ``pl.n_workers`` parallel ``Popen``s
    per point, each running ``ddssm.app --multirun`` against the same local DB.
    Points run sequentially (one at a time); workers within a point run in
    parallel.

    sbatch rendering is rejected — use :class:`OptunaMultiNode` for the cluster
    shape. If you need per-worker GPU binding, set ``CUDA_VISIBLE_DEVICES``
    yourself before invoking ``python -m ddssm.launch ... --local`` (the
    orchestrator does not inject device assignments).
    """

    name = "local_parallel"
    supports_sbatch = False


class _Stub(LaunchStrategy):
    def hydra_overrides(self, point, pl, ctx, *, worker_idx=0):
        raise NotImplementedError(
            f"the {self.name!r} launch strategy is a documented extension point "
            f"(ADR-0008) and is not implemented yet"
        )


class SlurmArray(_Stub):
    """One SLURM array task per point or per trial. (stub)"""

    name = "slurm_array"


_STRATEGIES: dict[str, LaunchStrategy] = {
    s.name: s
    for s in (
        SingleJob(),
        OptunaSingleNode(),
        OptunaMultiNode(),
        OptunaPackedNode(),
        LocalParallel(),
        SlurmArray(),
    )
}


# ---------------------------------------------------------------------------
# Study registry (so the CLI can resolve a study by name)
# ---------------------------------------------------------------------------


STUDY_REGISTRY: dict[str, Study] = {}


def register_study(
    study: Study, into: Callable[..., Any] | None = None
) -> Study:
    """Register a study so ``python -m ddssm.launch <name>`` can find it.

    Pass ``into=<store>`` (e.g. ``ddssm.stores.experiment_store``) to also
    publish every point's config so ``experiment=<point_name>`` resolves — one
    call keeps the launcher registry and the experiment store in sync instead
    of requiring a separate ``study.register(store)``.
    """
    STUDY_REGISTRY[study.name] = study
    if into is not None:
        study.register(into)
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
    # Optional shared Optuna storage URL (e.g. ``postgresql://host/db``). When
    # set, every cell lands in this one DB as a distinct ``study_name``.
    storage_url: str | None = None

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
        """Return ``[(job_name, sbatch_text), ...]`` for the given points.

        Multi-worker strategies (e.g. ``optuna_multi_node``) emit one sbatch per
        worker per point; the job name carries a ``_w<idx>`` suffix and each
        worker's overrides include a distinct ``hydra.sweep.subdir`` while
        ``study_name`` / ``storage`` / ``sweep.dir`` parent are shared.

        Under ``pl.preemptive=True`` the orchestrator builds a
        :class:`PreemptSpec` per worker — its ``storage_url`` + ``study_name``
        match the strategy's emitted ``hydra.sweeper.storage`` /
        ``hydra.sweeper.study_name`` so the preamble's ``launch_remaining``
        invocation hits the same SQLite DB as the workers — and passes it
        into ``render_sbatch``.
        """
        ctx = LaunchContext(
            self.study_prefix, self.storage_dir, self.sweeps_root,
            seed=seed, storage_url=self.storage_url,
        )
        out: list[tuple[str, str]] = []
        for point in points:
            pl = self._point_launch(point, launch_override)
            strategy = _STRATEGIES[pl.strategy]
            if not strategy.supports_sbatch:
                raise ValueError(
                    f"strategy {pl.strategy!r} does not support sbatch rendering; "
                    f"use --local (and a local-capable strategy)"
                )
            _validate_preempt_compat(pl, point)
            n_w = strategy.n_workers_per_point(pl)
            per_job = max(1, strategy.workers_per_job(pl))
            base_name = _job_name(point, ctx)

            def _worker_overrides(w_idx: int) -> list[str]:
                ov = strategy.hydra_overrides(point, pl, ctx, worker_idx=w_idx)
                ov += self._variant_overrides(point, size)
                ov += list(pl.extra_overrides)
                if seed is not None:
                    ov.append(f"experiment.seed={seed}")
                return ov

            # Group workers into sbatch jobs: one-per-job for the historical
            # shapes (per_job == 1), or ``workers_per_gpu``-sized packs.
            groups = [
                list(range(i, min(i + per_job, n_w)))
                for i in range(0, n_w, per_job)
            ]
            for g_idx, w_idxs in enumerate(groups):
                # ``experiment=<point.name>`` is the registered preset; the
                # job_name + output_pattern carry the worker/group/seed suffix so
                # each sbatch is distinguishable in squeue and logs separately.
                if per_job == 1:
                    # One sbatch per worker: bare name for a lone worker, else _w<i>.
                    job = base_name if n_w == 1 else f"{base_name}_w{w_idxs[0]}"
                else:
                    # Packed: bare name when it's the only group, else _g<i>.
                    job = base_name if len(groups) == 1 else f"{base_name}_g{g_idx}"
                # PreemptSpec divides the cell-wide budget by the TOTAL worker
                # count (n_w), not the per-job group size.
                preempt = (
                    self._make_preempt_spec(pl, ctx, point, n_w, w_idxs[0])
                    if pl.preemptive else None
                )
                if per_job == 1:
                    script = render_sbatch(
                        point.name,
                        exp_sbatch=pl.resources,
                        hydra_overrides=_worker_overrides(w_idxs[0]),
                        cli_overrides={"job_name": f"ddssm-{job}"},
                        output_pattern=f"runs/{job}/slurm-%j.out",
                        preempt=preempt,
                    )
                else:
                    script = render_packed_sbatch(
                        point.name,
                        exp_sbatch=pl.resources,
                        worker_overrides=[(w, _worker_overrides(w)) for w in w_idxs],
                        cli_overrides={"job_name": f"ddssm-{job}"},
                        output_pattern=f"runs/{job}/slurm-%j.out",
                        preempt=preempt,
                    )
                out.append((job, script))
        return out

    def _make_preempt_spec(
        self,
        pl: PointLaunch,
        ctx: LaunchContext,
        point: StudyPoint,
        n_workers: int,
        worker_idx: int,
    ) -> PreemptSpec:
        """Build the per-worker :class:`PreemptSpec`.

        ``storage_url`` + ``study_name`` MUST match what the strategy emits in
        its ``hydra_overrides`` (see ``OptunaSingleNode.hydra_overrides`` for
        the canonical paths) — otherwise the preamble's ``launch_remaining``
        invocation talks to a different DB than the worker.
        """
        job = _job_name(point, ctx)
        return PreemptSpec(
            grace_seconds=pl.preempt_grace_seconds,
            storage_url=_storage_for(ctx, job),
            study_name=f"{ctx.study_prefix}_{job}",
            target=pl.n_trials,
            n_workers=n_workers,
            worker_idx=worker_idx,
        )

    def render_colocated(
        self,
        points,
        *,
        n_gpus: int,
        workers_per_cell_per_gpu: int,
        target: int,
        sweep: str,
        resources: SBatch,
        preemptive: bool = True,
        grace_seconds: int = 180,
        size: str | None = None,
        seed: int | None = None,
    ) -> list[tuple[str, str]]:
        """Render ONE sbatch per GPU, each co-locating ALL ``points`` on its card.

        Where :meth:`render` emits one job per cell (a cell's workers own a GPU),
        this packs EVERY cell onto each of ``n_gpus`` GPUs with
        ``workers_per_cell_per_gpu`` workers apiece — so a cell's total
        concurrency is ``workers_per_cell_per_gpu × n_gpus`` against one shared
        study, while each GPU hosts ``len(points) × workers_per_cell_per_gpu``
        processes. Used to ADD trials to an existing shared-DB study at low
        per-cell concurrency (let NSGA-II evolve) without burning a GPU per cell.

        ``target`` is the per-cell TOTAL trial budget; under ``preemptive`` the
        existing study's COMPLETE trials count toward it (each cell's preamble
        runs its own ``launch_remaining``), so re-running converges to ``target``
        rather than over-running it. The reused :class:`OptunaPackedNode` override
        layout keeps the per-worker ``hydra.sweep.subdir`` + NSGA-II sampler seed
        distinct; worker indices are cell-global (GPU ``g`` owns ``[g·k, g·k+k)``)
        so the same cell's subdirs never collide across its GPU jobs.
        """
        ctx = LaunchContext(
            self.study_prefix, self.storage_dir, self.sweeps_root,
            seed=seed, storage_url=self.storage_url,
        )
        pts = list(points)
        k = max(1, workers_per_cell_per_gpu)
        total_per_cell = n_gpus * k
        strategy = _STRATEGIES["optuna_packed_node"]
        out: list[tuple[str, str]] = []
        for g in range(n_gpus):
            cell_workers: list[CellWorker] = []
            preempts: dict[str, PreemptSpec] = {}
            for point in pts:
                job = _job_name(point, ctx)
                pl = PointLaunch(
                    strategy="optuna_packed_node",
                    sweep=sweep,
                    n_trials=target,
                    n_workers=total_per_cell,
                    workers_per_gpu=k,
                    preemptive=preemptive,
                    preempt_grace_seconds=grace_seconds,
                    resources=resources,
                )
                _validate_preempt_compat(pl, point)
                for j in range(k):
                    w_idx = g * k + j  # cell-global; distinct across GPUs
                    ov = strategy.hydra_overrides(point, pl, ctx, worker_idx=w_idx)
                    ov += self._variant_overrides(point, size)
                    ov += list(pl.extra_overrides)
                    if seed is not None:
                        ov.append(f"experiment.seed={seed}")
                    cell_workers.append(
                        CellWorker(
                            experiment=point.name, cell_key=job,
                            worker_idx=w_idx, overrides=ov,
                        )
                    )
                if preemptive:
                    preempts[job] = PreemptSpec(
                        grace_seconds=grace_seconds,
                        storage_url=_storage_for(ctx, job),
                        study_name=f"{self.study_prefix}_{job}",
                        target=target,
                        n_workers=total_per_cell,
                    )
            gpu_job = f"{self.study_prefix}_colo_g{g}"
            spec = dataclasses.replace(resources, job_name=f"ddssm-{gpu_job}")
            script = render_multicell_packed_sbatch(
                cell_workers,
                spec=spec,
                output_pattern=f"runs/{gpu_job}/slurm-%j.out",
                preempts=preempts if preemptive else None,
            )
            out.append((gpu_job, script))
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
        if submit:
            # Deterministically kill the CREATE TABLE race: with multi-GPU cells
            # a cell fans out into several sbatch jobs that each run a schema-init
            # step on the shared NFS SQLite DB. Touch each DB once here (before any
            # job starts) so the schema exists and no two jobs race on DDL.
            self._precreate_storage(points, seeds)
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

    def _precreate_storage(self, points, seeds) -> None:
        """Create each cell's Optuna schema ONCE before submitting any jobs.

        Constructing an ``RDBStorage`` runs Optuna's ``_init_tables`` (a no-op if
        the tables already exist), so after this the ``studies``/``trials`` tables
        exist and the per-job init steps + workers only contend on the study-row
        INSERT — which ``create_study(load_if_exists=True)`` handles. No
        directions needed; the sweeper still creates the study itself. The DB
        path mirrors what the strategies emit (``_make_preempt_spec``).
        """
        from optuna.storages import RDBStorage

        # Shared storage (e.g. Postgres): one DB for all cells, so init the schema
        # ONCE. (Cells are distinct study_names the sweeper creates on demand.)
        if self.storage_url is not None:
            RDBStorage(self.storage_url)
            print("# Pre-created Optuna schema in shared storage.", file=sys.stderr)
            return

        seen: set[str] = set()
        for seed in seeds:
            ctx = LaunchContext(
                self.study_prefix, self.storage_dir, self.sweeps_root, seed=seed
            )
            for point in points:
                db = os.path.join(
                    self.storage_dir, f"{self.study_prefix}_{_job_name(point, ctx)}.db"
                )
                if db in seen:
                    continue
                seen.add(db)
                RDBStorage(f"sqlite:///{db}")
        if seen:
            print(f"# Pre-created Optuna schema for {len(seen)} cell DB(s).", file=sys.stderr)

    def run_local(self, points, *, size=None, seeds=(None,), out_dir="runs/local",
                  launch_override=None) -> int:
        """Run each (point × seed) locally.

        Single-worker strategies (``single_job``, ``optuna_single_node``) run as
        one subprocess per point (no multirun — preserves the existing smoke
        flow). The ``local_parallel`` strategy spawns ``pl.n_workers`` parallel
        subprocesses per point, all sharing a local SQLite DB. Points run
        sequentially regardless. Strategies that don't support local execution
        (e.g. ``optuna_multi_node``) raise immediately.
        """
        failures: list[tuple[str, int]] = []
        for seed in seeds:
            ctx = LaunchContext(
            self.study_prefix, self.storage_dir, self.sweeps_root,
            seed=seed, storage_url=self.storage_url,
        )
            for point in points:
                pl = self._point_launch(point, launch_override)
                strategy = _STRATEGIES[pl.strategy]
                if not strategy.supports_local:
                    raise ValueError(
                        f"strategy {pl.strategy!r} does not support --local; "
                        f"use --write-dir, or switch the point's strategy to "
                        f"local_parallel for on-machine multi-worker"
                    )
                _validate_preempt_compat(pl, point)
                n_w = strategy.n_workers_per_point(pl)
                base_name = _job_name(point, ctx)

                if n_w == 1:
                    run_dir = os.path.join(out_dir, f"{self.study_prefix}_{base_name}")
                    os.makedirs(run_dir, exist_ok=True)
                    cmd = [
                        sys.executable, "-m", "ddssm.app",
                        f"experiment={point.name}",
                        f"hydra.run.dir={run_dir}",
                        *self._variant_overrides(point, size),
                    ]
                    if seed is not None:
                        cmd.append(f"experiment.seed={seed}")
                    # Single-worker preemptive on the local backend is a smoke
                    # convenience: the trainer's signal handler is still active
                    # via DDSSM_PREEMPTIVE=1, but there's no multirun so no
                    # n_trials substitution is needed.
                    print(f"[local] {base_name} ...", flush=True)
                    rc = subprocess.run(
                        cmd, check=False, env=_preempt_env(pl, worker_idx=0),
                    ).returncode
                    if rc != 0:
                        failures.append((base_name, rc))
                else:
                    # Multi-worker preempt: substitute __N_PER_WORKER__ with the
                    # literal ceil(n_trials / n_workers) (the sbatch path uses
                    # a shell var; --local has no shell layer between us and
                    # ddssm.app).
                    n_per_worker_literal = (
                        str(math.ceil(pl.n_trials / n_w)) if pl.preemptive else None
                    )
                    procs: list[tuple[str, subprocess.Popen]] = []
                    for w_idx in range(n_w):
                        overrides = strategy.hydra_overrides(point, pl, ctx, worker_idx=w_idx)
                        overrides += self._variant_overrides(point, size)
                        overrides += list(pl.extra_overrides)
                        if seed is not None:
                            overrides.append(f"experiment.seed={seed}")
                        # Distinct sampler seed per local worker (no SLURM here);
                        # idx+1 keeps it nonzero and unique across the pack.
                        overrides = [
                            o.replace(_SAMPLER_SEED_PLACEHOLDER, str(w_idx + 1))
                            for o in overrides
                        ]
                        if n_per_worker_literal is not None:
                            overrides = [
                                o.replace(_N_PER_WORKER_PLACEHOLDER, n_per_worker_literal)
                                for o in overrides
                            ]
                        cmd = [
                            sys.executable, "-m", "ddssm.app",
                            f"experiment={point.name}",
                            *overrides,
                        ]
                        job = f"{base_name}_w{w_idx}"
                        print(f"[local] {job} ...", flush=True)
                        procs.append(
                            (job, subprocess.Popen(cmd, env=_preempt_env(pl, worker_idx=w_idx)))
                        )
                    for job, proc in procs:
                        rc = proc.wait()
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
    p.add_argument("--storage-url", default=None,
                   help="shared Optuna storage URL (e.g. postgresql://host/db). "
                        "When set, ALL cells land in this one DB as distinct "
                        "studies; --storage-dir is then ignored for the DB path.")
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
        storage_url=args.storage_url,
    )

    if args.local:
        # Local backend runs single jobs (no Optuna multirun) — for smoke/debug.
        return orch.run_local(points, size=args.size, seeds=seeds, out_dir=args.out_dir)
    return orch.launch(points, size=args.size, seeds=seeds,
                       write_dir=args.write_dir, submit=args.submit)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "STUDY_REGISTRY",
    "LaunchStrategy",
    "LocalParallel",
    "OptunaMultiNode",
    "OptunaPackedNode",
    "OptunaSingleNode",
    "PointLaunch",
    "ResourceSpec",
    "SingleJob",
    "StudyOrchestrator",
    "register_study",
]
