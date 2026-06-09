"""The init-centering ablation, declared as a library :class:`ddssm.cluster.study.Study`.

Two comparison axes — `cell` (the 12-point grid) × `dataset` (1d/mv) — cross into
24 registered presets named ``init_<cell>__<dataset>``. The `cell` axis exposes
its sub-fields (baseline_form/mode/tracking) as tags for filtering + reporting.
Each point's launch intent is a single-node Optuna sweep; size variants (paper,
smoke) are override hooks the orchestrator applies.
"""

from __future__ import annotations

from typing import Any, Mapping

from ddssm.launch import PointLaunch, ResourceSpec, register_study
from experiments._make import experiment
from ddssm.cluster.study import Axis, Study, StudyPoint
from ddssm.experiment.stores import experiment_store
from experiments.init_centering.cells import Cell, iter_cells
from experiments.init_centering.evals import PilotEval, PilotMOObjective
from experiments.init_centering.model import SmokeModel
from experiments.init_centering.hparams import StagesB, Training800, SmokeHparams
from experiments.init_centering.datasets import ABLATION_DATASETS, paper_latent


def _build(coords: Mapping[str, Any]):
    """One (cell × dataset) experiment, baking the real dataset + dims (tiny size)."""
    cell: Cell = coords["cell"]
    ds = coords["dataset"]
    return experiment(
        data=ds.data_preset,
        model=SmokeModel(
            baseline_form=cell.baseline_form,
            baseline_mode=cell.baseline_mode,
            tracking_mode=cell.tracking_mode,
            data_dim=ds.data_dim,
            latent_dim=ds.latent_dim,
        ),
        hparams=SmokeHparams,
        training=Training800,
        # One source of baseline_mode -> model + stage mask/anchor.
        # n_stage2 set EXPLICITLY (not the hparams default of 1000): the old
        # shell launcher passed n_stage2=5000 on the CLI; the Study refactor
        # dropped that override and silently fell back to 1000, leaving trials
        # badly under-trained (still descending at the budget). Pin it here.
        stages=StagesB(baseline_mode=cell.baseline_mode, n_stage2=5000),
        eval=PilotEval,
        objective=PilotMOObjective,
    )


# Compute-node environment bring-up (Tempest). Emitted after ``cd
# "$SLURM_SUBMIT_DIR"`` and before any ``python`` call so the interpreter +
# CUDA resolve on the allocated node. ``source .venv/bin/activate`` is relative
# to the submit dir (the repo root), so it is user/path agnostic.
_TEMPEST_SETUP = (
    # epyc/gpu sbatch shells do not pre-define `module`; source /etc/profile to
    # get the Lmod function. /etc/profile + Lmod reference unset vars and return
    # nonzero, so relax `set -eu` around the whole block (pipefail stays on).
    # Without this, the merged `set -euo pipefail` aborts the job at `module
    # purge` with "module: command not found".
    "set +eu",
    "source /etc/profile",
    "module purge",
    "module load Python/3.13.5-GCCcore-14.3.0 CUDA/13.0.0 tools/uv/0.9.22",
    "source .venv/bin/activate",
    "set -eu",
)

# Slurm accounts (Tempest). gpupriority only admits the ``priority-*`` accounts
# in its AllowAccounts list — a job submitted under any other account sits
# PENDING forever with reason ``(PartitionConfig)``. gpuunsafe takes the
# matching ``group-*`` account. Both gate on the michaelwojnowicz project here.
_PRIORITY_ACCOUNT = "--account=priority-michaelwojnowicz"
_UNSAFE_ACCOUNT = "--account=group-michaelwojnowicz"


# Non-headline cells routed to the b6000 (Blackwell, 96 GB) GPUs. The two
# ``pinned`` cells (persistence/zero) were moved here off A40: the A40 nodes have
# only 32 CPU for BOTH GPUs (16/GPU), so they can't be un-starved, whereas b6000
# is ~4x faster AND well-fed at 2 CPU/worker. Each cell takes 2 GPUs, so these 5
# cells (10 GPUs) oversubscribe the 6 b6000s and time-share them via the queue —
# still well under the A100 headline cell's wall time. Only ``mlp_pinned`` stays
# on the guaranteed A100 (gpupriority, now 4 CPU/worker).
_B6000_CELLS = frozenset(
    {
        "init_mlp_learnable_per_t",
        "init_linear_learnable_per_t",
        "init_linear_pinned_per_t",
        "init_persistence_pinned_per_t",
        "init_zero_pinned_per_t",
    }
)


def _launch(point: StudyPoint) -> PointLaunch:
    """Round-1 (trans-KL-fix rerun) launch intent: GPU-packed Optuna MOO.

    Every cell uses the WIDE ``init_ablation_moo`` search space, **2 CPUs/worker**
    (16 workers/GPU on the 32-CPU per-GPU half-node share), 96-trial budget.

    - Headline ``mlp_pinned_per_t`` → 1 guaranteed **A100** on ``gpupriority``
      (non-preempt), 16-pack (6 trials/worker).
    - 3 cells → 2 × **b6000** each (uses all 6 idle Blackwells), 32 workers
      (3/worker), ``gpuunsafe``.
    - 2 cells → 2 × **A40** each, 32 workers, ``gpuunsafe``.

    The packed strategy splits a cell's budget across SAME-gres GPUs, so each
    cell uses 2 of one type. Preemption is safe now: the reaper is
    liveness-based (``fail_stale_trials``) and the preamble no longer age-reaps,
    so requeues/joins never fail live trials.
    """
    cell = point.coords["cell"]

    # Headline cell -> the guaranteed A100 (gpupriority, non-preempt). 64/16 = 4.
    if cell.name == "init_mlp_pinned_per_t":
        return PointLaunch(
            strategy="optuna_packed_node",
            sweep="init_ablation_moo_r2",
            n_trials=64,
            n_workers=16,
            workers_per_gpu=16,
            preemptive=False,
            resources=ResourceSpec(
                partition="gpupriority",
                gpus=1,
                cpus=64,  # 16 workers x 4 CPUs
                mem="80G",
                time="48:00:00",
                extra_flags=("--gres=gpu:a100:1", _PRIORITY_ACCOUNT),
                setup=_TEMPEST_SETUP,
            ),
        )

    # Other cells -> 2 GPUs each (b6000 or A40), gpuunsafe/preemptible. 64/32 = 2.
    gres = "--gres=gpu:b6000:1" if cell.name in _B6000_CELLS else "--gres=gpu:a40:1"
    return PointLaunch(
        strategy="optuna_packed_node",
        sweep="init_ablation_moo_r2",
        n_trials=64,
        n_workers=32,
        workers_per_gpu=16,
        preemptive=True,
        resources=ResourceSpec(
            partition="gpuunsafe",
            gpus=1,
            cpus=32,  # 16 workers x 2 CPUs = 32 (the per-GPU half-node share)
            mem="80G",
            time="48:00:00",
            extra_flags=(gres, _UNSAFE_ACCOUNT),
            setup=_TEMPEST_SETUP,
        ),
    )


def _paper_overrides(point: StudyPoint) -> list[str]:
    return [f"experiment.model.latent_dim={paper_latent(point.coords['dataset'])}"]


def _smoke_overrides(point: StudyPoint) -> list[str]:
    return [
        "experiment.training.stages.n_pretrain=5",
        "experiment.training.stages.n_stage2=5",
        "experiment.training.stages.log_every=1",
        "experiment.training.stages.checkpoint_every=100",
    ]


INIT_CENTERING_STUDY = register_study(
    Study.from_axes(
        "init_centering",
        axes=[
            Axis(
                "cell",
                list(iter_cells()),
                key=lambda c: c.name,
                tags=lambda c: {
                    "baseline_form": c.baseline_form,
                    "baseline_mode": c.baseline_mode,
                    "tracking_mode": c.tracking_mode,
                },
            ),
            Axis("dataset", ABLATION_DATASETS, key=lambda d: d.label),
        ],
        build=_build,
        name_point=lambda tags: f"{tags['cell']}__{tags['dataset']}",
        launch=_launch,
        variants={
            "tiny": lambda p: [],
            "paper": _paper_overrides,
            "smoke": _smoke_overrides,
        },
    ),
    # Publish the cell points to the experiment store in the same call, so the
    # launcher registry and ``experiment=<cell>`` resolution can't desync.
    into=experiment_store,
)

__all__ = ["INIT_CENTERING_STUDY"]
