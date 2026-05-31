"""The init-centering ablation, declared as a library :class:`ddssm.study.Study`.

Two comparison axes — `cell` (the 12-point grid) × `dataset` (1d/mv) — cross into
24 registered presets named ``init_<cell>__<dataset>``. The `cell` axis exposes
its sub-fields (baseline_form/mode/tracking) as tags for filtering + reporting.
Each point's launch intent is a single-node Optuna sweep; size variants (paper,
smoke) are override hooks the orchestrator applies.
"""

from __future__ import annotations

from typing import Any, Mapping

from ddssm.study import Axis, Study, StudyPoint
from ddssm.launch import PointLaunch, ResourceSpec, register_study
from ddssm.stores import experiment_store
from experiments._make import experiment
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
    "module purge",
    "module load Python/3.13.5-GCCcore-14.3.0 CUDA/13.0.0 tools/uv/0.9.22",
    "source .venv/bin/activate",
)

# Slurm accounts (Tempest). gpupriority only admits the ``priority-*`` accounts
# in its AllowAccounts list — a job submitted under any other account sits
# PENDING forever with reason ``(PartitionConfig)``. gpuunsafe takes the
# matching ``group-*`` account. Both gate on the michaelwojnowicz project here.
_PRIORITY_ACCOUNT = "--account=priority-michaelwojnowicz"
_UNSAFE_ACCOUNT = "--account=group-michaelwojnowicz"


def _launch(point: StudyPoint) -> PointLaunch:
    """Round-1 (fresh restart) launch intent: GPU-packed Optuna MOO.

    Every cell uses the WIDE ``init_ablation_moo`` search space (the full
    round-1 priors). The narrowed ``init_ablation_moo_r2`` space was derived
    from round-1 data that a since-found code bug invalidated, so we restart
    from scratch on the wide ranges. Each trial is <4 GiB, so we pack 8 workers
    per GPU (4 CPUs each, thread-pinned — the round-1 starvation fix).
    ``n_trials`` (96) is the cell-total budget split across ALL the cell's
    workers (~``n_trials / n_workers`` each), independent of ``preemptive``.

    Launched per_t-only (``--select tracking_mode=per_t`` → 6 cells), so each
    cell can claim 2 GPUs (``n_workers=16`` = 2 × 8-pack, sharing the cell's
    Optuna DB) and finish 96×5k-step trials in ~13h instead of ~26h. GPU pool =
    1 A100 (16-pack) for the headline ``mlp_pinned_per_t`` + 2 A40 on
    ``gpupriority`` (non-preempt) for ``mlp_learnable_per_t`` + 2 A40 each on
    ``gpuunsafe`` (preemptible → checkpoint/resume, ADR-0009) for the other 4
    cells ≈ 11 GPUs. 16 workers/cell is the proven NFS-SQLite worker level.
    """
    cell = point.coords["cell"]
    sweep = "init_ablation_moo"  # WIDE round-1 priors for every cell

    # --- gpupriority (non-preempt): two dedicated cells, the 2 priority GPUs ---
    # Headline cell -> the A100 (2x memory -> 2x the pack). 96 / 16 = 6 trials/worker.
    if cell.name == "init_mlp_pinned_per_t":
        return PointLaunch(
            strategy="optuna_packed_node",
            sweep=sweep,
            n_trials=96,
            n_workers=16,
            workers_per_gpu=16,
            preemptive=False,
            resources=ResourceSpec(
                partition="gpupriority",
                gpus=1,
                cpus=64,
                mem="96G",
                time="48:00:00",
                extra_flags=("--gres=gpu:a100:1", _PRIORITY_ACCOUNT),
                setup=_TEMPEST_SETUP,
            ),
        )
    # Second non-preempt cell -> 2 gpupriority A40s (16 workers = 2 x 8-pack).
    if cell.name == "init_mlp_learnable_per_t":
        return PointLaunch(
            strategy="optuna_packed_node",
            sweep=sweep,
            n_trials=96,
            n_workers=16,
            workers_per_gpu=8,
            preemptive=False,
            resources=ResourceSpec(
                partition="gpupriority",
                gpus=1,
                cpus=32,
                mem="48G",
                time="48:00:00",
                extra_flags=("--gres=gpu:a40:1", _PRIORITY_ACCOUNT),
                setup=_TEMPEST_SETUP,
            ),
        )

    # --- gpuunsafe (preemptible): every other cell, 2 A40s (16 workers = 2 x 8-pack) ---
    return PointLaunch(
        strategy="optuna_packed_node",
        sweep=sweep,
        n_trials=96,
        n_workers=16,
        workers_per_gpu=8,
        preemptive=True,
        resources=ResourceSpec(
            partition="gpuunsafe",
            gpus=1,
            cpus=32,
            mem="48G",
            time="48:00:00",
            extra_flags=(_UNSAFE_ACCOUNT,),
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
