"""The init-centering ablation, declared as a library :class:`ddssm.study.Study`.

Two comparison axes — `cell` (the 12-point grid) × `dataset` (1d/mv) — cross into
24 registered presets named ``init_<cell>__<dataset>``. The `cell` axis exposes
its sub-fields (baseline_form/mode/tracking) as tags for filtering + reporting.
Each point's launch intent is a single-node Optuna sweep; size variants (paper,
smoke) are override hooks the orchestrator applies.
"""

from __future__ import annotations

from typing import Any, Mapping

from ddssm.launch import PointLaunch, ResourceSpec, register_study
from ddssm.study import Axis, Study, StudyPoint

from experiments._make import experiment
from experiments.init_centering.cells import Cell, iter_cells
from experiments.init_centering.datasets import ABLATION_DATASETS, paper_latent
from experiments.init_centering.evals import PilotEval, PilotMOObjective
from experiments.init_centering.hparams import SmokeHparams, StagesB, Training800
from experiments.init_centering.model import SmokeModel


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
        stages=StagesB(baseline_mode=cell.baseline_mode),
        eval=PilotEval,
        objective=PilotMOObjective,
    )


def _launch(point: StudyPoint) -> PointLaunch:
    """Round-2 launch intent: GPU-packed Optuna MOO, one GPU per cell.

    Round-2 sweeps the ``per_t`` cells only (drop ``fixed`` at launch via
    ``--select tracking_mode=per_t``). Every cell uses the narrowed
    ``init_ablation_moo_r2`` search space. Each trial is <4 GiB, so we pack
    workers onto one GPU (4 CPUs each, thread-pinned — the round-1 starvation
    fix). ``n_trials`` is the cell-total budget split across packed workers
    (~``n_trials / n_workers`` each), independent of ``preemptive``.

    GPU pool = 1 A100 + 1 A40 on ``gpupriority`` (non-preempt) + 10 A40 on
    ``gpuunsafe`` (preemptible → checkpoint/resume, ADR-0009). The two
    gpupriority GPUs are dedicated to two named, non-preemptive cells (so the
    rest can't grab them); everything else runs preemptibly on gpuunsafe.
    """
    cell = point.coords["cell"]
    sweep = "init_ablation_moo_r2"  # narrowed r2 for every cell

    # --- gpupriority (non-preempt): two dedicated cells, the 2 priority GPUs ---
    # Headline cell -> the A100 (2x memory -> 2x the pack). 64 / 16 = 4 trials/worker.
    if cell.name == "init_mlp_pinned_per_t":
        return PointLaunch(
            strategy="optuna_packed_node", sweep=sweep,
            n_trials=64, n_workers=16, workers_per_gpu=16, preemptive=False,
            resources=ResourceSpec(
                partition="gpupriority", gpus=1, cpus=64, mem="96G", time="24:00:00",
                extra_flags=("--gres=gpu:a100:1",),  # confirm Tempest's A100 gres name
            ),
        )
    # Second non-preempt cell -> the gpupriority A40 (swap this name to re-pick).
    if cell.name == "init_mlp_learnable_per_t":
        return PointLaunch(
            strategy="optuna_packed_node", sweep=sweep,
            n_trials=64, n_workers=8, workers_per_gpu=8, preemptive=False,
            resources=ResourceSpec(
                partition="gpupriority", gpus=1, cpus=32, mem="48G", time="24:00:00",
                extra_flags=("--gres=gpu:a40:1",),  # confirm Tempest's A40 gres name
            ),
        )

    # --- gpuunsafe (preemptible): every other cell, 8 packed workers on an A40 ---
    return PointLaunch(
        strategy="optuna_packed_node", sweep=sweep,
        n_trials=64, n_workers=8, workers_per_gpu=8, preemptive=True,
        resources=ResourceSpec(
            partition="gpuunsafe", gpus=1, cpus=32, mem="48G", time="24:00:00",
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
        variants={"tiny": lambda p: [], "paper": _paper_overrides, "smoke": _smoke_overrides},
    )
)

__all__ = ["INIT_CENTERING_STUDY"]
