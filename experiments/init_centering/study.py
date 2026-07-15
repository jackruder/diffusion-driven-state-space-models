"""The init-centering ablation, declared as a library :class:`ddssm.cluster.study.Study`.

Two comparison axes — `cell` (the 4-point grid) × `dataset` (1d/mv) — cross into
8 registered presets named ``init_<cell>__<dataset>``. The `cell` axis exposes
its sub-fields (baseline_form/tracking_mode) as tags for filtering + reporting.
Each point's launch intent is a single-node Optuna sweep; size variants (paper,
smoke) are override hooks the orchestrator applies.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

from ddssm.launch import PointLaunch, ResourceSpec, register_study
from experiments._make import experiment
from ddssm.cluster.study import Axis, Study, StudyPoint
from ddssm.experiment.stores import experiment_store
from experiments.init_centering.cells import Cell, iter_cells
from experiments.init_centering.evals import PilotEval, PilotMOObjective
from experiments.init_centering.model import SmokeModel
from experiments.init_centering.hparams import Training800, SmokeHparams
from experiments.init_centering.datasets import ABLATION_DATASETS, paper_latent


def _build(coords: Mapping[str, Any]):
    """One (cell × dataset) experiment, baking the real dataset + dims (tiny size)."""
    cell: Cell = coords["cell"]
    ds = coords["dataset"]
    return experiment(
        data=ds.data_preset,
        model=SmokeModel(
            baseline_form=cell.baseline_form,
            tracking_mode=cell.tracking_mode,
            data_dim=ds.data_dim,
            latent_dim=ds.latent_dim,
        ),
        hparams=SmokeHparams,
        training=Training800,
        eval=PilotEval,
        objective=PilotMOObjective,
    )


_TEMPEST_SETUP = (
    "set +eu",
    "source /etc/profile",
    "module purge",
    "module load Python/3.13.5-GCCcore-14.3.0 CUDA/13.0.0 tools/uv/0.9.22",
    "source .venv/bin/activate",
    "set -eu",
)

_PRIORITY_ACCOUNT = "--account=priority-michaelwojnowicz"
_UNSAFE_ACCOUNT = "--account=group-michaelwojnowicz"


def _launch(point: StudyPoint) -> PointLaunch:
    """Launch intent: GPU-packed Optuna MOO on the ``gpuunsafe`` partition."""
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
            cpus=32,
            mem="80G",
            time="48:00:00",
            extra_flags=("--gres=gpu:a40:1", _UNSAFE_ACCOUNT),
            setup=_TEMPEST_SETUP,
        ),
    )


def _paper_overrides(point: StudyPoint) -> list[str]:
    return [
        f"experiment.model.module.latent_dim={paper_latent(point.coords['dataset'])}"
    ]


def _smoke_overrides(point: StudyPoint) -> list[str]:
    return [
        "experiment.training.steps=5",
        "experiment.training.log_every=1",
        "experiment.training.checkpoint_every=100",
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
    into=experiment_store,
)

__all__ = ["INIT_CENTERING_STUDY"]
