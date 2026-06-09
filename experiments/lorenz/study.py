"""The Lorenz attractor sweep study.

Two structural axes — ``latent_dim`` ∈ {4, 8} × ``frozen_enc`` ∈ {False, True} —
cross into 4 registered presets named ``lorenz_4d_open_enc``,
``lorenz_4d_frozen_enc``, ``lorenz_8d_open_enc``, ``lorenz_8d_frozen_enc``.

Each cell is launched as a single-node MOO Optuna sweep (``lorenz_ablation_moo``)
over 7 HPO axes (base_lr, dec_mult, trans_mult, stage2_trans_lr, n_pretrain,
n_stage2, stage_2_warmup_frac).  The Pareto front separates fast-converging
configs from deep-fitting ones along the two objectives:
  obj0: steps to first reach loss/total ≤ 120 (minimize)
  obj1: stage2_elbo_surrogate tail mean (minimize)
"""

from __future__ import annotations

from typing import Any, Mapping

from ddssm.cluster.study import Axis, Study, StudyPoint
from ddssm.experiment.stores import experiment_store
from ddssm.launch import PointLaunch, ResourceSpec, register_study
from experiments._make import experiment
from experiments.init_centering.hparams import SmokeHparams, Training800
from experiments.init_centering.model import SmokeModel
from experiments.lorenz.cells import LorenzCell, iter_cells
from experiments.lorenz.data import LorenzDirect
from experiments.lorenz.evals import LorenzEval, LorenzMOObjective
from experiments.lorenz.hparams import LorenzStagesSweep, LorenzStagesSweepFrozenEnc


def _build(coords: Mapping[str, Any]):
    cell: LorenzCell = coords["cell"]
    stages = LorenzStagesSweepFrozenEnc if cell.frozen_enc else LorenzStagesSweep
    return experiment(
        data=LorenzDirect,
        model=SmokeModel(
            baseline_form="mlp",
            baseline_mode="pinned",
            tracking_mode="per_t",
            latent_dim=cell.latent_dim,
            data_dim=3,
            T_max=64,
        ),
        hparams=SmokeHparams,
        training=Training800,
        stages=stages,
        eval=LorenzEval,
        objective=LorenzMOObjective,
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


def _launch(point: StudyPoint) -> PointLaunch:
    """All Lorenz cells run on a single A100 (gpupriority), 32 trials, 8 workers."""
    return PointLaunch(
        strategy="optuna_packed_node",
        sweep="lorenz_ablation_moo",
        n_trials=32,
        n_workers=8,
        workers_per_gpu=8,
        preemptive=False,
        resources=ResourceSpec(
            partition="gpupriority",
            gpus=1,
            cpus=32,  # 8 workers x 4 CPUs
            mem="64G",
            time="24:00:00",
            extra_flags=("--gres=gpu:a100:1", _PRIORITY_ACCOUNT),
            setup=_TEMPEST_SETUP,
        ),
    )


def _smoke_overrides(point: StudyPoint) -> list[str]:
    return [
        "experiment.training.stages.n_pretrain=5",
        "experiment.training.stages.n_stage2=5",
        "experiment.training.stages.log_every=1",
        "experiment.training.stages.checkpoint_every=100",
    ]


LORENZ_STUDY = register_study(
    Study.from_axes(
        "lorenz",
        axes=[
            Axis(
                "cell",
                list(iter_cells()),
                key=lambda c: c.name,
                tags=lambda c: {
                    "latent_dim": str(c.latent_dim),
                    "frozen_enc": str(c.frozen_enc),
                },
            ),
        ],
        build=_build,
        name_point=lambda tags: tags["cell"],
        launch=_launch,
        variants={
            "tiny": lambda p: [],
            "smoke": _smoke_overrides,
        },
    ),
    into=experiment_store,
)

__all__ = ["LORENZ_STUDY"]
