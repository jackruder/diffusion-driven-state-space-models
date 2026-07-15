"""The gluonts_forecast benchmark as a library :class:`ddssm.cluster.study.Study`.

One axis — ``dataset`` (solar/electricity/traffic/taxi/wiki) — gives 5 registered
presets ``gluonts_forecast__<dataset>``. Each point's launch intent is a
guaranteed-A100 Optuna sweep on the validation ELBO, 3 workers sharing the
per-dataset study (one worker per A100). Budgets/launch sizes are pilot-tunable.
"""

from __future__ import annotations

from typing import Any
import dataclasses
from collections.abc import Mapping

from ddssm.launch import PointLaunch, ResourceSpec, register_study
from experiments._make import experiment
from ddssm.cluster.study import Axis, Study, StudyPoint
from ddssm.experiment.stores import experiment_store
from experiments.gluonts_forecast.evals import GluonEval, ValElboObjective
from experiments.gluonts_forecast.model import GluonModel
from experiments.gluonts_forecast.hparams import (
    GluonHparams,
    GluonTraining,
)
from experiments.gluonts_forecast.datasets import GLUONTS_DATASETS


def _build(coords: Mapping[str, Any]):
    """One per-dataset experiment, baking data_dim / T_max / per-dataset batch."""
    ds = coords["dataset"]
    return experiment(
        data=ds.data_preset,
        model=GluonModel(data_dim=ds.data_dim, T_max=ds.T_max, latent_dim=64),
        hparams=dataclasses.replace(GluonHparams, batch_size=ds.batch_size),
        training=GluonTraining,
        eval=GluonEval,  # dormant during the sweep (csv objective);
        objective=ValElboObjective,  # run via ddssm.evaluate on finalists.
    )


# Compute-node bring-up (Tempest). Mirrors the init-centering study.
_TEMPEST_SETUP = (
    "set +eu",
    "source /etc/profile",
    "module purge",
    "module load Python/3.13.5-GCCcore-14.3.0 CUDA/13.0.0 tools/uv/0.9.22",
    "source .venv/bin/activate",
    # torch.compile/triton: use the CUDA-toolkit ptxas (triton's bundled one can
    # be incompatible) and point at libcuda only where the NixOS driver path
    # exists — a no-op on a normal cluster where ldconfig finds it. Compile is
    # on by default (DDSSM_TORCH_COMPILE=auto); these make it real, not eager.
    "export TRITON_PTXAS_PATH=$(command -v ptxas)",
    "[ -e /run/opengl-driver/lib/libcuda.so.1 ] && export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib || true",
    "set -eu",
)
# gpupriority admits only the priority-* accounts.
_PRIORITY_ACCOUNT = "--account=priority-michaelwojnowicz"


def _launch(point: StudyPoint) -> PointLaunch:
    """Guaranteed A100 (gpupriority), 3 workers sharing the per-dataset study.

    ``optuna_packed_node`` with ``workers_per_gpu=1`` emits one sbatch per
    worker, so n_workers=3 → 3 independent A100 jobs joining the same Optuna DB
    (1 trial/GPU in flight). Non-preemptive (guaranteed slots) → no requeue
    machinery. Budgets/worker-count are pilot-tunable.
    """
    return PointLaunch(
        strategy="optuna_packed_node",
        sweep="gluonts_lean",
        n_trials=128,
        n_workers=3,
        workers_per_gpu=1,
        preemptive=False,
        resources=ResourceSpec(
            partition="gpupriority",
            gpus=1,
            cpus=8,
            mem="80G",
            time="48:00:00",
            extra_flags=("--gres=gpu:a100:1", _PRIORITY_ACCOUNT),
            setup=_TEMPEST_SETUP,
        ),
    )


def _smoke_overrides(point: StudyPoint) -> list[str]:
    return [
        "experiment.training.steps=40",
        "experiment.training.log_every=5",
        "experiment.training.validate_every=10",
        "experiment.training.checkpoint_every=100",
        "experiment.model.module.latent_dim=16",
    ]


def _pilot_overrides(point: StudyPoint) -> list[str]:
    # Reduced training budget for the solar pilot (pin T_train + proxy validation).
    return ["experiment.training.steps=8000"]


GLUONTS_FORECAST_STUDY = register_study(
    Study.from_axes(
        "gluonts_forecast",
        axes=[
            Axis("dataset", GLUONTS_DATASETS, key=lambda d: d.label),
        ],
        build=_build,
        name_point=lambda tags: f"gluonts_forecast__{tags['dataset']}",
        launch=_launch,
        variants={
            "full": lambda p: [],
            "pilot": _pilot_overrides,
            "smoke": _smoke_overrides,
        },
    ),
    into=experiment_store,
)

__all__ = ["GLUONTS_FORECAST_STUDY"]
