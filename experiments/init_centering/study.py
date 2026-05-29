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
    # Single-node Optuna sweep per point. Resource shape could scale off
    # ``point.coords`` (e.g. larger latent -> more cpus); uniform for now.
    return PointLaunch(
        strategy="optuna_single_node",
        sweep="init_ablation_moo",
        n_trials=40,
        resources=ResourceSpec(time="08:00:00", gpus=1, cpus=4, mem="32G"),
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
