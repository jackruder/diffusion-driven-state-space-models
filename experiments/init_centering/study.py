"""The init-centering ablation as a first-class :class:`~experiments._study.Study`.

A :class:`Cell` is one point of the 3-axis grid (baseline_form × baseline_mode
× tracking_mode). Crossed with the dataset axis (:mod:`.datasets`) it yields the
study's registered points. :meth:`Cell.build` is the *single* place that turns a
cell into an experiment — it bakes the real ablation dataset + dims and threads
``baseline_mode`` to both the model and the stage builder from one source.

Size (tiny vs paper-headline) is not a registered axis: presets bake the tiny
``latent_dim``; ``StudyPoint.size_overrides("paper")`` emits the 2× override the
paper launcher applies.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments._make import experiment
from experiments._study import Study, StudyPoint
from experiments.init_centering.cells import cell_name, iter_cells
from experiments.init_centering.datasets import (
    ABLATION_DATASETS,
    AblationDataset,
    paper_latent,
)
from experiments.init_centering.evals import PilotEval, PilotMOObjective
from experiments.init_centering.hparams import SmokeHparams, StagesB, Training800
from experiments.init_centering.model import SmokeModel


@dataclass(frozen=True)
class Cell:
    """One point of the (baseline_form × baseline_mode × tracking_mode) grid."""

    baseline_form: str
    baseline_mode: str
    tracking_mode: str

    @property
    def name(self) -> str:
        return cell_name(self.baseline_form, self.baseline_mode, self.tracking_mode)

    def build(self, ds: AblationDataset):
        """The ``(cell × dataset)`` experiment config, baked at the tiny size."""
        return experiment(
            data=ds.data_preset,
            model=SmokeModel(
                baseline_form=self.baseline_form,
                baseline_mode=self.baseline_mode,
                tracking_mode=self.tracking_mode,
                data_dim=ds.data_dim,
                latent_dim=ds.latent_dim,
            ),
            hparams=SmokeHparams,
            training=Training800,
            # Single source of ``baseline_mode`` -> both the model factory and
            # the stage builder (trainable mask + R_μp anchor strength).
            stages=StagesB(baseline_mode=self.baseline_mode),
            eval=PilotEval,
            # Multi-objective (wallclock_to_target, stage2_elbo_surrogate); pair
            # with the ``init_ablation_moo`` sweeper. Override the target via
            # ``experiment.eval.kwargs.wallclock_to_target.target_value=...``.
            objective=PilotMOObjective,
        )


def _study_point(cell: Cell, ds: AblationDataset) -> StudyPoint:
    def size_overrides(size: str, ds: AblationDataset = ds) -> list[str]:
        if size == "tiny":
            return []
        if size == "paper":
            return [f"experiment.model.latent_dim={paper_latent(ds)}"]
        raise ValueError(f"unknown size {size!r}; expected 'tiny' or 'paper'")

    return StudyPoint(
        name=f"{cell.name}__{ds.label}",
        config=cell.build(ds),
        tags={
            "cell": cell.name,
            "baseline_form": cell.baseline_form,
            "baseline_mode": cell.baseline_mode,
            "tracking_mode": cell.tracking_mode,
            "dataset": ds.label,
        },
        size_overrides=size_overrides,
    )


def _cells() -> list[Cell]:
    return [Cell(form, mode, tracking) for form, mode, tracking in iter_cells()]


# 12 cells × 2 datasets = 24 registered points.
INIT_CENTERING_STUDY = Study(
    name="init_centering",
    points=tuple(
        _study_point(cell, ds) for cell in _cells() for ds in ABLATION_DATASETS
    ),
    sweep="init_ablation_moo",
)


__all__ = ["Cell", "INIT_CENTERING_STUDY"]
