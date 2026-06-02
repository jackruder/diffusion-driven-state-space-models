"""The ``synthetic_validation`` family as a :class:`ddssm.cluster.study.Study`.

This is the registration source for the family and the worked example for
``docs/authoring/study.md``. One comparison **axis** — `dataset` — crosses into
one point per dataset, each a full experiment built by :func:`_build`. The study
is registered for launching (``python -m ddssm.launch synthval``) and, via
``into=experiment_store``, its points are published as ``experiment=synthval__*``
presets in one call so the two registries can't desync.

The per-point launch intent is a single training run (no Optuna sweep) — a
"does the model recover known dynamics?" check, not a hyperparameter search.
"""

from __future__ import annotations

from typing import Any, Mapping

from ddssm.launch import PointLaunch, register_study
from ddssm.data.presets import LGSSM, Bimodal, Harmonic
from experiments._make import experiment
from ddssm.cluster.study import Axis, Study, StudyPoint
from ddssm.experiment.builders import Eval, Hparams, Objective, Training
from ddssm.experiment.stores import experiment_store
from experiments.init_centering.hparams import StagesB
from experiments.synthetic_validation.model import SynthValModel

# Dataset axis: tag -> library dataset preset (all D=1, T=32, so one model fits).
DATASETS = {"harmonic": Harmonic, "lgssm": LGSSM, "bimodal": Bimodal}

# One model shape / hparams / training spec, shared across datasets.
_HPARAMS = Hparams(
    S=1, batch_size=32, grad_accum_steps=1,
    enc_lr=5e-4, dec_lr=5e-4, trans_lr=5e-4, ema_decay=0.997,
)
# `steps` is ignored under `stages`; kept > 0 (sanity convention).
_TRAINING = Training(steps=400, log_every=25, amp=True)


def _build(coords: Mapping[str, Any]):
    """Build one dataset's experiment (data + model + training + eval)."""
    data = DATASETS[coords["dataset"]]
    return experiment(
        data=data,
        model=SynthValModel(data_dim=1, latent_dim=1, j=1),
        hparams=_HPARAMS,
        training=_TRAINING,
        # Small two-stage budget so a cell finishes quickly.
        stages=StagesB(baseline_mode="pinned", n_pretrain=100, n_stage2=300),
        eval=Eval(metrics=["mae", "crps_sum", "stage2_elbo_surrogate"], split="val"),
        objective=Objective(metric="loss/total", split="train", source="csv"),
    )


def _launch(point: StudyPoint) -> PointLaunch:
    """One training run per point (no sweep); local-friendly, default resources."""
    return PointLaunch(strategy="single_job", n_trials=1)


def _smoke_overrides(point: StudyPoint) -> list[str]:
    """`--size smoke`: a few steps per stage for a fast end-to-end check."""
    return [
        "experiment.training.stages.n_pretrain=4",
        "experiment.training.stages.n_stage2=4",
        "experiment.training.stages.log_every=1",
        "experiment.training.stages.checkpoint_every=100",
    ]


SYNTHVAL_STUDY = register_study(
    Study.from_axes(
        "synthval",
        axes=[Axis("dataset", list(DATASETS), key=lambda tag: tag)],
        build=_build,
        name_point=lambda tags: f"synthval__{tags['dataset']}",
        launch=_launch,
        variants={"tiny": lambda p: [], "smoke": _smoke_overrides},
    ),
    # Publish each point as ``experiment=synthval__<dataset>`` too.
    into=experiment_store,
)

__all__ = ["SYNTHVAL_STUDY", "DATASETS"]
