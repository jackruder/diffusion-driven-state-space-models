"""Named gluonts_forecast presets.

The five per-dataset presets ``gluonts_forecast__<dataset>`` are registered by
the Study (``study.py`` → ``register_study(..., into=experiment_store)``). This
module adds one tiny **solar smoke** (``gluonts_smoke``) for pipeline validation:

    python -m ddssm.app experiment=gluonts_smoke
"""

from __future__ import annotations

import dataclasses

from experiments._make import experiment
from ddssm.experiment.stores import experiment_store
from experiments.gluonts_forecast.evals import GluonEval, ValElboObjective
from experiments.gluonts_forecast.model import GluonModel
from experiments.gluonts_forecast.hparams import (
    GluonStages,
    GluonHparams,
    GluonTraining,
)
from experiments.gluonts_forecast.datasets import GLUONTS_BY_LABEL

_solar = GLUONTS_BY_LABEL["solar"]

gluonts_smoke = experiment(
    data=_solar.data_preset,
    model=GluonModel(data_dim=_solar.data_dim, T_max=_solar.T_max, latent_dim=16),
    hparams=dataclasses.replace(GluonHparams, batch_size=16),
    training=GluonTraining,
    stages=GluonStages(
        n_pretrain=20,
        n_stage2=20,
        log_every=5,
        validate_every=10,
        checkpoint_every=100,
    ),
    eval=GluonEval,
    objective=ValElboObjective,
)
experiment_store(gluonts_smoke, name="gluonts_smoke")

__all__ = ["gluonts_smoke"]
