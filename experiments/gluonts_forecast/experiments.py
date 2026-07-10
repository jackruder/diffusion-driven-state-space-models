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
    GluonHparams,
    GluonTraining,
)
from experiments.gluonts_forecast.datasets import GLUONTS_BY_LABEL

_solar = GLUONTS_BY_LABEL["solar"]

_smoke_training = dataclasses.replace(
    GluonTraining,
    steps=40,
    log_every=5,
    validate_every=10,
    checkpoint_every=100,
)

gluonts_smoke = experiment(
    data=_solar.data_preset,
    # nheads=4 keeps head_dim = 2·latent/nheads = 8, the SDPA minimum.
    model=GluonModel(
        data_dim=_solar.data_dim, T_max=_solar.T_max, latent_dim=16, nheads=4
    ),
    hparams=dataclasses.replace(GluonHparams, batch_size=16),
    training=_smoke_training,
    eval=GluonEval,
    objective=ValElboObjective,
)
experiment_store(gluonts_smoke, name="gluonts_smoke")

__all__ = ["gluonts_smoke"]
