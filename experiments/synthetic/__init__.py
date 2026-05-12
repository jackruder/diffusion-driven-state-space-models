"""Synthetic-data experiments family.

Importing this subpackage triggers registration of every named
encoder / decoder / z_init / unet / schedule / transition /
dataset / hparams / training / eval / viz / model / experiment
to the corresponding store in :mod:`conf.registry`.

Dependency order matters — leaf pieces register first, then
composed configs. The imports below reflect that order.
"""

# Leaf component groups.
from . import encoders, decoders, z_inits, unets, schedules
from . import hparams, training, evals, vizs, datasets

# Composed groups (depend on leaves).
from . import transitions
from . import models
from . import experiments

# Optuna sweep presets.
from . import sweeps

__all__ = [
    "encoders", "decoders", "z_inits", "unets", "schedules",
    "hparams", "training", "evals", "vizs", "datasets",
    "transitions", "models", "experiments", "sweeps",
]
