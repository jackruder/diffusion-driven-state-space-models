"""Variance-probe experiments family.

DiffusionV2 transition + Probe spec over short (300-step) runs on a
handful of synthetic datasets. Importing this subpackage registers
every named config; the dependency graph requires
:mod:`experiments.synthetic` to be imported first (this subpackage's
models reuse synthetic encoders/decoders/z_inits) — the top-level
:mod:`experiments` ``__init__`` handles the ordering.
"""

from . import schedules, hparams, training, datasets
from . import transitions
from . import models
from . import experiments

__all__ = [
    "schedules", "hparams", "training", "datasets",
    "transitions", "models", "experiments",
]
