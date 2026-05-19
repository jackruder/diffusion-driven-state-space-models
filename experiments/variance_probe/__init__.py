"""Variance-probe experiments family.

DiffusionV2 transition + Probe spec over short (300-step) runs on a
handful of synthetic datasets. Reuses the synthetic family's encoder /
decoder / z_init shapes — :mod:`experiments.synthetic` must import
first (the top-level :mod:`experiments` ``__init__`` enforces the order).

Five files per family (see :mod:`experiments.synthetic` for the convention):

* :mod:`.model`        — DiffusionV2 transition + DDSSM composition.
* :mod:`.data`         — :class:`SyntheticDataModule` configs at smaller ``N_per_split``.
* :mod:`.hparams`      — :class:`Hparams` + training-scalar presets.
* :mod:`.evals`        — :class:`Objective` + :class:`Probe` specs.
* :mod:`.experiments`  — named compositions + ``variance_probe`` sweep preset.
"""

from . import model
from . import data
from . import hparams
from . import evals
from . import experiments


__all__ = ["model", "data", "hparams", "evals", "experiments"]
