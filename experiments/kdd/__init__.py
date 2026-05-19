"""KDD Cup 2018 PM2.5 experiments family.

Five files per family (see :mod:`experiments.synthetic` for the convention):

* :mod:`.model`        — arch primitives + ``KDD`` shape-namespace + composed DDSSM.
* :mod:`.data`         — :class:`KDDDataModule` config.
* :mod:`.hparams`      — :class:`Hparams` + training-scalar presets.
* :mod:`.evals`        — :class:`Eval` + :class:`Viz` specs.
* :mod:`.experiments`  — named compositions + ``kdd_phase1`` Optuna sweep.
"""

from . import model
from . import data
from . import hparams
from . import evals
from . import experiments


__all__ = ["model", "data", "hparams", "evals", "experiments"]
