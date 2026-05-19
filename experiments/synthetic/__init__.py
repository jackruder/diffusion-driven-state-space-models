"""Synthetic-data experiments family.

Importing this subpackage triggers registration of every encoder /
decoder / z_init / unet / schedule / transition / dataset / hparams /
training / eval / viz / model / experiment / sweep to the corresponding
store in :mod:`conf.registry`.

Five files per family:

* :mod:`.model`        — arch primitives + shape-namespace classes +
                          encoder/decoder/z_init/transition/unet/
                          schedule/model registrations.
* :mod:`.data`         — :class:`SyntheticDataModule` configs.
* :mod:`.hparams`      — :class:`Hparams` + training-scalar presets.
* :mod:`.evals`        — :class:`Eval` + :class:`Viz` specs.
* :mod:`.experiments`  — named :class:`ExperimentC` compositions + Optuna sweeps.
"""

from . import model
from . import data
from . import hparams
from . import evals
from . import experiments


__all__ = ["model", "data", "hparams", "evals", "experiments"]
