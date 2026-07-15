"""Experiments package — imports the dataset presets + experiment families
to populate every hydra-zen store in :mod:`ddssm.experiment.stores`.

``datasets`` registers the library dataset configs (``data=NAME``) and
is imported first; ``init_centering`` is the live model-v2 family;
``synthetic_validation`` is the worked example from the authoring guide
(``docs/authoring/``).
"""

from . import (
    csdi,
    datasets,
    init_centering,
    gluonts_forecast,
    arflow_headtohead,
    synthetic_validation,
)
from ._make import run, to_yaml, override, from_yaml, save_yaml, experiment

__all__ = [
    "arflow_headtohead",
    "csdi",
    "datasets",
    "experiment",
    "from_yaml",
    "gluonts_forecast",
    "init_centering",
    "override",
    "run",
    "save_yaml",
    "synthetic_validation",
    "to_yaml",
]
