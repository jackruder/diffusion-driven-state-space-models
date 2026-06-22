"""Experiments package — imports the dataset presets + experiment families
to populate every hydra-zen store in :mod:`ddssm.experiment.stores`.

``datasets`` registers the library dataset configs (``data=NAME``) and
is imported first; ``init_centering`` is the live model-v2 family;
``synthetic_validation`` is the worked example from the authoring guide
(``docs/authoring/``).
"""

from . import (
    datasets,  # noqa: F401  -- registers library dataset presets
    init_centering,  # noqa: F401  -- model-v2 VHP / centering family
    gluonts_forecast,  # noqa: F401  -- CSDI/TimeGrad NIPS forecasting benchmark
    synthetic_validation,  # noqa: F401  -- authoring-guide worked-example family
)
from ._make import run, to_yaml, override, from_yaml, save_yaml, experiment

__all__ = [
    "experiment", "run", "to_yaml", "save_yaml", "from_yaml", "override",
    "datasets", "init_centering", "gluonts_forecast", "synthetic_validation",
]
