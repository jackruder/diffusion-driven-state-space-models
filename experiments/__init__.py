"""Experiments package — imports the dataset presets + experiment families
to populate every hydra-zen store in :mod:`ddssm.stores`.

``datasets`` registers the library dataset configs (``data=NAME``) and
is imported first; ``init_centering`` is the live model-v2 family.
"""

from . import datasets         # noqa: F401  -- registers library dataset presets
from . import init_centering   # noqa: F401  -- model-v2 VHP / centering family

from ._make import experiment, from_yaml, override, run, save_yaml, to_yaml

__all__ = [
    "experiment", "run", "to_yaml", "save_yaml", "from_yaml", "override",
    "datasets", "init_centering",
]
