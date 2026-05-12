"""Experiments package — imports the three family subpackages to
populate every hydra-zen store in :mod:`conf.registry`.

Dependency order matters: ``variance_probe`` reuses
encoder/decoder/z_init Confs from ``synthetic``, so ``synthetic``
must import first.
"""

from . import synthetic       # noqa: F401  -- imports first; defines small + robot encoders/etc
from . import variance_probe  # noqa: F401  -- depends on synthetic encoders/decoders/z_inits
from . import kdd             # noqa: F401  -- independent

from ._make import experiment, from_yaml, override, run, save_yaml, to_yaml

__all__ = [
    "experiment", "run", "to_yaml", "save_yaml", "from_yaml", "override",
    "synthetic", "variance_probe", "kdd",
]
