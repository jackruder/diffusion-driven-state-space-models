"""Worked-example experiment family for the authoring guide (``docs/authoring/``).

Trains one simple, hand-built DDSSM model across several 1-D synthetic datasets
(a "does it recover known dynamics?" validation harness). Importing this package
registers the ``synthval__<dataset>`` presets into the hydra-zen ``experiment``
store; the import is triggered from :mod:`experiments`.
"""

from . import study  # noqa: F401  -- registers the synthval study + synthval__* presets
from . import sin_overfit  # noqa: F401  -- registers the sin_overfit sanity-check preset
