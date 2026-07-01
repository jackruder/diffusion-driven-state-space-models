"""The experiment composition layer.

:class:`~ddssm.experiment.experiment.Experiment` is the single composition
point that owns the data module, model, trainer factory, and the eval/viz/
variance specs; it is built via hydra-zen and dispatched by Hydra. The
``builders``, ``stores``, and ``registry`` modules provide the config
surface and preset registration around it.

The public objects are re-exported here so ``from ddssm.experiment import
Experiment`` keeps working after the move from a flat module to a package.
"""

from ddssm.experiment.experiment import (
    SBatch,
    Experiment,
    Objectives,
    ObjectiveSpec,
    TrainingScalars,
    _as_objective_spec,
)

__all__ = [
    "Experiment",
    "ObjectiveSpec",
    "Objectives",
    "SBatch",
    "TrainingScalars",
    "_as_objective_spec",
]
