"""Smoke-test experiment family for the model-v2 baseline-centering core.

Phase 6 of the staged implementation.  One named preset
(``init_centering_smoke``) wires together every piece of the model-v2
machinery from Phases 1–5:

* Encoder + decoder from the synthetic ``Small1D`` shape (reuse).
* :class:`MLPBaseline` shared between the stage-1 ``BaselineGaussian``
  transition and the stage-2 ``DiffusionV3`` transition.
* :class:`AuxPosterior` for the VHP-via-diffusion init term.
* :class:`SigmaDataBuffer` in "fixed" tracking mode.
* :class:`StagesConf` running ``stage_1`` → ``stage_2`` with a
  :class:`CenteringHandoffConf` between them.

The 18-cell ablation grid and Optuna sweep are explicitly out of
scope for this PR.
"""

from . import data
from . import experiments
from . import hparams
from . import model

__all__ = ["data", "experiments", "hparams", "model"]
