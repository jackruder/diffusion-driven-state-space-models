"""Init-centering experiment family for the model-v2 baseline-centering core.

The single named preset ``init_centering_smoke`` wires together every
piece of model-v2 machinery from Phases 1–5 *and* the parametric
factory introduced in Phase B (cell parametrisation):

* Encoder + decoder from the synthetic ``Small1D`` shape (reuse).
* A :class:`BaseBaseline` (zero / identity / linear / MLP) shared
  between the stage-1 ``BaselineGaussian`` transition and the
  stage-2 ``DiffusionV3`` transition.  The canonical cell uses
  :class:`MLPBaseline`.
* :class:`AuxPosterior` for the VHP-via-diffusion init term.
* :class:`SigmaDataBuffer` in the requested tracking mode.  The
  canonical cell uses per-t EMA per ``init-experiment.org`` § 18-cell
  grid.
* :class:`StagesConf` running ``stage_1`` → ``stage_2`` with a
  :class:`CenteringHandoffConf` between them; stage-2's
  ``StageTrainableConf.baseline`` mirrors ``baseline_mode``.

Phase C adds the pilot Optuna sweep (``init_centering_pilot``
experiment + ``init_pilot`` sweep preset).  The full 18-cell grid +
named cell presets land in Phase D.
"""

from . import data, evals, experiments, hparams, model, sweeps

__all__ = ["data", "evals", "experiments", "hparams", "model", "sweeps"]
