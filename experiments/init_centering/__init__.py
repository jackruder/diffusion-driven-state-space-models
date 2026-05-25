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
experiment + ``init_pilot`` sweep preset).  Phase D registers the full
18-cell grid (one named preset per cell, see :mod:`.cells` for the
enumerator) plus two explicit control presets
(``init_canonical_ctrl_sigma0`` and ``init_canonical_ctrl_npretrain0``)
that pin the two sweep knobs at zero — values Optuna's log-uniform
prior cannot sample.  See :mod:`.launch_phase_d` for the SLURM sbatch
helper that emits one job per cell.

Phase E (:mod:`.report`) is the reporting layer.  It scans every cell's
sweep dir + matching Optuna DB, serialises the result to
``summary.csv`` + ``records.jsonl``, and renders the three headline
artifacts (σ_data drift trajectory plot, wallclock-to-target bar
chart, markdown headline table) from the JSONL records alone — so
plot iterations never touch the model or re-scan disk.
"""

from . import cells, data, evals, experiments, hparams, model, sweeps

__all__ = [
    "cells",
    "data",
    "evals",
    "experiments",
    "hparams",
    "model",
    "sweeps",
]
