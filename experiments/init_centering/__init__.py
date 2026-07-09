"""Init-centering experiment family for the model-v2 baseline-centering core.

Two role-specific smoke presets are the canonical entry points:

* ``init_smoke_simple`` — ``(zero, fixed)`` on the 1D ablation dataset.
  Minimum surface + V2 numerical anchor.
* ``init_smoke_high_surface`` — ``(persistence, per_t)`` on the MV
  ablation dataset. Exercises the parameter-free persistence baseline +
  per-t σ_data EMA + MV observation lift.

Both smokes wire the factory ``_build_init_centering_model``: a
parameter-free :class:`BaseBaseline` (zero / persistence, ``σ_p² = 1``)
consumed by the :class:`DiffusionTransition`, an :class:`AuxPosterior`
for the VHP-via-diffusion init term, and a :class:`SigmaDataBuffer` in
the requested tracking mode. Training is a single
``trainer.fit(...)`` keyed on ``training.steps``.

The full ablation grid is a first-class library :class:`~ddssm.cluster.study.Study`
(:mod:`.study`); see :mod:`.cells` for the cell enumerator and
:mod:`.datasets` for the dataset axis. 4 cells × 2 datasets = 8
registered presets named ``init_<cell>__<dataset>``. Launching is via the
generic ``python -m ddssm.launch init_centering`` CLI (ADR-0007/0008).

:mod:`.report` is the reporting layer.  It scans every study point's
sweep dir + matching Optuna DB, serialises the result to
``summary.csv`` + ``records.jsonl``, and renders the headline artifacts.
"""

from . import data, cells, evals, model, study, sweeps, hparams, experiments

__all__ = [
    "cells",
    "data",
    "evals",
    "experiments",
    "hparams",
    "model",
    "study",
    "sweeps",
]
