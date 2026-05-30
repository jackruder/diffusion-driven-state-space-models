"""Init-centering experiment family for the model-v2 baseline-centering core.

Two role-specific smoke presets are the canonical entry points
(CONTEXT.md § Simple-smoke cell / High-surface-smoke cell):

* ``init_smoke_simple`` — ``(zero, pinned, fixed)`` on the 1D
  ablation dataset. Minimum surface + V2 numerical anchor.
* ``init_smoke_high_surface`` — ``(mlp, learnable, per_t)`` on the MV
  ablation dataset. Exercises every code path of the cell machinery
  (parametric μ_p, R_μp regulariser under Learnable, per-t σ_data
  EMA, MV observation lift).

Both smokes wire the parametric factory introduced in Phase B
(``_build_init_centering_model``): a shared :class:`BaseBaseline`
(zero / identity / linear / MLP) between the stage-1
``BaselineGaussian`` transition and the stage-2 ``DiffusionV3``
transition, plus :class:`AuxPosterior` for the VHP-via-diffusion init
term, :class:`SigmaDataBuffer` in the requested tracking mode, and a
:class:`StagesConf` running ``stage_1`` → ``stage_2`` with a
:class:`CenteringHandoffConf` between them.

The full ablation grid is a first-class library :class:`~ddssm.study.Study`
(:mod:`.study`); see :mod:`.cells` for the cell enumerator and
:mod:`.datasets` for the dataset axis. 12 cells × 2 datasets = 24 registered
presets named ``init_<cell>__<dataset>``. The Optuna sweep
``+sweep=init_ablation_moo`` defines the multi-objective search space; the
two ``init_canonical_ctrl_*`` presets were removed per
``docs/adr/0002-drop-canonical-controls.md`` (σ_pert > 0 is mandatory
protocol; n_pretrain = 0 is meaningless for parametric μ_p cells). The
legacy ``init_centering_smoke`` / ``init_centering_pilot`` presets are
replaced by the two role-specific smokes above (CONTEXT.md drops the "pilot"
terminology). Launching is via the generic
``python -m ddssm.launch init_centering`` CLI (ADR-0007/0008); the per-family
launchers ``launch_phase_d`` / ``launch_ablation_tiny`` /
``launch_paper_headline`` / ``smoke_phase_d`` were deleted.

:mod:`.report` is the reporting layer.  It scans every study point's
sweep dir + matching Optuna DB, serialises the result to
``summary.csv`` + ``records.jsonl``, and renders the three headline
artifacts (σ_data drift trajectory plot, wallclock-to-target bar
chart, markdown headline table) from the JSONL records alone — so
plot iterations never touch the model or re-scan disk.
"""

from . import data, cells, evals, model, sweeps, hparams, experiments

__all__ = [
    "cells",
    "data",
    "evals",
    "experiments",
    "hparams",
    "model",
    "sweeps",
]
