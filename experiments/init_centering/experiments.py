"""Named init-centering experiments: two role-specific smokes + ablation grid.

The two smoke presets are the canonical entry points (CONTEXT.md § Simple-smoke
cell / High-surface-smoke cell):

- :data:`init_smoke_simple` — ``(zero, pinned, fixed)`` on the 1D ablation
  dataset. Minimum surface + numerical V2 anchor. Use to validate the
  pipeline is wired correctly without changing any cell-machinery
  defaults.
- :data:`init_smoke_high_surface` — ``(mlp, learnable, per_t)`` on the MV
  ablation dataset. Exercises every code path: parametric μ_p, R_μp
  regulariser under Learnable, per-t σ_data EMA, multivariate
  observation lift. If this trains end-to-end, every grid cell does.

The legacy ``init_centering_smoke`` and ``init_centering_pilot`` presets
were dropped per CONTEXT.md (the term "pilot" was overloaded). Use the
two smokes above + the ablation grid + the ``init_ablation`` sweep
instead.
"""

from __future__ import annotations

from conf.registry import experiment_store
from experiments._make import experiment
from experiments.init_centering.data import (
    Harmonic,
    NonlinBimodalLift1D,
    NonlinBimodalLiftMV,
)
from experiments.init_centering.cells import (
    cell_name,
    iter_cells,
)
from experiments.init_centering.evals import (
    PilotEval,
    PilotMOObjective,
    PilotObjective,
)
from experiments.init_centering.model import SmokeModel
from experiments.init_centering.hparams import StagesB, Training800, SmokeHparams

# ---------------------------------------------------------------------------
# Simple-smoke cell: (zero, pinned, fixed) on the 1D ablation dataset.
# No eval wired; ``train()`` returns the trainer for inspection.
# Pairs with the V2-reduction test as the project's correctness anchor.
#
# Run: ``python -m ddssm.app experiment=init_smoke_simple``.
# ---------------------------------------------------------------------------

init_smoke_simple = experiment(
    data=NonlinBimodalLift1D,
    model=SmokeModel(
        baseline_form="zero",
        baseline_mode="pinned",
        tracking_mode="fixed",
        latent_dim=1,
        data_dim=1,
    ),
    hparams=SmokeHparams,
    training=Training800,
    stages=StagesB(baseline_mode="pinned"),
)
experiment_store(init_smoke_simple, name="init_smoke_simple")


# ---------------------------------------------------------------------------
# High-surface-smoke cell: (mlp, learnable, per_t) on the MV ablation
# dataset. Exercises parametric μ_p + R_μp + per-t σ_data EMA + the MV
# observation lift. Eval + objective wired so it can also act as a
# single-trial inspection target before launching the full sweep.
#
# Run a single trial: ``python -m ddssm.app experiment=init_smoke_high_surface``
# Run an exploratory sweep:
#   python -m ddssm.app --multirun \
#       experiment=init_smoke_high_surface +sweep=init_ablation \
#       hydra.sweeper.n_trials=10
# ---------------------------------------------------------------------------

init_smoke_high_surface = experiment(
    data=NonlinBimodalLiftMV,
    model=SmokeModel(
        baseline_form="mlp",
        baseline_mode="learnable",
        tracking_mode="per_t",
        latent_dim=4,
        data_dim=8,
    ),
    hparams=SmokeHparams,
    training=Training800,
    stages=StagesB(baseline_mode="learnable"),
    eval=PilotEval,
    objective=PilotObjective,
)
experiment_store(init_smoke_high_surface, name="init_smoke_high_surface")


# ---------------------------------------------------------------------------
# Phase D — the full ablation grid.
#
# One named preset per cell, all sharing the canonical-cell training
# scaffold but with the three cell axes (``baseline_form``,
# ``baseline_mode``, ``tracking_mode``) varied across the grid.  Each
# cell wires the Phase-A eval pipeline + the JSON-source
# ``stage2_elbo_surrogate`` objective so it can plug straight into the
# pilot Optuna sweep (``+sweep=init_pilot``).
#
# Run a single cell:
#   python -m ddssm.app experiment=init_mlp_pinned_per_t
#
# Sweep a single cell (20 trials):
#   python -m ddssm.app --multirun \
#       experiment=init_mlp_pinned_per_t +sweep=init_pilot \
#       hydra.sweeper.n_trials=20 \
#       hydra.sweeper.study_name=phase_d_mlp_pinned_per_t
#
# Submit every cell via SLURM:
#   python -m experiments.init_centering.launch_phase_d --write-dir runs/sbatch/phase_d
# ---------------------------------------------------------------------------

for _form, _mode, _tracking in iter_cells():
    _cell_exp = experiment(
        data=Harmonic,
        model=SmokeModel(
            baseline_form=_form,
            baseline_mode=_mode,
            tracking_mode=_tracking,
        ),
        hparams=SmokeHparams,
        training=Training800,
        stages=StagesB(baseline_mode=_mode),
        eval=PilotEval,
        # Multi-objective: (wallclock_to_target, stage2_elbo_surrogate).
        # Pair with the ``ddssm_optuna_moo`` sweeper preset that sets
        # ``direction: [minimize, minimize]``. Override the target
        # value via Hydra
        # ``experiment.eval.kwargs.wallclock_to_target.target_value=...``.
        objective=PilotMOObjective,
    )
    experiment_store(_cell_exp, name=cell_name(_form, _mode, _tracking))


# NOTE: ``init_canonical_ctrl_sigma0`` and ``init_canonical_ctrl_npretrain0``
# were removed per docs/adr/0002-drop-canonical-controls.md: σ_pert > 0
# is mandatory protocol (no σ_pert=0 mode) and n_pretrain=0 is meaningless
# for parametric μ_p cells. The sweep range on σ_pert covers
# "operationally indistinguishable from 0" instead.
