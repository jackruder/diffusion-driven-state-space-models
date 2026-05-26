"""Named init-centering experiments: smoke + pilot + Phase-D 18-cell grid."""

from __future__ import annotations

from conf.registry import experiment_store
from experiments._make import experiment
from experiments.init_centering.cells import (
    cell_name,
    iter_cells,
)
from experiments.init_centering.data import Harmonic
from experiments.init_centering.evals import PilotEval, PilotObjective
from experiments.init_centering.hparams import SmokeHparams, StagesB, Training800
from experiments.init_centering.model import SmokeModel


# ---------------------------------------------------------------------------
# Smoke preset — canonical cell, no objective, ``train()`` returns the
# :class:`DDSSMTrainer` for inspection.  Wires every Phase 1–5 piece:
#
#   - shared baseline between BaselineGaussianTransition (stage 1) and
#     DiffusionV3Transition (stage 2);
#   - AuxPosterior + SigmaDataBuffer slots on DDSSM_base;
#   - StagesConf with a CenteringHandoffConf between stage 1 and stage 2;
#   - Harmonic data at T=32 (matches T_MAX in the model).
#
# Run: ``python -m ddssm.app experiment=init_centering_smoke``.
# ---------------------------------------------------------------------------

init_centering_smoke = experiment(
    data=Harmonic,
    model=SmokeModel(stages=StagesB),
    hparams=SmokeHparams,
    training=Training800,
)
experiment_store(init_centering_smoke, name="init_centering_smoke")


# ---------------------------------------------------------------------------
# Pilot preset — same canonical cell, with the Phase-A eval pipeline +
# the ``stage2_elbo_surrogate`` JSON-source objective wired so that
# ``Experiment.train`` returns a scalar suitable for the Optuna sweep
# in ``sweeps.py``.
#
# Run a single trial: ``python -m ddssm.app experiment=init_centering_pilot``
# Run the sweep:      ``python -m ddssm.app --multirun \
#                          experiment=init_centering_pilot +sweep=init_pilot``
# ---------------------------------------------------------------------------

init_centering_pilot = experiment(
    data=Harmonic,
    model=SmokeModel(stages=StagesB),
    hparams=SmokeHparams,
    training=Training800,
    eval=PilotEval,
    objective=PilotObjective,
)
experiment_store(init_centering_pilot, name="init_centering_pilot")


# ---------------------------------------------------------------------------
# Phase D — the full 18-cell ablation grid.
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
# Submit all 18 cells via SLURM:
#   python -m experiments.init_centering.launch_phase_d --write-dir runs/sbatch/phase_d
# ---------------------------------------------------------------------------

for _form, _mode, _tracking in iter_cells():
    _cell_exp = experiment(
        data=Harmonic,
        model=SmokeModel(
            baseline_form=_form,
            baseline_mode=_mode,
            tracking_mode=_tracking,
            stages=StagesB,
        ),
        hparams=SmokeHparams,
        training=Training800,
        eval=PilotEval,
        objective=PilotObjective,
    )
    experiment_store(_cell_exp, name=cell_name(_form, _mode, _tracking))


# NOTE: ``init_canonical_ctrl_sigma0`` and ``init_canonical_ctrl_npretrain0``
# were removed per docs/adr/0002-drop-canonical-controls.md: σ_pert > 0
# is mandatory protocol (no σ_pert=0 mode) and n_pretrain=0 is meaningless
# for parametric μ_p cells. The sweep range on σ_pert covers
# "operationally indistinguishable from 0" instead.
