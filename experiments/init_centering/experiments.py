"""Named init-centering experiments: two role-specific smokes + the ablation study.

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

The ablation grid itself is a first-class :class:`~experiments._study.Study`
(``experiments.init_centering.study.INIT_CENTERING_STUDY``): 12 cells × 2
datasets = 24 registered presets named ``init_<cell>__<dataset>`` (e.g.
``init_mlp_pinned_per_t__1d``). Registration, launching, and reporting all
flow through that Study.
"""

from __future__ import annotations

from conf.registry import experiment_store
from experiments._make import experiment
from experiments.init_centering.data import (
    NonlinBimodalLift1D,
    NonlinBimodalLiftMV,
)
from experiments.init_centering.evals import (
    PilotEval,
    PilotObjective,
)
from experiments.init_centering.model import SmokeModel
from experiments.init_centering.hparams import StagesB, Training800, SmokeHparams
from experiments.init_centering.study import INIT_CENTERING_STUDY

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
# The ablation study — 12 cells × 2 datasets = 24 presets named
# ``init_<cell>__<dataset>`` (e.g. ``init_mlp_pinned_per_t__1d``). Each bakes
# the real ablation dataset + dims and wires the Phase-A eval pipeline + the
# multi-objective (wallclock_to_target, stage2_elbo_surrogate) objective.
#
# Run a single point:
#   python -m ddssm.app experiment=init_mlp_pinned_per_t__1d
# Sweep / launch the whole study:
#   python -m experiments.init_centering.launch_study --mode tiny --write-dir runs/sbatch/tiny
#
# NOTE: controls (``init_canonical_ctrl_*``) were removed per
# docs/adr/0002-drop-canonical-controls.md (σ_pert > 0 is mandatory; n_pretrain=0
# is meaningless for parametric μ_p). The sweep's σ_pert lower bound covers
# "operationally indistinguishable from 0" instead.
# ---------------------------------------------------------------------------

INIT_CENTERING_STUDY.register(experiment_store)
