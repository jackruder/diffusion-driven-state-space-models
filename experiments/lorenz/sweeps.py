"""Optuna sweep presets for the Lorenz attractor family.

Seven sweep axes, informed by EDA findings and init_centering round-2 analysis:

* ``base_lr``           — log-uniform [1e-4, 1e-3].  Floor raised from init_centering
                          round-1 bottom (1e-5) — the bottom 1.5 decades were dead
                          (|rho|=0.76 dominant signal in round-1).
* ``dec_mult``          — log-uniform [0.3, 10.0].  init_centering: bottom [0.1, 0.3]
                          never in best-25% → floor raised.
* ``trans_mult``        — log-uniform [0.5, 10.0].  init_centering: bottom [0.1, 0.5]
                          weak → floor raised.
* ``stage2_trans_lr``   — log-uniform [1e-4, 8e-4].  Lorenz-specific axis:
                          default 5e-4 caused reconstruction spikes; 1e-4 was stable
                          but underfit. Sweep finds the sweet spot.
* ``n_pretrain``        — log-int [200, 2000].  Broader than init_centering (5–500)
                          because Lorenz dynamics are slower to saturate; the
                          Gaussian plateau appears around 800 steps.
* ``n_stage2``          — log-int [1000, 8000].  Tests whether the stage-2 plateau
                          (~2000 steps in lorenz_smoke) is genuine stalling or just
                          a slow descent that needs more budget.
* ``stage_2_warmup_frac`` — log-uniform [0.02, 0.25].  Controls how quickly λ
                          ramps to 1.0 in stage 2; faster ramp risks a
                          loss-form shock, slower ramp delays score-net training.

Run (single-objective)::

    python -m ddssm.app --multirun \\
        experiment=lorenz_4d_open_enc +sweep=lorenz_ablation \\
        hydra.sweeper.n_trials=40 \\
        hydra.sweeper.study_name=lorenz_$(date +%s)

Run (MOO, recommended — gives Pareto front)::

    python -m ddssm.app --multirun \\
        experiment=lorenz_4d_open_enc +sweep=lorenz_ablation_moo \\
        hydra.sweeper.n_trials=64 \\
        hydra.sweeper.study_name=lorenz_moo_$(date +%s)
"""

from __future__ import annotations

from experiments._sweep import SweepSpace
from ddssm.experiment.stores import sweep_store
from experiments.lorenz.evals import LorenzMOObjective
from experiments.lorenz.hparams import LorenzStagesSweep

# Field names validated against LorenzStagesSweep at import time — a typo
# raises here rather than crashing every trial after launch.
_lorenz = SweepSpace(target=LorenzStagesSweep, prefix="experiment.training.stages")
_lorenz.log("base_lr", 1e-4, 1e-3)
_lorenz.log("dec_mult", 0.3, 10.0)
_lorenz.log("trans_mult", 0.5, 10.0)
_lorenz.log("stage2_trans_lr", 1e-4, 8e-4)
_lorenz.log_int("n_pretrain", 200, 2000)
_lorenz.log_int("n_stage2", 1000, 8000)
_lorenz.log("stage_2_warmup_frac", 0.02, 0.25)


LorenzAblation = _lorenz.build(sweeper="ddssm_optuna", direction="minimize")
sweep_store(LorenzAblation, name="lorenz_ablation")


LorenzAblationMOO = _lorenz.build(
    sweeper="ddssm_optuna_moo",
    direction=["minimize", "minimize"],
    objectives=LorenzMOObjective,
)
sweep_store(LorenzAblationMOO, name="lorenz_ablation_moo")


__all__ = ["LorenzAblation", "LorenzAblationMOO"]
