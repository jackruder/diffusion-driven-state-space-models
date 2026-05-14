"""Named experiments + Optuna sweep presets for the synthetic family.

Each ``<name>`` below composes a registered model + dataset + hparams
+ training/eval/viz from this subpackage and registers the result to
``experiment_store``. Reachable as
``python -m ddssm.app experiment=<name>`` or
``python -m experiments run <name>``.

Sweeps live at the bottom of this file (one section) so a search
preset sits next to the experiments it pairs with.
"""

from __future__ import annotations

import dataclasses

from hydra_zen import make_config

from conf.registry import experiment_store, sweep_store

from experiments._make import experiment
from experiments.synthetic.data import (
    Bimodal, Harmonic, LGSSM, Robot2D,
)
from experiments.synthetic.evals import (
    BimodalEnergy_Eval, BimodalForecast1D_Viz,
    Forecast1D_Eval, Forecast1D_Viz,
    LGSSM_Eval, LGSSM_Viz,
    Robot2D_Eval, Robot2D_Viz,
)
from experiments.synthetic.hparams import (
    Base1D, Bimodal as HparamsBimodal,
    Diff1k, Diff2k, Gauss1k, RobotDiff, RobotGauss, Smoke500,
)
from experiments.synthetic.model import Robot2D as Robot2DShape, Small1D


# ---------------------------------------------------------------------------
# LGSSM smoke tests.
# ---------------------------------------------------------------------------

synthetic_gauss = experiment(
    data=LGSSM, model=Small1D.gauss_model,
    hparams=Base1D,
    training=Smoke500,
    eval=LGSSM_Eval, viz=LGSSM_Viz,
)
experiment_store(synthetic_gauss, name="synthetic_gauss")

synthetic_diffusion = experiment(
    data=LGSSM, model=Small1D.diff_model,
    hparams=dataclasses.replace(Base1D, lambda_warmup_steps=300),
    training=Diff1k,
    eval=LGSSM_Eval, viz=LGSSM_Viz,
)
experiment_store(synthetic_diffusion, name="synthetic_diffusion")


# ---------------------------------------------------------------------------
# Harmonic (1D clean sine).
# ---------------------------------------------------------------------------

harmonic_gauss = experiment(
    data=Harmonic, model=Small1D.gauss_model,
    hparams=Base1D,
    training=Gauss1k,
    eval=Forecast1D_Eval, viz=Forecast1D_Viz,
)
experiment_store(harmonic_gauss, name="harmonic_gauss")

harmonic_diffusion = experiment(
    data=Harmonic, model=Small1D.diff_model,
    hparams=dataclasses.replace(Base1D, lambda_warmup_steps=400),
    training=Diff2k,
    eval=Forecast1D_Eval, viz=Forecast1D_Viz,
)
experiment_store(harmonic_diffusion, name="harmonic_diffusion")


# ---------------------------------------------------------------------------
# Bimodal (1D, S=4, energy-score).
# ---------------------------------------------------------------------------

bimodal_gauss = experiment(
    data=Bimodal, model=Small1D.gauss_model,
    hparams=HparamsBimodal,
    training=Gauss1k,
    eval=BimodalEnergy_Eval, viz=BimodalForecast1D_Viz,
)
experiment_store(bimodal_gauss, name="bimodal_gauss")

bimodal_diffusion = experiment(
    data=Bimodal, model=Small1D.diff_model,
    hparams=dataclasses.replace(HparamsBimodal, lambda_warmup_steps=400),
    training=Diff2k,
    eval=BimodalEnergy_Eval, viz=BimodalForecast1D_Viz,
)
experiment_store(bimodal_diffusion, name="bimodal_diffusion")


# ---------------------------------------------------------------------------
# Robot navigation 2D (D=2, j=2).
# ---------------------------------------------------------------------------

robot_2d_gauss = experiment(
    data=Robot2D, model=Robot2DShape.gauss_model,
    hparams=dataclasses.replace(Base1D, lambda_warmup_steps=400),
    training=RobotGauss,
    eval=Robot2D_Eval, viz=Robot2D_Viz,
)
experiment_store(robot_2d_gauss, name="robot_2d_gauss")

robot_2d_diffusion = experiment(
    data=Robot2D, model=Robot2DShape.diff_model,
    hparams=dataclasses.replace(Base1D, lambda_warmup_steps=800),
    training=RobotDiff,
    eval=Robot2D_Eval, viz=Robot2D_Viz,
)
experiment_store(robot_2d_diffusion, name="robot_2d_diffusion")


# ---------------------------------------------------------------------------
# Optuna sweep presets.
#
# Registered under ``group="sweep"`` and activated via
# ``+sweep=<name>``. ``package="_global_"`` on ``sweep_store`` merges
# the preset at root, matching the legacy ``# @package _global_`` YAML
# semantic.
#
# Example::
#
#     python -m ddssm.app --multirun \
#         experiment=synthetic_gauss \
#         +sweep=synthetic_lr \
#         hydra.sweeper.n_trials=20
# ---------------------------------------------------------------------------

SyntheticLR = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(
        sweeper=dict(
            direction="minimize",
            params={
                "experiment.hyperparams.enc_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.dec_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.trans_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.zinit_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.lambda_warmup_steps":
                    "range(50, 400, step=50)",
                "experiment.hyperparams.lambda_end": "interval(0.5, 2.0)",
                "experiment.hyperparams.batch_size": "choice(16, 32, 64)",
            },
        ),
    ),
)
sweep_store(SyntheticLR, name="synthetic_lr")
