"""Named experiments for the synthetic-data family.

Each ``<name>`` below composes a registered model + dataset + hparams
+ training/eval/viz from this subpackage and registers the result to
``experiment_store`` under that name. Reachable as
``python -m ddssm.app experiment=<name>`` or via
``from experiments.synthetic.experiments import <name>``.
"""

from __future__ import annotations

import dataclasses

from conf.registry import experiment_store

from experiments._make import experiment
from experiments.synthetic.datasets import (
    Bimodal, Harmonic, LGSSM, Robot2D,
)
from experiments.synthetic.evals import (
    BimodalEnergy, Forecast1D as EvalForecast1D, LGSSM as EvalLGSSM,
    Robot2D as EvalRobot2D,
)
from experiments.synthetic.hparams import Base1D, Bimodal as HparamsBimodal
from experiments.synthetic.models import (
    Robot2DDiff, Robot2DGauss, SmallDiff, SmallGauss,
)
from experiments.synthetic.training import (
    Diff1k, Diff2k, Gauss1k, RobotDiff, RobotGauss, Smoke500,
)
from experiments.synthetic.vizs import (
    BimodalForecast1D, Forecast1D as VizForecast1D, LGSSM as VizLGSSM,
    Robot2DForecast,
)


# ---------------------------------------------------------------------------
# LGSSM smoke tests.
# ---------------------------------------------------------------------------

synthetic_gauss = experiment(
    data=LGSSM, model=SmallGauss,
    hparams=Base1D,
    training=Smoke500,
    eval=EvalLGSSM, viz=VizLGSSM,
)
experiment_store(synthetic_gauss, name="synthetic_gauss")

synthetic_diffusion = experiment(
    data=LGSSM, model=SmallDiff,
    hparams=dataclasses.replace(Base1D, lambda_warmup_steps=300),
    training=Diff1k,
    eval=EvalLGSSM, viz=VizLGSSM,
)
experiment_store(synthetic_diffusion, name="synthetic_diffusion")


# ---------------------------------------------------------------------------
# Harmonic (1D clean sine).
# ---------------------------------------------------------------------------

harmonic_gauss = experiment(
    data=Harmonic, model=SmallGauss,
    hparams=Base1D,
    training=Gauss1k,
    eval=EvalForecast1D, viz=VizForecast1D,
)
experiment_store(harmonic_gauss, name="harmonic_gauss")

harmonic_diffusion = experiment(
    data=Harmonic, model=SmallDiff,
    hparams=dataclasses.replace(Base1D, lambda_warmup_steps=400),
    training=Diff2k,
    eval=EvalForecast1D, viz=VizForecast1D,
)
experiment_store(harmonic_diffusion, name="harmonic_diffusion")


# ---------------------------------------------------------------------------
# Bimodal (1D, S=4, energy-score).
# ---------------------------------------------------------------------------

bimodal_gauss = experiment(
    data=Bimodal, model=SmallGauss,
    hparams=HparamsBimodal,
    training=Gauss1k,
    eval=BimodalEnergy, viz=BimodalForecast1D,
)
experiment_store(bimodal_gauss, name="bimodal_gauss")

bimodal_diffusion = experiment(
    data=Bimodal, model=SmallDiff,
    hparams=dataclasses.replace(HparamsBimodal, lambda_warmup_steps=400),
    training=Diff2k,
    eval=BimodalEnergy, viz=BimodalForecast1D,
)
experiment_store(bimodal_diffusion, name="bimodal_diffusion")


# ---------------------------------------------------------------------------
# Robot navigation 2D (D=2, j=2).
# ---------------------------------------------------------------------------

robot_2d_gauss = experiment(
    data=Robot2D, model=Robot2DGauss,
    hparams=dataclasses.replace(Base1D, lambda_warmup_steps=400),
    training=RobotGauss,
    eval=EvalRobot2D, viz=Robot2DForecast,
)
experiment_store(robot_2d_gauss, name="robot_2d_gauss")

robot_2d_diffusion = experiment(
    data=Robot2D, model=Robot2DDiff,
    hparams=dataclasses.replace(Base1D, lambda_warmup_steps=800),
    training=RobotDiff,
    eval=EvalRobot2D, viz=Robot2DForecast,
)
experiment_store(robot_2d_diffusion, name="robot_2d_diffusion")
