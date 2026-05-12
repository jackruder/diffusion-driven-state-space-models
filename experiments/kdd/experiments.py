"""Named KDD Cup 2018 PM2.5 experiments."""

from __future__ import annotations

from conf.registry import experiment_store

from experiments._make import experiment
from experiments.kdd.datasets import KDDData
from experiments.kdd.evals import KDD as EvalKDD
from experiments.kdd.hparams import KDD as HparamsKDD
from experiments.kdd.models import KDDDiffModel, KDDGaussModel
from experiments.kdd.training import Diff8k, Gauss5k
from experiments.kdd.vizs import KDD as VizKDD


kdd_gauss = experiment(
    data=KDDData, model=KDDGaussModel,
    hparams=HparamsKDD,
    training=Gauss5k,
    eval=EvalKDD, viz=VizKDD,
)
experiment_store(kdd_gauss, name="kdd_gauss")

kdd_diffusion = experiment(
    data=KDDData, model=KDDDiffModel,
    hparams=HparamsKDD,
    training=Diff8k,
    eval=EvalKDD, viz=VizKDD,
)
experiment_store(kdd_diffusion, name="kdd_diffusion")
