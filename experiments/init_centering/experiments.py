"""Named smoke experiment for the model-v2 baseline-centering core."""

from __future__ import annotations

from conf.registry import experiment_store
from experiments._make import experiment
from experiments.init_centering.data import Harmonic
from experiments.init_centering.hparams import SmokeHparams, StagesB, Training800
from experiments.init_centering.model import SmokeModel


# The smoke preset wires every Phase 1–5 piece end-to-end:
#   - shared MLPBaseline between BaselineGaussianTransition (stage 1) and
#     DiffusionV3Transition (stage 2);
#   - AuxPosterior + SigmaDataBuffer slots on DDSSM_base;
#   - StagesConf with a CenteringHandoffConf between stage 1 and stage 2;
#   - Harmonic data at T=32 (matches T_MAX in the model).
#
# Run: ``python -m ddssm.app experiment=init_centering_smoke``.
init_centering_smoke = experiment(
    data=Harmonic,
    model=SmokeModel(stages=StagesB),
    hparams=SmokeHparams,
    training=Training800,
)
experiment_store(init_centering_smoke, name="init_centering_smoke")
