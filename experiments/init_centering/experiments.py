"""Named init-centering experiments: smoke + pilot."""

from __future__ import annotations

from conf.registry import experiment_store
from experiments._make import experiment
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
