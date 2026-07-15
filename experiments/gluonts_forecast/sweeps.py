"""Lean Optuna sweep for the gluonts_forecast family.

Tunes ``latent_dim`` (model), ``enc_lr``/``dec_lr``/``trans_lr`` +
``batch_size`` (hparams). Everything else is FIXED. Single objective:
minimise the validation ELBO (read from ``metrics.csv``; no per-trial
forecast sampling).
"""

from __future__ import annotations

from hydra_zen import make_config

from experiments._sweep import SweepSpace
from ddssm.experiment.stores import sweep_store
from ddssm.experiment.builders import Hparams
from experiments.gluonts_forecast.model import GluonModel

_model = SweepSpace(target=GluonModel, prefix="experiment.model.module")
_model.raw("latent_dim", "choice(16, 32, 64, 128, 256, 512)")

_hparams = SweepSpace(target=Hparams, prefix="experiment.hparams")
_hparams.log("enc_lr", 1e-4, 2e-3)
_hparams.log("dec_lr", 1e-4, 2e-3)
_hparams.log("trans_lr", 1e-4, 2e-3)
_hparams.raw("batch_size", "choice(32, 64, 128)")

_params = {**_model.params(), **_hparams.params()}

GluonLeanSweep = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(sweeper=dict(direction="minimize", params=_params)),
)
sweep_store(GluonLeanSweep, name="gluonts_lean")


_pilot_model = SweepSpace(target=GluonModel, prefix="experiment.model.module")
_pilot_model.raw("latent_dim", "choice(16, 32, 64, 128, 256)")
_pilot_hparams = SweepSpace(target=Hparams, prefix="experiment.hparams")
_pilot_hparams.log("enc_lr", 1e-4, 2e-3)
_pilot_hparams.log("dec_lr", 1e-4, 2e-3)
_pilot_hparams.log("trans_lr", 1e-4, 2e-3)
_pilot_hparams.raw("batch_size", "choice(32, 64)")
_pilot_params = {**_pilot_model.params(), **_pilot_hparams.params()}
GluonPilotSweep = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(sweeper=dict(direction="minimize", params=_pilot_params)),
)
sweep_store(GluonPilotSweep, name="gluonts_pilot")

__all__ = ["GluonLeanSweep", "GluonPilotSweep"]
