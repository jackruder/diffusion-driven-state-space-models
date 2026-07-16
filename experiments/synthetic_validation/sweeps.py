"""Optuna sweep for the sin_overfit sanity-check preset.

Single-objective minimize recon_mse. Sweeps training hyperparams (LRs)
and two architecture knobs (latent_dim, diffusion_num_steps).

Run::

    .venv/bin/python -m ddssm.app --multirun \\
        experiment=sin_overfit +sweep=sin_overfit_mse \\
        hydra.sweeper.n_trials=30 \\
        hydra.sweeper.study_name=sin_overfit_mse
"""

from __future__ import annotations

from hydra_zen import make_config

from experiments._sweep import SweepSpace
from ddssm.experiment.stores import sweep_store
from ddssm.experiment.builders import Hparams
from experiments.synthetic_validation.model import SynthValModel

_hparams = SweepSpace(target=Hparams, prefix="experiment.model.config.training")
_hparams.log("enc_lr", 1e-4, 3e-3)
_hparams.log("dec_lr", 1e-4, 3e-3)
_hparams.log("trans_lr", 1e-4, 3e-3)

_model = SweepSpace(target=SynthValModel, prefix="experiment.model")
_model.raw("latent_dim", "choice(2, 4, 8)")
_model.raw("diffusion_num_steps", "choice(32, 64, 128)")

_all_params = {**_hparams.params(), **_model.params()}

SinOverfitMSE = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(sweeper=dict(direction="minimize", params=_all_params)),
)
sweep_store(SinOverfitMSE, name="sin_overfit_mse")

__all__ = ["SinOverfitMSE"]
