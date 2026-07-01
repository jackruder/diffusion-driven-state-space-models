"""Optuna sweep for the sin_overfit sanity-check preset.

Stage-2-only, ZeroBaseline, single-objective minimize recon_mse.
Sweeps training hyperparams (LR, λ schedule, step budget) and two
architecture knobs (latent_dim, diffusion_num_steps).

Run::

    .venv/bin/python -m ddssm.app --multirun \
        experiment=sin_overfit +sweep=sin_overfit_mse \
        hydra.sweeper.n_trials=30 \
        hydra.sweeper.study_name=sin_overfit_mse
"""

from __future__ import annotations

from hydra_zen import make_config

from experiments._sweep import SweepSpace
from ddssm.experiment.stores import sweep_store
from experiments.init_centering.hparams import StagesB
from experiments.synthetic_validation.model import SynthValModel

_stages = SweepSpace(target=StagesB, prefix="experiment.training.stages")
_stages.log("base_lr", 1e-4, 3e-3)
_stages.log_int("n_stage2", 2000, 8000)
_stages.uniform("stage_2_warmup_frac", 0.05, 0.30)
_stages.log("stage_2_lambda_start", 0.001, 0.1)
_stages.log("stage_2_lambda_end", 0.1, 5.0)
_stages.log("dec_mult", 0.3, 3.0)
_stages.log("trans_mult", 0.3, 3.0)

_model = SweepSpace(target=SynthValModel, prefix="experiment.model")
_model.raw("latent_dim", "choice(2, 4, 8)")
_model.raw("diffusion_num_steps", "choice(32, 64, 128)")

_all_params = {**_stages.params(), **_model.params()}

SinOverfitMSE = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(sweeper=dict(direction="minimize", params=_all_params)),
)
sweep_store(SinOverfitMSE, name="sin_overfit_mse")

__all__ = ["SinOverfitMSE"]
