"""Lean Optuna sweep for the gluonts_forecast family.

Tunes ONLY: ``latent_dim`` (model), ``base_lr``/``dec_mult``/``trans_mult``
(stage builder), ``batch_size`` (hparams). Everything else — budgets, σ_pert,
λ-warmup, all diffusion knobs — is FIXED. Single objective: minimise the
validation ELBO (read from ``metrics.csv``; no per-trial forecast sampling).

``latent_dim`` is categorical over powers-of-two-ish values that are all ÷4, so
``2×latent`` (the summary/encoder/decoder width) stays ÷8 for the 8-head
transformer summary.
"""

from __future__ import annotations

from hydra_zen import make_config

from experiments._sweep import SweepSpace
from ddssm.experiment.stores import sweep_store
from ddssm.experiment.builders import Hparams
from experiments.gluonts_forecast.model import GluonModel
from experiments.gluonts_forecast.hparams import GluonStages

# Each SweepSpace validates its fields against the matching config at import
# time, so a typo / renamed factory arg fails on ``python -m experiments list``
# rather than after launch.
_model = SweepSpace(target=GluonModel, prefix="experiment.model")
_model.raw("latent_dim", "choice(16, 32, 64, 128, 256, 512)")

_stages = SweepSpace(target=GluonStages, prefix="experiment.training.stages")
_stages.log("base_lr", 1e-4, 2e-3)   # encoder LR; dec/trans via multipliers
_stages.log("dec_mult", 0.3, 3.0)
_stages.log("trans_mult", 0.5, 3.0)

_hparams = SweepSpace(target=Hparams, prefix="experiment.hparams")
_hparams.raw("batch_size", "choice(32, 64, 128)")

_params = {**_model.params(), **_stages.params(), **_hparams.params()}

GluonLeanSweep = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(sweeper=dict(direction="minimize", params=_params)),
)
sweep_store(GluonLeanSweep, name="gluonts_lean")


# Reduced-scope validation sweep for a memory-limited box (e.g. the 24 GB dev
# GPU): latent ≤256 + batch ≤64 fit ~15 GB at time_chunk=16, vs the full
# gluonts_lean (latent→512, ~34 GB) which needs the 80 GB A100s. Same LR ranges,
# same val-ELBO objective — used for the local proxy-validation pilot. (Sweeper
# params can't be capped on the CLI: a `choice(...)` override is read as sweeping
# a `hydra.*` key, which Hydra rejects — so the scope lives in this preset.)
_pilot_model = SweepSpace(target=GluonModel, prefix="experiment.model")
_pilot_model.raw("latent_dim", "choice(16, 32, 64, 128, 256)")
_pilot_stages = SweepSpace(target=GluonStages, prefix="experiment.training.stages")
_pilot_stages.log("base_lr", 1e-4, 2e-3)
_pilot_stages.log("dec_mult", 0.3, 3.0)
_pilot_stages.log("trans_mult", 0.5, 3.0)
_pilot_hparams = SweepSpace(target=Hparams, prefix="experiment.hparams")
_pilot_hparams.raw("batch_size", "choice(32, 64)")
_pilot_params = {
    **_pilot_model.params(), **_pilot_stages.params(), **_pilot_hparams.params()
}
GluonPilotSweep = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(sweeper=dict(direction="minimize", params=_pilot_params)),
)
sweep_store(GluonPilotSweep, name="gluonts_pilot")

__all__ = ["GluonLeanSweep", "GluonPilotSweep"]
