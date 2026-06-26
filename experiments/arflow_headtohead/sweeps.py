"""Optuna search spaces for the encoder head-to-head (two phases).

* ``h2h_lr_only`` (Phase 1, capacity probe): a clean base_lr grid only. Minimise
  val ``recon_mse`` (μ_x; the λ=0 distortion NLL is degenerate). Run against a
  ``h2h_cap__<enc>__<ds>`` cell.
* ``h2h_full`` (Phase 2, full model): the complete stage-hyperparameter space.
  Minimise held-out **val forecast CRPS-sum** (the cell's json-source objective).
  Run against a ``h2h__<enc>__<ds>`` cell.

Every field is validated against ``GluonStages`` at import (``SweepSpace``), so a
renamed factory arg fails on ``python -m experiments list`` rather than per-trial
on the cluster. λ-warmup fracs are swept in both stages (their absence drives
marginal-collapse underfit).
"""

from __future__ import annotations

from experiments._sweep import SweepSpace
from ddssm.experiment.stores import sweep_store
from experiments.gluonts_forecast.hparams import GluonStages

# --- Phase 1: capacity probe — base_lr only (a clean discrete grid). ---
_lr = SweepSpace(target=GluonStages, prefix="experiment.training.stages")
_lr.raw("base_lr", "choice(3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3)")
H2HLrOnlySweep = _lr.build(sweeper="ddssm_optuna", direction="minimize")
sweep_store(H2HLrOnlySweep, name="h2h_lr_only")

# --- Phase 2: full two-stage ELBO space (minimise val CRPS-sum). ---
_full = SweepSpace(target=GluonStages, prefix="experiment.training.stages")
_full.log("base_lr", 1e-3, 8e-3)
_full.log("dec_mult", 0.3, 3.0)
_full.log("trans_mult", 0.5, 3.0)
_full.log_int("n_pretrain", 250, 900)          # low n_pretrain collapsed → floor at 250
_full.log("sigma_pert", 1e-3, 1e-1)            # handoff σ_pert>0 is mandatory protocol
_full.uniform("stage_1_warmup_frac", 0.05, 0.6)
_full.uniform("stage_2_warmup_frac", 0.05, 0.6)  # the λ knob that trains the diffusion
_full.log("stage_1_lambda_start", 1e-3, 1e-1)
_full.log("stage_2_lambda_start", 1e-3, 1e-1)
_full.log("lambda_sigma_p", 1e-3, 1e-1)
H2HFullSweep = _full.build(sweeper="ddssm_optuna", direction="minimize")
sweep_store(H2HFullSweep, name="h2h_full")

__all__ = ["H2HLrOnlySweep", "H2HFullSweep"]
