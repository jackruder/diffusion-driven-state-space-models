"""Optuna search spaces for the encoder head-to-head (two phases).

* ``h2h_lr_only`` (Phase 1, capacity probe): a clean LR grid on Hparams.
* ``h2h_full`` (Phase 2, full model): LR + dec/trans LR ratios.
"""

from __future__ import annotations

from experiments._sweep import SweepSpace
from ddssm.experiment.stores import sweep_store
from ddssm.experiment.builders import Hparams

_lr = SweepSpace(target=Hparams, prefix="experiment.hparams")
_lr.raw("enc_lr", "choice(3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3)")
H2HLrOnlySweep = _lr.build(sweeper="ddssm_optuna", direction="minimize")
sweep_store(H2HLrOnlySweep, name="h2h_lr_only")

_full = SweepSpace(target=Hparams, prefix="experiment.hparams")
_full.log("enc_lr", 1e-3, 8e-3)
_full.log("dec_lr", 3e-4, 2e-2)
_full.log("trans_lr", 5e-4, 2e-2)
H2HFullSweep = _full.build(sweeper="ddssm_optuna", direction="minimize")
sweep_store(H2HFullSweep, name="h2h_full")

__all__ = ["H2HLrOnlySweep", "H2HFullSweep"]
