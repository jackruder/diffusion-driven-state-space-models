"""Optuna search spaces for the encoder head-to-head (two phases).

* ``h2h_lr_only`` (Phase 1, capacity probe): a clean LR grid on Hparams.
* ``h2h_full`` (Phase 2, full model): LR + dec/trans LR ratios.
* ``h2h_arch_lr_lambda`` (post-wideenc): joint sweep over architecture size
  (encoder/score-net capacity), LRs (enc/trans; dec tied to enc), and
  lambda-ramp scheduling. Preset-agnostic — attach with ``+sweep=`` on the
  chosen base ``experiment=``.
"""

from __future__ import annotations

from hydra_zen import make_config

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


# Post-wideenc joint sweep over architecture size + LRs + lambda scheduling.
# dec_lr is TIED to enc_lr (fires as `experiment.hparams.dec_lr=${...enc_lr}`
# on every trial via the sweep's params dict — see below for the raw form).
# 7 dims + 1 tie = effectively 7 tunable knobs.
_arch_lr_lambda_params = {
    # LRs (dec_lr tied via interpolation to enc_lr — see the value below)
    "experiment.hparams.enc_lr": "tag(log, interval(3e-4, 5e-3))",
    "experiment.hparams.trans_lr": "tag(log, interval(3e-4, 5e-3))",
    # Note: this is not a search dim — Optuna leaves it as a fixed override
    # so dec_lr always equals whatever value enc_lr takes for the current
    # trial. Optuna sees only enc_lr / trans_lr as tunable LRs.
    "experiment.hparams.dec_lr": r"${experiment.hparams.enc_lr}",
    # Lambda ramp
    "experiment.hparams.lambda_ramp.start": "tag(log, interval(1e-7, 1e-3))",
    "experiment.hparams.lambda_ramp.steps": "range(2000, 8000)",
    # Architecture size (score-net + encoder width)
    "experiment.model.channels": "choice(48, 64, 96, 128)",
    "experiment.model.diffusion_layers": "choice(3, 4, 5, 6)",
    "experiment.model.encoder_hidden_dim": "choice(32, 64, 96, 128)",
}

H2HArchLrLambdaSweep = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(
        sweeper=dict(direction="minimize", params=_arch_lr_lambda_params),
    ),
)
sweep_store(H2HArchLrLambdaSweep, name="h2h_arch_lr_lambda")


__all__ = [
    "H2HLrOnlySweep",
    "H2HFullSweep",
    "H2HArchLrLambdaSweep",
]
