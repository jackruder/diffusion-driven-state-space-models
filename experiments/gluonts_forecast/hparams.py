"""Hparams + training scalars for the gluonts_forecast family.

Single-phase training keyed on ``training.steps``. Persistence baseline
is parameter-free (``σ_p² = 1``); the diffusion transition consumes it
by reference.
"""

from __future__ import annotations

from ddssm.experiment.builders import Hparams, Training

BASE_LR = 5e-4


GluonHparams = Hparams(
    batch_size=64,
    grad_accum_steps=1,
    ema_decay=0.999,
    enc_lr=BASE_LR,
    dec_lr=BASE_LR,
    trans_lr=BASE_LR,
)

# validate_every MUST be > 0: Experiment.train only builds the val_loader when
# ``training.validate_every > 0`` (experiment.py), and the val ELBO is the sweep
# objective.
GluonTraining = Training(
    steps=20000,
    log_every=100,
    validate_every=500,
    checkpoint_every=2000,
    amp=True,
)


__all__ = [
    "GluonHparams",
    "GluonTraining",
]
