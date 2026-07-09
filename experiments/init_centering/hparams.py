"""Hparams + training scalars for the init-centering preset.

Single-phase training: a plain :class:`TrainingScalars` with a cosine
λ ramp on the KL terms. All staged-training / centering-handoff
machinery was retired when the two-phase training path was removed.
"""

from __future__ import annotations

from ddssm.experiment.builders import Hparams, Training

LR = 5e-4


SmokeHparams = Hparams(
    S=1,
    batch_size=16,
    grad_accum_steps=1,
    ema_decay=0.997,
    enc_lr=LR,
    dec_lr=LR,
    trans_lr=LR,
)


# Single-phase training budget for the smoke presets.
Training800 = Training(steps=800, log_every=25, checkpoint_every=200, amp=True)


__all__ = [
    "SmokeHparams",
    "Training800",
]
