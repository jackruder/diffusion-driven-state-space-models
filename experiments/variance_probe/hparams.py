"""Hparams + training scalars for variance-probe experiments.

``checkpoint_every`` is set tight (25 steps → 12 checkpoints) so the
``ddssm.variance ... +per_step=true`` sweep can probe the entire
training trajectory and animate how the variance landscape evolves.
"""

from __future__ import annotations

from ddssm.builders import Hparams, Training

from conf.registry import hparams_store, training_store


LR = 5e-4
LAMBDA_WARMUP = 50


Probe = Hparams(
    S=1, batch_size=32, grad_accum_steps=1,
    lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
    lambda_warmup_steps=LAMBDA_WARMUP,
    enc_lr=LR, dec_lr=LR, zinit_lr=LR, trans_lr=LR,
)


Probe300 = Training(steps=300, log_every=20, checkpoint_every=25, amp=True)


hparams_store(Probe, name="probe")
training_store(Probe300, name="probe_300")
