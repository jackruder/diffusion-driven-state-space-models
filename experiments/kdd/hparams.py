"""Hparams + training scalars for the KDD experiments.

Linear λ schedule (KDD trains longer than synthetic so the shorter
exponential warmup is enough). AMP is on for both training presets —
KDD's full CSDI U-Net plus the 128-batch makes fp16 worth the
arithmetic.
"""

from __future__ import annotations

from ddssm.builders import Hparams, Training

from conf.registry import hparams_store, training_store


LR = 5e-4
BATCH_SIZE = 128
LAMBDA_WARMUP = 500


KDDHparams = Hparams(
    S=1, batch_size=BATCH_SIZE, grad_accum_steps=1,
    lambda_schedule="linear", lambda_start=0.001, lambda_end=1.0,
    lambda_warmup_steps=LAMBDA_WARMUP,
    enc_lr=LR, dec_lr=LR, zinit_lr=LR, trans_lr=LR,
)


Gauss5k = Training(steps=5000, log_every=50, checkpoint_every=500, amp=True)
Diff8k = Training(steps=8000, log_every=50, checkpoint_every=500, amp=True)


hparams_store(KDDHparams, name="kdd")
training_store(Gauss5k, name="kdd_gauss_5k")
training_store(Diff8k, name="kdd_diff_8k")
