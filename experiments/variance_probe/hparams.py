"""Hparams variant used by variance-probe experiments (short warmup)."""

from __future__ import annotations

from ddssm.builders import Hparams

from conf.registry import hparams_store


Probe = Hparams(
    S=1, batch_size=32, grad_accum_steps=1,
    lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
    lambda_warmup_steps=50,
    enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
)

hparams_store(Probe, name="probe")
