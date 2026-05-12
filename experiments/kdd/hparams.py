"""Hparams for the KDD experiments (linear schedule, batch=128)."""

from __future__ import annotations

from ddssm.builders import Hparams

from conf.registry import hparams_store


KDD = Hparams(
    S=1, batch_size=128, grad_accum_steps=1,
    lambda_schedule="linear", lambda_start=0.001, lambda_end=1.0,
    lambda_warmup_steps=500,
    enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
)

hparams_store(KDD, name="kdd")
