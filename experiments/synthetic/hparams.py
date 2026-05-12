"""Hparams variants used by the synthetic-data experiments.

The lambda warmup differs across (transition × dataset budget); the
per-experiment ``override(...)`` calls below adjust the warmup step
count where the named preset is not exactly right. Hparams are also
used as the model's ``hyperparams`` field — :func:`experiment` keeps
them in sync.
"""

from __future__ import annotations

from ddssm.builders import Hparams

from conf.registry import hparams_store


# S=1 (point forecasts: harmonic, robot, lgssm).
Base1D = Hparams(
    S=1, batch_size=32, grad_accum_steps=1,
    lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
    lambda_warmup_steps=200,
    enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
)

# S=4 (energy-score: bimodal series).
Bimodal = Hparams(
    S=4, batch_size=32, grad_accum_steps=1,
    lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
    lambda_warmup_steps=200,
    enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
)

hparams_store(Base1D, name="base_1d")
hparams_store(Bimodal, name="bimodal")
