"""Hparams + training scalars for the synthetic-data family.

Two axes:

* **Hparams** — batch size, S, λ schedule, LRs. Used as both the
  trainer's hyperparam source and the model's ``hyperparams`` field.
* **Training scalars** — step budget, log/checkpoint cadence, AMP.

Per-experiment ``override(...)`` calls in :mod:`experiments.synthetic.experiments`
adjust λ-warmup where the named preset is not exactly right.
"""

from __future__ import annotations

from ddssm.builders import Hparams, Training

from conf.registry import hparams_store, training_store


# ---------------------------------------------------------------------------
# Hparams. Shared LRs at the top so tweaking one number propagates.
# ---------------------------------------------------------------------------

LR = 5e-4
LAMBDA_SCHEDULE = "cosine"
LAMBDA_START = 0.001
LAMBDA_END = 1.0
LAMBDA_WARMUP = 200


# S=1 (point forecasts: harmonic, robot, lgssm).
Base1D = Hparams(
    S=1, batch_size=32, grad_accum_steps=1,
    lambda_schedule=LAMBDA_SCHEDULE,
    lambda_start=LAMBDA_START, lambda_end=LAMBDA_END,
    lambda_warmup_steps=LAMBDA_WARMUP,
    enc_lr=LR, dec_lr=LR, zinit_lr=LR, trans_lr=LR,
)

# S=4 (energy-score: bimodal series).
Bimodal = Hparams(
    S=4, batch_size=32, grad_accum_steps=1,
    lambda_schedule=LAMBDA_SCHEDULE,
    lambda_start=LAMBDA_START, lambda_end=LAMBDA_END,
    lambda_warmup_steps=LAMBDA_WARMUP,
    enc_lr=LR, dec_lr=LR, zinit_lr=LR, trans_lr=LR,
)


# ---------------------------------------------------------------------------
# Training scalars.
# ---------------------------------------------------------------------------

# 500-step smoke (LGSSM).
Smoke500 = Training(steps=500, log_every=25, amp=True)
# 1000-step Gaussian harmonic / bimodal default.
Gauss1k = Training(steps=1000, log_every=25, checkpoint_every=200, amp=True)
# 1000-step diffusion smoke (LGSSM diffusion).
Diff1k = Training(steps=1000, log_every=25, amp=True)
# 2000-step diffusion harmonic / bimodal / robot2d_gauss default.
Diff2k = Training(steps=2000, log_every=25, checkpoint_every=500, amp=True)
# 2000-step robot 2D Gaussian (log_every=50).
RobotGauss = Training(steps=2000, log_every=50, checkpoint_every=500, amp=True)
# 4000-step robot 2D diffusion.
RobotDiff = Training(steps=4000, log_every=50, checkpoint_every=500, amp=True)


hparams_store(Base1D, name="base_1d")
hparams_store(Bimodal, name="bimodal")

training_store(Smoke500, name="smoke_500")
training_store(Gauss1k, name="gauss_1k")
training_store(Diff1k, name="diff_1k")
training_store(Diff2k, name="diff_2k")
training_store(RobotGauss, name="robot_gauss_2k")
training_store(RobotDiff, name="robot_diff_4k")
