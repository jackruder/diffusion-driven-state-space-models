"""Hparams + training scalars + multi-stage config for the smoke preset."""

from __future__ import annotations

from hydra_zen import builds

from ddssm.builders import CenteringHandoff, Hparams, Training
from ddssm.stages import (
    StageLrsConf,
    StageSpecConf,
    StagesConf,
    StageTrainableConf,
)


LR = 5e-4
LAMBDA_WARMUP = 50  # short: smoke runs are ~800 total steps


SmokeHparams = Hparams(
    S=1,
    batch_size=16,
    grad_accum_steps=1,
    lambda_schedule="cosine",
    lambda_start=0.001,
    lambda_end=1.0,
    lambda_warmup_steps=LAMBDA_WARMUP,
    enc_lr=LR,
    dec_lr=LR,
    zinit_lr=LR,
    trans_lr=LR,
    lambda_sigma_p=1e-2,  # stage-1 log-variance anchor (per model-v2.org
    # § State-conditional prior variance; suggested 1e-2 starting point).
)


# Single-fit fallback (used by Experiment.train *only* if model.config.stages
# is None — the smoke preset configures stages so this is informational only).
Training800 = Training(steps=800, log_every=25, checkpoint_every=200, amp=False)


# Two-stage config: stage 1 trains encoder/decoder/baseline (and aux_posterior
# through encoder=True) with the Gaussian transition; stage 2 swaps to V3 with
# the centering handoff.
StagesB = builds(
    StagesConf,
    populate_full_signature=True,
    stage_1=builds(
        StageSpecConf,
        populate_full_signature=True,
        steps=200,
        trainable=builds(
            StageTrainableConf,
            populate_full_signature=True,
            encoder=True,
            decoder=True,
            z_init=False,  # legacy path off
            transition=True,  # stage1_transition's μ_p/σ_p heads train
        ),
        lrs=builds(StageLrsConf, populate_full_signature=True, enc_lr=LR),
        log_every=25,
        val_every=0,
        checkpoint_every=200,
    ),
    stage_2=builds(
        StageSpecConf,
        populate_full_signature=True,
        steps=600,
        trainable=builds(
            StageTrainableConf,
            populate_full_signature=True,
            encoder=True,
            decoder=True,
            z_init=False,
            transition=True,  # stage-2 transition (V3) trains
        ),
        lrs=builds(StageLrsConf, populate_full_signature=True, enc_lr=LR),
        log_every=25,
        val_every=0,
        checkpoint_every=200,
        centering_handoff=CenteringHandoff(sigma_pert=1e-2),
    ),
    run=["stage_1", "stage_2"],
)


__all__ = ["SmokeHparams", "StagesB", "Training800"]
