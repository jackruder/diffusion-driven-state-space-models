"""KDD Cup 2018 experiment presets: kdd_gauss, kdd_diffusion.

PM2.5 air-quality data from Beijing.  Past 72 hours → forecast 48 hours.
D=6 aligned features, latent_dim=8, 3 covariates (time-of-day, etc.).

Registered experiment names
----------------------------
- ``kdd_gauss``      — Gaussian transition, 5000 steps, AMP enabled
- ``kdd_diffusion``  — Diffusion transition, 8000 steps, AMP enabled
"""

from __future__ import annotations

from .._infra import (
    KDDDataModuleConf,
    ObjectiveSpecConf,
    TrainingScalarsConf,
    DDSSMHyperParamsConf,
    TransitionGaussianConf,
    TransitionDiffusionConf,
    store,
    _experiment_conf,
)
from .._eval_viz import KDDVizConf, KDDEvalConf

# ---------------------------------------------------------------------------
# KDD + Gaussian transition (real data via data/kdd.pt).
# ---------------------------------------------------------------------------

KDDGaussExperimentConf = _experiment_conf(
    data_conf=KDDDataModuleConf(batch_size=128, eval_step_size=24),
    transition_conf=TransitionGaussianConf,
    hyperparams_conf=DDSSMHyperParamsConf(
        batch_size=128,
        grad_accum_steps=1,
        lambda_schedule="linear",
        lambda_start=0.001,
        lambda_end=1.0,
        lambda_warmup_steps=500,
        enc_lr=5e-4,
        dec_lr=5e-4,
        zinit_lr=5e-4,
        trans_lr=5e-4,
        S=1,
    ),
    training_conf=TrainingScalarsConf(
        steps=5000, log_every=50, checkpoint_every=500, amp=True
    ),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=KDDEvalConf,
    viz_conf=KDDVizConf,
    data_dim=6,
    latent_dim=8,
    emb_time_dim=32,
    covariate_dim=3,
    use_observation_mask=False,
)

store(KDDGaussExperimentConf, group="experiment", name="kdd_gauss")


# ---------------------------------------------------------------------------
# KDD + Diffusion transition.
# ---------------------------------------------------------------------------

KDDDiffusionExperimentConf = _experiment_conf(
    data_conf=KDDDataModuleConf(batch_size=64, eval_step_size=24),
    transition_conf=TransitionDiffusionConf,
    hyperparams_conf=DDSSMHyperParamsConf(
        batch_size=64,
        grad_accum_steps=1,
        lambda_schedule="linear",
        lambda_start=0.001,
        lambda_end=1.0,
        lambda_warmup_steps=500,
        enc_lr=5e-4,
        dec_lr=5e-4,
        zinit_lr=5e-4,
        trans_lr=5e-4,
        S=1,
    ),
    training_conf=TrainingScalarsConf(
        steps=8000, log_every=50, checkpoint_every=500, amp=True
    ),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=KDDEvalConf,
    viz_conf=KDDVizConf,
    data_dim=6,
    latent_dim=8,
    emb_time_dim=32,
    covariate_dim=3,
    use_observation_mask=False,
)

store(KDDDiffusionExperimentConf, group="experiment", name="kdd_diffusion")
