"""Component-test experiment presets: synthetic_gauss, synthetic_diffusion.

These are the two smallest presets — a D=1 LGSSM with Gaussian or
Diffusion transition, trained for 500–1000 steps.  They serve as smoke
tests and CI sanity checks.

Registered experiment names
----------------------------
- ``synthetic_gauss``   — lgssm mode, Gaussian transition, 500 steps
- ``synthetic_diffusion`` — lgssm mode, Diffusion transition, 1000 steps
"""

from __future__ import annotations

from .._infra import (
    _experiment_conf,
    DDSSMHyperParamsConf,
    ObjectiveSpecConf,
    SyntheticDataModuleConf,
    TrainingScalarsConf,
    TransitionDiffusionConf,
    TransitionGaussianConf,
    store,
)
from .._eval_viz import SynthEvalConf, SynthVizConf


# ---------------------------------------------------------------------------
# Synthetic + Gaussian transition — small LGSSM run for smoke tests / CI.
# ---------------------------------------------------------------------------

SyntheticGaussExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="lgssm", T=64, N_per_split=512, batch_size=32),
    transition_conf=TransitionGaussianConf,
    hyperparams_conf=DDSSMHyperParamsConf(
        batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
        lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=200,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=1,
    ),
    training_conf=TrainingScalarsConf(steps=500, log_every=25, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=SynthEvalConf,
    viz_conf=SynthVizConf,
    data_dim=1, latent_dim=4, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

store(SyntheticGaussExperimentConf, group="experiment", name="synthetic_gauss")


# ---------------------------------------------------------------------------
# Synthetic + Diffusion transition.
# ---------------------------------------------------------------------------

SyntheticDiffusionExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="lgssm", T=64, N_per_split=512, batch_size=32),
    transition_conf=TransitionDiffusionConf,
    hyperparams_conf=DDSSMHyperParamsConf(
        batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
        lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=300,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=1,
    ),
    training_conf=TrainingScalarsConf(steps=1000, log_every=25, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=SynthEvalConf,
    viz_conf=SynthVizConf,
    data_dim=1, latent_dim=4, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

store(SyntheticDiffusionExperimentConf, group="experiment", name="synthetic_diffusion")
