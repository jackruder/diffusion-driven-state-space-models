"""Synthetic verification experiment presets.

Covers harmonic, harmonic-noisy, bimodal, and robot-navigation families
with both Gaussian and Diffusion transitions.  Each family pair is the
primary vehicle for the verification experiments documented in
``verifications.org``.

Registered experiment names
----------------------------
Harmonic (clean periodic signal):
- ``harmonic_gauss``      — j=1, Gaussian, 1000 steps
- ``harmonic_diff``       — j=1, Diffusion, 2000 steps
- ``harmonic_gauss_j2``   — j=2 (second-order AR latent), Gaussian, 1000 steps
- ``harmonic_diff_j2``    — j=2, Diffusion, 2000 steps

Harmonic-noisy (4× observation noise):
- ``harmonic_noisy_gauss`` — j=1, Gaussian, 1000 steps
- ``harmonic_noisy_diff``  — j=1, Diffusion, 2000 steps

Bimodal (multimodality benchmark, headline metric: energy score):
- ``bimodal_gauss`` — j=1, Gaussian, 1000 steps
- ``bimodal_diff``  — j=1, Diffusion, 2000 steps

Robot navigation 2D (spatial trajectories, D=2, j=2):
- ``robot_gauss_2d`` — Gaussian, 2000 steps
- ``robot_diff_2d``  — Diffusion, 4000 steps
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
from .._eval_viz import (
    BimodalEvalConf,
    BimodalVizConf,
    HarmonicEvalConf,
    HarmonicVizConf,
    Robot2DEvalConf,
    Robot2DVizConf,
)


# ---------------------------------------------------------------------------
# Shared hyperparameter templates (private to this module).
# ---------------------------------------------------------------------------

_HarmonicHyperGauss = DDSSMHyperParamsConf(
    batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
    lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=200,
    enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=1,
)
_HarmonicHyperDiff = DDSSMHyperParamsConf(
    batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
    lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=400,
    enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=1,
)

_BimodalHyperGauss = DDSSMHyperParamsConf(
    batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
    lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=200,
    enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=4,
)
_BimodalHyperDiff = DDSSMHyperParamsConf(
    batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
    lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=400,
    enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=4,
)

_RobotHyperGauss = DDSSMHyperParamsConf(
    batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
    lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=400,
    enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=1,
)
_RobotHyperDiff = DDSSMHyperParamsConf(
    batch_size=32, grad_accum_steps=1, lambda_schedule="cosine",
    lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=800,
    enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=1,
)


# ---------------------------------------------------------------------------
# Harmonic: clean sine-wave signal.
# ---------------------------------------------------------------------------

HarmonicGaussExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="harmonic", T=64, N_per_split=1024, batch_size=32),
    transition_conf=TransitionGaussianConf,
    hyperparams_conf=_HarmonicHyperGauss,
    training_conf=TrainingScalarsConf(steps=1000, log_every=25, checkpoint_every=200, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=HarmonicEvalConf,
    viz_conf=HarmonicVizConf,
    data_dim=1, latent_dim=4, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

HarmonicDiffExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="harmonic", T=64, N_per_split=1024, batch_size=32),
    transition_conf=TransitionDiffusionConf,
    hyperparams_conf=_HarmonicHyperDiff,
    training_conf=TrainingScalarsConf(steps=2000, log_every=25, checkpoint_every=500, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=HarmonicEvalConf,
    viz_conf=HarmonicVizConf,
    data_dim=1, latent_dim=4, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

# j=2: second-order AR latent — the transition sees the last two latent states.
HarmonicGaussJ2ExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="harmonic", T=64, N_per_split=1024, batch_size=32),
    transition_conf=TransitionGaussianConf,
    hyperparams_conf=_HarmonicHyperGauss,
    training_conf=TrainingScalarsConf(steps=1000, log_every=25, checkpoint_every=200, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=HarmonicEvalConf,
    viz_conf=HarmonicVizConf,
    data_dim=1, latent_dim=4, j=2, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

HarmonicDiffJ2ExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="harmonic", T=64, N_per_split=1024, batch_size=32),
    transition_conf=TransitionDiffusionConf,
    hyperparams_conf=_HarmonicHyperDiff,
    training_conf=TrainingScalarsConf(steps=2000, log_every=25, checkpoint_every=500, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=HarmonicEvalConf,
    viz_conf=HarmonicVizConf,
    data_dim=1, latent_dim=4, j=2, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

store(HarmonicGaussExperimentConf, group="experiment", name="harmonic_gauss")
store(HarmonicDiffExperimentConf, group="experiment", name="harmonic_diff")
store(HarmonicGaussJ2ExperimentConf, group="experiment", name="harmonic_gauss_j2")
store(HarmonicDiffJ2ExperimentConf, group="experiment", name="harmonic_diff_j2")


# ---------------------------------------------------------------------------
# Harmonic-noisy: sine wave with higher observation noise (ε ~ N(0, 0.2)).
# ---------------------------------------------------------------------------

HarmonicNoisyGaussExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="harmonic-noisy", T=64, N_per_split=1024, batch_size=32),
    transition_conf=TransitionGaussianConf,
    hyperparams_conf=_HarmonicHyperGauss,
    training_conf=TrainingScalarsConf(steps=1000, log_every=25, checkpoint_every=200, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=HarmonicEvalConf,
    viz_conf=HarmonicVizConf,
    data_dim=1, latent_dim=4, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

HarmonicNoisyDiffExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="harmonic-noisy", T=64, N_per_split=1024, batch_size=32),
    transition_conf=TransitionDiffusionConf,
    hyperparams_conf=_HarmonicHyperDiff,
    training_conf=TrainingScalarsConf(steps=2000, log_every=25, checkpoint_every=500, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=HarmonicEvalConf,
    viz_conf=HarmonicVizConf,
    data_dim=1, latent_dim=4, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

store(HarmonicNoisyGaussExperimentConf, group="experiment", name="harmonic_noisy_gauss")
store(HarmonicNoisyDiffExperimentConf, group="experiment", name="harmonic_noisy_diff")


# ---------------------------------------------------------------------------
# Bimodal: multimodality comparison — energy score is the headline metric.
# ---------------------------------------------------------------------------

BimodalGaussExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="bimodal", T=64, N_per_split=1024, batch_size=32),
    transition_conf=TransitionGaussianConf,
    hyperparams_conf=_BimodalHyperGauss,
    training_conf=TrainingScalarsConf(steps=1000, log_every=25, checkpoint_every=200, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=BimodalEvalConf,
    viz_conf=BimodalVizConf,
    data_dim=1, latent_dim=4, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

BimodalDiffExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="bimodal", T=64, N_per_split=1024, batch_size=32),
    transition_conf=TransitionDiffusionConf,
    hyperparams_conf=_BimodalHyperDiff,
    training_conf=TrainingScalarsConf(steps=2000, log_every=25, checkpoint_every=500, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=BimodalEvalConf,
    viz_conf=BimodalVizConf,
    data_dim=1, latent_dim=4, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

store(BimodalGaussExperimentConf, group="experiment", name="bimodal_gauss")
store(BimodalDiffExperimentConf, group="experiment", name="bimodal_diff")


# ---------------------------------------------------------------------------
# Robot navigation 2D: spatial trajectory, D=2, j=2.
# ---------------------------------------------------------------------------

RobotGauss2DExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="robot-basis-pursuit", T=64, D=2,
                                      N_per_split=1024, batch_size=32),
    transition_conf=TransitionGaussianConf,
    hyperparams_conf=_RobotHyperGauss,
    training_conf=TrainingScalarsConf(steps=2000, log_every=50, checkpoint_every=500, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=Robot2DEvalConf,
    viz_conf=Robot2DVizConf,
    data_dim=2, latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

RobotDiff2DExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="robot-basis-pursuit", T=64, D=2,
                                      N_per_split=1024, batch_size=32),
    transition_conf=TransitionDiffusionConf,
    hyperparams_conf=_RobotHyperDiff,
    training_conf=TrainingScalarsConf(steps=4000, log_every=50, checkpoint_every=500, amp=False),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=Robot2DEvalConf,
    viz_conf=Robot2DVizConf,
    data_dim=2, latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
    use_observation_mask=False,
)

store(RobotGauss2DExperimentConf, group="experiment", name="robot_gauss_2d")
store(RobotDiff2DExperimentConf, group="experiment", name="robot_diff_2d")
