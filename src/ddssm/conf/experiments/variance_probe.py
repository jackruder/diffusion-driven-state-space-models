"""Variance probe experiments across synthetic datasets."""

from __future__ import annotations

from .._infra import (
    ObjectiveSpecConf,
    TrainingScalarsConf,
    DDSSMHyperParamsConf,
    SyntheticDataModuleConf,
    TransitionDiffusionV2Conf,
    store,
    build_experiment_conf,
)
from .._variance import (
    LGSSMVarianceConf,
    BimodalCleanVarianceConf,
    BimodalNoisyVarianceConf,
    NonlinearBimodalLiftVarianceConf,
)

_TRAINING = TrainingScalarsConf(
    steps=300,
    log_every=20,
    checkpoint_every=100,
    amp=False,
)
_HP = DDSSMHyperParamsConf(
    batch_size=32,
    grad_accum_steps=1,
    lambda_schedule="cosine",
    lambda_start=0.001,
    lambda_end=1.0,
    lambda_warmup_steps=50,
    enc_lr=5e-4,
    dec_lr=5e-4,
    zinit_lr=5e-4,
    trans_lr=5e-4,
    S=1,
)

VarianceProbeLGSSMExperimentConf = build_experiment_conf(
    data_conf=SyntheticDataModuleConf(
        mode="lgssm", T=64, D=1, N_per_split=256, batch_size=32
    ),
    transition_conf=TransitionDiffusionV2Conf,
    hyperparams_conf=_HP,
    training_conf=_TRAINING,
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    variance_conf=LGSSMVarianceConf,
    checkpoint_dir="${oc.env:PWD,.}/runs/variance_probe/variance_probe_lgssm/checkpoints",
    data_dim=1,
    latent_dim=4,
    emb_time_dim=16,
    covariate_dim=0,
    use_observation_mask=False,
)
store(VarianceProbeLGSSMExperimentConf, group="experiment", name="variance_probe_lgssm")

VarianceProbeBimodalCleanExperimentConf = build_experiment_conf(
    data_conf=SyntheticDataModuleConf(
        mode="bimodal", T=64, D=1, N_per_split=256, batch_size=32
    ),
    transition_conf=TransitionDiffusionV2Conf,
    hyperparams_conf=_HP,
    training_conf=_TRAINING,
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    variance_conf=BimodalCleanVarianceConf,
    checkpoint_dir="${oc.env:PWD,.}/runs/variance_probe/variance_probe_bimodal_clean/checkpoints",
    data_dim=1,
    latent_dim=4,
    emb_time_dim=16,
    covariate_dim=0,
    use_observation_mask=False,
)
store(
    VarianceProbeBimodalCleanExperimentConf,
    group="experiment",
    name="variance_probe_bimodal_clean",
)

VarianceProbeBimodalNoisyExperimentConf = build_experiment_conf(
    data_conf=SyntheticDataModuleConf(
        mode="bimodal-noisy", T=64, D=1, N_per_split=256, batch_size=32
    ),
    transition_conf=TransitionDiffusionV2Conf,
    hyperparams_conf=_HP,
    training_conf=_TRAINING,
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    variance_conf=BimodalNoisyVarianceConf,
    checkpoint_dir="${oc.env:PWD,.}/runs/variance_probe/variance_probe_bimodal_noisy/checkpoints",
    data_dim=1,
    latent_dim=4,
    emb_time_dim=16,
    covariate_dim=0,
    use_observation_mask=False,
)
store(
    VarianceProbeBimodalNoisyExperimentConf,
    group="experiment",
    name="variance_probe_bimodal_noisy",
)

VarianceProbeNonlinearBimodalLiftExperimentConf = build_experiment_conf(
    data_conf=SyntheticDataModuleConf(
        mode="nonlinear-bimodal-lift",
        T=64,
        D=4,
        N_per_split=256,
        batch_size=32,
    ),
    transition_conf=TransitionDiffusionV2Conf,
    hyperparams_conf=_HP,
    training_conf=_TRAINING,
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    variance_conf=NonlinearBimodalLiftVarianceConf,
    checkpoint_dir="${oc.env:PWD,.}/runs/variance_probe/variance_probe_nonlinear_bimodal_lift/checkpoints",
    data_dim=4,
    latent_dim=8,
    emb_time_dim=16,
    covariate_dim=0,
    use_observation_mask=False,
)
store(
    VarianceProbeNonlinearBimodalLiftExperimentConf,
    group="experiment",
    name="variance_probe_nonlinear_bimodal_lift",
)
