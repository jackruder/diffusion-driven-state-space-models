"""Variance probe experiments across synthetic datasets."""

from __future__ import annotations

from .._infra import (
    DDSSMHyperParamsConf,
    ObjectiveSpecConf,
    SyntheticDataModuleConf,
    TrainingScalarsConf,
    TransitionDiffusionV2Conf,
    _experiment_conf,
    store,
)
from .._variance import (
    BimodalCleanVarianceConf,
    BimodalNoisyVarianceConf,
    LGSSMVarianceConf,
    NonlinearBimodalLiftVarianceConf,
)


_TRAINING = TrainingScalarsConf(
    steps=1000,
    log_every=25,
    checkpoint_every=200,
    amp=False,
)
_HP = DDSSMHyperParamsConf(
    batch_size=32,
    grad_accum_steps=1,
    lambda_schedule="cosine",
    lambda_start=0.001,
    lambda_end=1.0,
    lambda_warmup_steps=200,
    enc_lr=5e-4,
    dec_lr=5e-4,
    zinit_lr=5e-4,
    trans_lr=5e-4,
    S=1,
)

VarianceProbeLGSSMExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="lgssm", T=64, D=1, N_per_split=512, batch_size=32),
    transition_conf=TransitionDiffusionV2Conf,
    hyperparams_conf=_HP,
    training_conf=_TRAINING,
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    variance_conf=LGSSMVarianceConf,
    data_dim=1,
    latent_dim=4,
    emb_time_dim=16,
    covariate_dim=0,
    use_observation_mask=False,
)
store(VarianceProbeLGSSMExperimentConf, group="experiment", name="variance_probe_lgssm")

VarianceProbeBimodalCleanExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="bimodal", T=64, D=1, N_per_split=512, batch_size=32),
    transition_conf=TransitionDiffusionV2Conf,
    hyperparams_conf=_HP,
    training_conf=_TRAINING,
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    variance_conf=BimodalCleanVarianceConf,
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

VarianceProbeBimodalNoisyExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(mode="bimodal-noisy", T=64, D=1, N_per_split=512, batch_size=32),
    transition_conf=TransitionDiffusionV2Conf,
    hyperparams_conf=_HP,
    training_conf=_TRAINING,
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    variance_conf=BimodalNoisyVarianceConf,
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

VarianceProbeNonlinearBimodalLiftExperimentConf = _experiment_conf(
    data_conf=SyntheticDataModuleConf(
        mode="nonlinear-bimodal-lift",
        T=64,
        D=4,
        N_per_split=512,
        batch_size=32,
    ),
    transition_conf=TransitionDiffusionV2Conf,
    hyperparams_conf=_HP,
    training_conf=_TRAINING,
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    variance_conf=NonlinearBimodalLiftVarianceConf,
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
