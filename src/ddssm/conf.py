"""Central hydra-zen configuration: ZenStore + config-only dataclasses.

Top-level ``*Conf`` classes live next to the classes they describe:
  - ``DDSSMHyperParamsConf`` and ``DDSSMConf`` in ``ddssm.dssd``
  - ``DDSSMTrainerConf`` in ``ddssm.train``
  - per-module ``*Conf`` in their respective modules

This module:
  - Re-exports those configs for ``from ddssm.conf import ...`` access.
  - Defines store-registered ``transition`` Confs with ``${...}`` interpolations
    on shape kwargs (``latent_dim``, ``j``, ``emb_time_dim``, ``covariate_dim``)
    so a defaults-list selection like ``- transition: gaussian`` produces a
    fully-wired structured config without needing per-field YAML.
  - Owns the ``ZenStore`` and registers the ``transition``, ``model``,
    ``trainer`` config groups, then materialises them into Hydra's ConfigStore
    so ``@hydra.main`` can resolve them.
  - Defines the slim Stages dataclasses (``StageSpecConf`` / ``StagesConf``)
    that are config-only (full stage logic lives in ``ddssm.stages``).
  - Provides ``load_yaml_config(yaml_path)`` for back-compat YAML loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from hydra_zen import builds, ZenStore, just
from omegaconf import MISSING

from .data.datamodule import (
    KDDDataModule,
    NullDataModule,
    SyntheticDataModule,
)
from .diffnets import ContextProducerConf, CSDIUnetConf
from .dssd import DDSSMConf, DDSSMHyperParamsConf, REWOConf
from .eval.runner import EvalSpec
from .experiment import Experiment, ObjectiveSpec, TrainableModules, TrainingScalars
from .viz.runner import PlotSpec, VizSpec
from .gaussians import GaussianHeadConf
from .train import DDSSMTrainer, DDSSMTrainerConf
from .transitions.diffusion import DiffusionScheduleConfig, DiffusionTransition
from .transitions.transitions import GaussianTransition


# ---------------------------------------------------------------------------
# Top-level transition Confs for the ``transition`` config group.
#
# Shape kwargs interpolate from root cfg keys; sub-module Confs stay nested
# (their own zen_partial defaults handle inner shape wiring at construction).
# ---------------------------------------------------------------------------

TransitionGaussianConf = builds(
    GaussianTransition,
    populate_full_signature=True,
    latent_dim="${experiment.latent_dim}",
    j="${experiment.j}",
    emb_time_dim="${experiment.emb_time_dim}",
    covariate_dim="${experiment.covariate_dim}",
    context=ContextProducerConf(),
    gaussian_head=GaussianHeadConf(),
)

TransitionDiffusionConf = builds(
    DiffusionTransition,
    populate_full_signature=True,
    latent_dim="${experiment.latent_dim}",
    j="${experiment.j}",
    emb_time_dim="${experiment.emb_time_dim}",
    covariate_dim="${experiment.covariate_dim}",
    unet=CSDIUnetConf(),
    schedule=DiffusionScheduleConfig(),
)


# ---------------------------------------------------------------------------
# ZenStore with config groups
# ---------------------------------------------------------------------------

store = ZenStore(name="ddssm")

store(TransitionGaussianConf, group="transition", name="gaussian")
store(TransitionDiffusionConf, group="transition", name="diffusion")
store(DDSSMHyperParamsConf, group="hyperparams", name="default")
store(DDSSMConf, group="model", name="default")
store(DDSSMTrainerConf, group="trainer", name="default")


# ---------------------------------------------------------------------------
# DataModule confs (one entry per concrete data module)
# ---------------------------------------------------------------------------

NullDataModuleConf = builds(NullDataModule, populate_full_signature=True)
SyntheticDataModuleConf = builds(
    SyntheticDataModule, populate_full_signature=True,
    D="${experiment.data_dim}",
    T=64,
    use_observation_mask="${experiment.use_observation_mask}",
)
KDDDataModuleConf = builds(
    KDDDataModule, populate_full_signature=True,
    use_observation_mask="${experiment.use_observation_mask}",
)


# ---------------------------------------------------------------------------
# Experiment dataclass confs
# ---------------------------------------------------------------------------

TrainableModulesConf = builds(TrainableModules, populate_full_signature=True)
TrainingScalarsConf = builds(
    TrainingScalars,
    populate_full_signature=True,
    trainable=TrainableModulesConf(),
)
ObjectiveSpecConf = builds(ObjectiveSpec, populate_full_signature=True)
EvalSpecConf = builds(EvalSpec, populate_full_signature=True)
PlotSpecConf = builds(PlotSpec, populate_full_signature=True)
VizSpecConf = builds(VizSpec, populate_full_signature=True)

# Pre-built ``trainable`` masks. Each experiment can pick one with
# ``training=...`` style overrides or by passing the conf directly.
TrainableJointConf = TrainableModulesConf()
TrainableReconOnlyConf = TrainableModulesConf(
    encoder=True, decoder=True, z_init=True, transition=False
)
TrainableTransOnlyConf = TrainableModulesConf(
    encoder=False, decoder=False, z_init=False, transition=True
)

# ``build_trainer`` is a partial of DDSSMTrainer that the Experiment
# completes at run time with model + device + run-dir-derived paths.
DDSSMTrainerPartial = builds(
    DDSSMTrainer, populate_full_signature=True, zen_partial=True,
)


def _experiment_conf(
    *,
    data_conf,
    transition_conf,
    hyperparams_conf,
    training_conf,
    objective_conf=None,
    eval_conf=None,
    viz_conf=None,
    data_dim: int,
    latent_dim: int,
    j: int = 1,
    emb_time_dim: int = 16,
    covariate_dim: int = 0,
    use_observation_mask: bool = False,
    checkpoint_dir: str = "./checkpoints",
    seed: int = 0,
):
    """Compose an Experiment config from its parts.

    Centralizes the wiring so each preset is a one-liner pointing at
    the right Confs.
    """
    return builds(
        Experiment,
        populate_full_signature=True,
        data=data_conf,
        model=DDSSMConf,  # interpolates from ${experiment.*}
        build_trainer=DDSSMTrainerPartial,
        training=training_conf,
        objective=objective_conf,
        eval=eval_conf,
        viz=viz_conf,
        seed=seed,
        data_dim=data_dim,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        covariate_dim=covariate_dim,
        use_observation_mask=use_observation_mask,
        checkpoint_dir=checkpoint_dir,
        transition=transition_conf,
        hyperparams=hyperparams_conf,
    )


# Eval defaults per dataset family. Synthetic uses recon-only because
# its sequence-format data has no canonical past/future split unless
# the user picks one; KDD always splits at L1 (carried in metadata).
SynthEvalConf = EvalSpecConf(metrics=["loss_tail", "recon_mse"], split="val")
KDDEvalConf = EvalSpecConf(metrics=["mae", "crps_sum"], split="test", num_samples=32)

# Viz defaults per dataset family.
SynthVizConf = VizSpecConf(
    plots=[
        PlotSpecConf(name="metrics_csv", save_filename="train_loss.png",
                     kwargs={"keys": ["loss/total"], "log_y": True}),
    ],
    split="val",
    num_samples=10,
    T_split=32,  # half of synthetic's default T=64; override per-experiment if needed
)
KDDVizConf = VizSpecConf(
    plots=[
        PlotSpecConf(name="forecast_1d", save_filename="forecast.png",
                     kwargs={"n_show": 4}),
        PlotSpecConf(name="metrics_csv", save_filename="train_loss.png",
                     kwargs={"keys": ["loss/total"], "log_y": True}),
    ],
    split="test",
    num_samples=32,
    # T_split picks up data.metadata.forecast_split == L1 automatically.
)


# Synthetic + Gaussian transition: small LGSSM run for smoke tests / CI.
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

# Synthetic + Diffusion transition.
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

# KDD + Gaussian transition (real data via data/kdd.pt).
KDDGaussExperimentConf = _experiment_conf(
    data_conf=KDDDataModuleConf(batch_size=128, eval_step_size=24),
    transition_conf=TransitionGaussianConf,
    hyperparams_conf=DDSSMHyperParamsConf(
        batch_size=128, grad_accum_steps=1, lambda_schedule="linear",
        lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=500,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=1,
    ),
    training_conf=TrainingScalarsConf(steps=5000, log_every=50, checkpoint_every=500, amp=True),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=KDDEvalConf,
    viz_conf=KDDVizConf,
    data_dim=6, latent_dim=8, emb_time_dim=32, covariate_dim=3,
    use_observation_mask=False,
)

# KDD + Diffusion transition.
KDDDiffusionExperimentConf = _experiment_conf(
    data_conf=KDDDataModuleConf(batch_size=64, eval_step_size=24),
    transition_conf=TransitionDiffusionConf,
    hyperparams_conf=DDSSMHyperParamsConf(
        batch_size=64, grad_accum_steps=1, lambda_schedule="linear",
        lambda_start=0.001, lambda_end=1.0, lambda_warmup_steps=500,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4, S=1,
    ),
    training_conf=TrainingScalarsConf(steps=8000, log_every=50, checkpoint_every=500, amp=True),
    objective_conf=ObjectiveSpecConf(metric="loss/total", split="train", tail_frac=0.1),
    eval_conf=KDDEvalConf,
    viz_conf=KDDVizConf,
    data_dim=6, latent_dim=8, emb_time_dim=32, covariate_dim=3,
    use_observation_mask=False,
)

store(SyntheticGaussExperimentConf, group="experiment", name="synthetic_gauss")
store(SyntheticDiffusionExperimentConf, group="experiment", name="synthetic_diffusion")
store(KDDGaussExperimentConf, group="experiment", name="kdd_gauss")
store(KDDDiffusionExperimentConf, group="experiment", name="kdd_diffusion")

# ---------------------------------------------------------------------------
# Eval/viz defaults for synthetic verification experiments.
# ---------------------------------------------------------------------------

HarmonicEvalConf = EvalSpecConf(metrics=["mae", "crps_sum"], split="val", num_samples=32, T_split=32)
HarmonicVizConf = VizSpecConf(
    plots=[
        PlotSpecConf(name="forecast_1d", save_filename="forecast.png",
                     kwargs={"n_show": 4}),
        PlotSpecConf(name="metrics_csv", save_filename="train_loss.png",
                     kwargs={"keys": ["loss/total"], "log_y": True}),
    ],
    split="val",
    num_samples=32,
    T_split=32,
)

BimodalEvalConf = EvalSpecConf(metrics=["energy_score", "crps_sum"], split="val", num_samples=64, T_split=32)
BimodalVizConf = VizSpecConf(
    plots=[
        PlotSpecConf(name="forecast_1d", save_filename="forecast.png",
                     kwargs={"n_show": 4}),
        PlotSpecConf(name="metrics_csv", save_filename="train_loss.png",
                     kwargs={"keys": ["loss/total"], "log_y": True}),
    ],
    split="val",
    num_samples=64,
    T_split=32,
)

Robot2DEvalConf = EvalSpecConf(metrics=["energy_score", "crps_sum"], split="val", num_samples=32, T_split=32)
Robot2DVizConf = VizSpecConf(
    plots=[
        PlotSpecConf(name="forecast_2d_spatial", save_filename="forecast_2d.png",
                     kwargs={"n_show": 4}),
        PlotSpecConf(name="metrics_csv", save_filename="train_loss.png",
                     kwargs={"keys": ["loss/total"], "log_y": True}),
    ],
    split="val",
    num_samples=32,
    T_split=32,
)

# ---------------------------------------------------------------------------
# Harmonic: clean sine-wave signal.
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

# ---------------------------------------------------------------------------
# Harmonic-noisy: sine wave with higher observation noise.
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

# ---------------------------------------------------------------------------
# Bimodal: multimodality comparison — energy score is the headline metric.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Robot navigation 2D: spatial trajectory, D=2, j=2.
# ---------------------------------------------------------------------------

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

store(HarmonicGaussExperimentConf, group="experiment", name="harmonic_gauss")
store(HarmonicDiffExperimentConf, group="experiment", name="harmonic_diff")
store(HarmonicGaussJ2ExperimentConf, group="experiment", name="harmonic_gauss_j2")
store(HarmonicDiffJ2ExperimentConf, group="experiment", name="harmonic_diff_j2")
store(HarmonicNoisyGaussExperimentConf, group="experiment", name="harmonic_noisy_gauss")
store(HarmonicNoisyDiffExperimentConf, group="experiment", name="harmonic_noisy_diff")
store(BimodalGaussExperimentConf, group="experiment", name="bimodal_gauss")
store(BimodalDiffExperimentConf, group="experiment", name="bimodal_diff")
store(RobotGauss2DExperimentConf, group="experiment", name="robot_gauss_2d")
store(RobotDiff2DExperimentConf, group="experiment", name="robot_diff_2d")

# Materialise the store into Hydra's ConfigStore so @hydra.main can resolve it.
# Importing this module is sufficient to activate the registrations.
store.add_to_hydra_store(overwrite_ok=True)


# ---------------------------------------------------------------------------
# Stages dataclasses (slim versions; full logic lives in stages.py)
# ---------------------------------------------------------------------------

@dataclass
class StageLrsConf:
    dec_lr: float = 5e-4
    zinit_lr: float = 5e-4
    trans_lr: float = 0.0


@dataclass
class StageTrainableConf:
    decoder: bool = True
    z_init: bool = True
    transition: bool = False


@dataclass
class StageSchedulerConf:
    warmup_steps: int = 0
    final_lr_scale: float = 1.0


@dataclass
class LambdaRampConf:
    end: float | None = 1.0
    delay: int = 0
    steps: int | None = None


@dataclass
class StageSpecConf:
    steps: int = MISSING
    trainable: StageTrainableConf = field(default_factory=StageTrainableConf)
    lrs: StageLrsConf = field(default_factory=StageLrsConf)
    scheduler: StageSchedulerConf = field(default_factory=StageSchedulerConf)
    carry_diff_moments: bool = False
    lambda_ramp: LambdaRampConf = field(default_factory=LambdaRampConf)
    log_every: int = 10
    val_every: int = 100
    checkpoint_every: int = 1000


@dataclass
class StagesConf:
    stage_2: StageSpecConf | None = None
    stage_3: StageSpecConf | None = None
    run: List[str] = field(default_factory=lambda: ["stage_1", "stage_2", "stage_3"])


# ---------------------------------------------------------------------------
# Convenience helper: load a Hydra YAML and return an instantiable config.
# ---------------------------------------------------------------------------

def load_yaml_config(yaml_path: str) -> Any:
    """Load a Hydra-compatible YAML and return an OmegaConf DictConfig.

    The returned object can be passed to ``hydra_zen.instantiate(cfg.model)``
    (or any sub-key) to construct the corresponding object.
    """
    from omegaconf import OmegaConf

    with open(yaml_path, "r") as f:
        cfg = OmegaConf.load(f)
    return cfg


__all__ = [
    "DDSSMConf",
    "DDSSMHyperParamsConf",
    "DDSSMTrainerConf",
    "DDSSMTrainerPartial",
    "REWOConf",
    "TransitionGaussianConf",
    "TransitionDiffusionConf",
    "NullDataModuleConf",
    "SyntheticDataModuleConf",
    "KDDDataModuleConf",
    "TrainingScalarsConf",
    "ObjectiveSpecConf",
    "SyntheticGaussExperimentConf",
    "SyntheticDiffusionExperimentConf",
    "KDDGaussExperimentConf",
    "KDDDiffusionExperimentConf",
    "HarmonicEvalConf",
    "HarmonicVizConf",
    "HarmonicGaussExperimentConf",
    "HarmonicDiffExperimentConf",
    "HarmonicGaussJ2ExperimentConf",
    "HarmonicDiffJ2ExperimentConf",
    "HarmonicNoisyGaussExperimentConf",
    "HarmonicNoisyDiffExperimentConf",
    "BimodalEvalConf",
    "BimodalVizConf",
    "BimodalGaussExperimentConf",
    "BimodalDiffExperimentConf",
    "Robot2DEvalConf",
    "Robot2DVizConf",
    "RobotGauss2DExperimentConf",
    "RobotDiff2DExperimentConf",
    "StageLrsConf",
    "StageTrainableConf",
    "StageSchedulerConf",
    "LambdaRampConf",
    "StageSpecConf",
    "StagesConf",
    "load_yaml_config",
    "store",
]
