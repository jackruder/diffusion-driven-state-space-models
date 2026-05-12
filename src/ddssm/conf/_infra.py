"""Infrastructure layer for the DDSSM Hydra configuration.

Defines the ``ZenStore`` plus the primitive config classes used by
Python-authored experiment configs: transitions, architecture groups, data
modules, trainable masks, trainer partials, and stages.

Nothing in this module registers experiment presets with the store —
that is done in the ``experiments/`` subpackage.  The store is
finalised (``add_to_hydra_store``) by ``conf/__init__.py`` after all
registrations have run.
"""

# ruff: noqa: ANN401

from __future__ import annotations

from typing import Any, List
from dataclasses import field, dataclass

from hydra_zen import ZenStore, just, builds  # noqa: F401 — ``just`` re-exported
from omegaconf import MISSING

from ..dssd import DDSSMConf, DDSSMHyperParamsConf
from ..train import DDSSMTrainer, DDSSMTrainerConf
from ..decoder import GaussianDecoder, GaussianDecoderConf
from ..encoder import (
    GaussianEncoder,
    GaussianInitPrior,
    GaussianEncoderConf,
    GaussianInitPriorConf,
)
from ..diffnets import (
    CSDIUnet,
    CSDIUnetConf,
    ContextProducer,
    MLPCSDIUnetConf,
    TimeMixerConfig,
    FeatureMixerConfig,
    ContextProducerConf,
    ResidualBlockConfig,
    MLPContextProducerConf,
    DiffResidualBlockConfig,
)
from ..gaussians import GaussianHeadConf
from ..experiment import Experiment, ObjectiveSpec, TrainingScalars, TrainableModules
from ..viz.runner import VizSpec, PlotSpec
from ..eval.runner import EvalSpec
from ..data.datamodule import (
    KDDDataModule,
    NullDataModule,
    SyntheticDataModule,
)
from ..variance.runner import ProbeCell, ProbeSpec, ProbePlotSpec, ProbeMetricSpec
from ..transitions.diffusion import DiffusionTransition, DiffusionScheduleConfig
from ..transitions.transitions import GaussianTransition
from ..transitions.diffusion_v2 import (
    DiffusionV2Transition,
    DiffusionV2ScheduleConfig,
)

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
    context="${context}",
    gaussian_head=GaussianHeadConf(),
)

TransitionDiffusionConf = builds(
    DiffusionTransition,
    populate_full_signature=True,
    latent_dim="${experiment.latent_dim}",
    j="${experiment.j}",
    emb_time_dim="${experiment.emb_time_dim}",
    covariate_dim="${experiment.covariate_dim}",
    unet="${unet}",
    schedule=DiffusionScheduleConfig(),
)

TransitionDiffusionV2Conf = builds(
    DiffusionV2Transition,
    populate_full_signature=True,
    latent_dim="${experiment.latent_dim}",
    j="${experiment.j}",
    emb_time_dim="${experiment.emb_time_dim}",
    covariate_dim="${experiment.covariate_dim}",
    unet="${unet}",
    schedule=DiffusionV2ScheduleConfig(),
)


# ---------------------------------------------------------------------------
# ZenStore (shared singleton — all experiment modules append to this store).
# ---------------------------------------------------------------------------

store = ZenStore(name="ddssm")


# ---------------------------------------------------------------------------
# ``time_mixer`` and ``feature_mixer`` config groups.
#
# These select the per-channel mixer architectures used inside the residual
# blocks of both ``CSDIUnet`` (diffusion U-Net) and ``ContextProducer``
# (encoder/decoder/init-prior context). Choices are wired in via top-level
# interpolation so that one CLI flag swaps the mixer everywhere it appears.
# ---------------------------------------------------------------------------

TimeMixerConvConf = builds(
    TimeMixerConfig,
    type="conv",
    kernel_size=3,
    populate_full_signature=True,
)
TimeMixerGRUConf = builds(
    TimeMixerConfig,
    type="gru",
    gru_layers=1,
    populate_full_signature=True,
)
TimeMixerIdentityConf = builds(
    TimeMixerConfig,
    type="identity",
    populate_full_signature=True,
)

FeatureMixerTransformerConf = builds(
    FeatureMixerConfig,
    type="transformer",
    nheads=8,
    n_layers=1,
    populate_full_signature=True,
)
FeatureMixerConvConf = builds(
    FeatureMixerConfig,
    type="conv",
    populate_full_signature=True,
)
FeatureMixerIdentityConf = builds(
    FeatureMixerConfig,
    type="identity",
    populate_full_signature=True,
)

store(TimeMixerConvConf, group="time_mixer", name="conv")
store(TimeMixerGRUConf, group="time_mixer", name="gru")
store(TimeMixerIdentityConf, group="time_mixer", name="identity")

store(FeatureMixerTransformerConf, group="feature_mixer", name="transformer")
store(FeatureMixerConvConf, group="feature_mixer", name="conv")
store(FeatureMixerIdentityConf, group="feature_mixer", name="identity")


# ---------------------------------------------------------------------------
# ``context`` config group: ``ContextProducer`` (default, CSDI-style residual
# stack) vs ``MLPContextProducer`` (MLP ablation). The selected entry is
# threaded into encoder / decoder / z_init / transition via ``${context}``.
#
# The CSDI variant interpolates its inner residual block's time/feature
# mixers from the corresponding groups so that all four module slots share
# a single ``time_mixer=…`` / ``feature_mixer=…`` override.
# ---------------------------------------------------------------------------

ResidualBlockConf = builds(
    ResidualBlockConfig,
    time="${time_mixer}",
    feature="${feature_mixer}",
    populate_full_signature=True,
)

ContextProducerCSDIConf = builds(
    ContextProducer,
    builds_bases=(ContextProducerConf,),
    residual_block=ResidualBlockConf,
    zen_partial=True,
)

store(ContextProducerCSDIConf, group="context", name="csdi")
store(MLPContextProducerConf, group="context", name="mlp")


# ---------------------------------------------------------------------------
# ``unet`` config group: ``CSDIUnet`` (default residual stack with selectable
# mixers) vs ``MLPCSDIUnet`` (MLP ablation). Selected via ``${unet}`` inside
# the diffusion transition Confs.
# ---------------------------------------------------------------------------

DiffResidualBlockConf = builds(
    DiffResidualBlockConfig,
    time="${time_mixer}",
    feature="${feature_mixer}",
    populate_full_signature=True,
)

CSDIUnetGroupConf = builds(
    CSDIUnet,
    builds_bases=(CSDIUnetConf,),
    residual_block=DiffResidualBlockConf,
    zen_partial=True,
)

store(CSDIUnetGroupConf, group="unet", name="csdi")
store(MLPCSDIUnetConf, group="unet", name="mlp")

# ---------------------------------------------------------------------------
# Encoder / Decoder / InitPrior group Confs.
#
# These mirror the ``transition`` group: each module is registered as a
# named choice inside its own config group, with shape kwargs interpolating
# from the active ``experiment.*`` subtree.  ``DDSSMConf`` then refers to
# whichever option the user selected via ``${experiment.encoder}``,
# ``${experiment.decoder}``, ``${experiment.z_init}``.
# ---------------------------------------------------------------------------

EncoderGaussianConf = builds(
    GaussianEncoder,
    builds_bases=(GaussianEncoderConf,),
    populate_full_signature=True,
    data_dim="${experiment.data_dim}",
    latent_dim="${experiment.latent_dim}",
    j="${experiment.j}",
    emb_time_dim="${experiment.emb_time_dim}",
    covariate_dim="${experiment.covariate_dim}",
    use_mask="${experiment.use_observation_mask}",
    context="${context}",
)

DecoderGaussianConf = builds(
    GaussianDecoder,
    builds_bases=(GaussianDecoderConf,),
    populate_full_signature=True,
    latent_dim="${experiment.latent_dim}",
    data_dim="${experiment.data_dim}",
    j="${experiment.j}",
    emb_time_dim="${experiment.emb_time_dim}",
    covariate_dim="${experiment.covariate_dim}",
    context="${context}",
)

InitPriorGaussianConf = builds(
    GaussianInitPrior,
    builds_bases=(GaussianInitPriorConf,),
    populate_full_signature=True,
    latent_dim="${experiment.latent_dim}",
    j="${experiment.j}",
    emb_time_dim="${experiment.emb_time_dim}",
    covariate_dim="${experiment.covariate_dim}",
    context="${context}",
    aux_context="${context}",
)

store(TransitionGaussianConf, group="transition", name="gaussian")
store(TransitionDiffusionConf, group="transition", name="diffusion")
store(TransitionDiffusionV2Conf, group="transition", name="diffusion_v2")
store(EncoderGaussianConf, group="encoder", name="gaussian")
store(DecoderGaussianConf, group="decoder", name="gaussian")
store(InitPriorGaussianConf, group="z_init", name="gaussian")
store(DDSSMHyperParamsConf, group="hyperparams", name="default")
store(DDSSMConf, group="model", name="default")
store(DDSSMTrainerConf, group="trainer", name="default")


# ---------------------------------------------------------------------------
# DataModule confs (one entry per concrete data module)
# ---------------------------------------------------------------------------

NullDataModuleConf = builds(NullDataModule, populate_full_signature=True)
SyntheticDataModuleConf = builds(
    SyntheticDataModule,
    populate_full_signature=True,
    D="${experiment.data_dim}",
    T=64,
    use_observation_mask="${experiment.use_observation_mask}",
)
KDDDataModuleConf = builds(
    KDDDataModule,
    populate_full_signature=True,
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
ProbeCellConf = builds(ProbeCell, populate_full_signature=True)
ProbeMetricSpecConf = builds(ProbeMetricSpec, populate_full_signature=True)
ProbePlotSpecConf = builds(ProbePlotSpec, populate_full_signature=True)
ProbeSpecConf = builds(ProbeSpec, populate_full_signature=True)

# Pre-built ``trainable`` masks.  Each experiment can pick one with
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
    DDSSMTrainer,
    populate_full_signature=True,
    zen_partial=True,
)


def build_experiment_conf(
    *,
    data_conf: Any,
    hyperparams_conf: Any,
    training_conf: Any,
    objective_conf: Any = None,
    eval_conf: Any = None,
    viz_conf: Any = None,
    variance_conf: Any = None,
    data_dim: int,
    latent_dim: int,
    j: int = 1,
    emb_time_dim: int = 16,
    covariate_dim: int = 0,
    use_observation_mask: bool = False,
    checkpoint_dir: str = "./checkpoints",
    seed: int = 0,
    transition_conf: Any = None,
    encoder_conf: Any = None,
    decoder_conf: Any = None,
    z_init_conf: Any = None,
) -> Any:
    """Compose an instantiable :class:`Experiment` config from Python parts.

    Experiment presets call this directly with all important knobs visible in
    Python source.  That keeps presets Pyright-friendly while preserving Hydra
    config-group interpolation for CLI overrides.

    ``transition_conf``, ``encoder_conf``, ``decoder_conf`` and
    ``z_init_conf`` are all optional.  When omitted the experiment uses
    the top-level config-group selection (defaults: ``transition=gaussian``,
    ``encoder=gaussian``, ``decoder=gaussian``, ``z_init=gaussian``), so
    callers can switch e.g. ``experiment=harmonic transition=diffusion
    encoder=gaussian`` from the CLI without touching Python.  Pass an
    explicit conf only when the preset must lock in a specific
    implementation.
    """
    resolved_transition = (
        transition_conf if transition_conf is not None else "${transition}"
    )
    resolved_encoder = encoder_conf if encoder_conf is not None else "${encoder}"
    resolved_decoder = decoder_conf if decoder_conf is not None else "${decoder}"
    resolved_z_init = z_init_conf if z_init_conf is not None else "${z_init}"
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
        variance=variance_conf,
        seed=seed,
        data_dim=data_dim,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        covariate_dim=covariate_dim,
        use_observation_mask=use_observation_mask,
        checkpoint_dir=checkpoint_dir,
        transition=resolved_transition,
        encoder=resolved_encoder,
        decoder=resolved_decoder,
        z_init=resolved_z_init,
        hyperparams=hyperparams_conf,
    )


# ---------------------------------------------------------------------------
# Stages dataclasses (config-only; full logic lives in ddssm.stages)
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
