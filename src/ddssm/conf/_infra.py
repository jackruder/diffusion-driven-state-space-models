"""Infrastructure layer for the DDSSM Hydra configuration.

Defines the ``ZenStore``, all primitive config classes (transitions,
data-modules, trainable masks, trainer partial, stages), and the
``_experiment_conf`` composer helper.

Nothing in this module registers experiment presets with the store —
that is done in the ``experiments/`` subpackage.  The store is
finalised (``add_to_hydra_store``) by ``conf/__init__.py`` after all
registrations have run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from hydra_zen import builds, ZenStore, just  # noqa: F401 — ``just`` re-exported
from omegaconf import MISSING

from ..data.datamodule import (
    KDDDataModule,
    NullDataModule,
    SyntheticDataModule,
)
from ..decoder import GaussianDecoder, GaussianDecoderConf
from ..diffnets import ContextProducerConf, CSDIUnetConf
from ..dssd import DDSSMConf, DDSSMHyperParamsConf, REWOConf
from ..encoder import (
    GaussianEncoder,
    GaussianEncoderConf,
    GaussianInitPrior,
    GaussianInitPriorConf,
)
from ..eval.runner import EvalSpec
from ..experiment import Experiment, ObjectiveSpec, TrainableModules, TrainingScalars
from ..viz.runner import PlotSpec, VizSpec
from ..gaussians import GaussianHeadConf
from ..train import DDSSMTrainer, DDSSMTrainerConf
from ..transitions.diffusion import DiffusionScheduleConfig, DiffusionTransition
from ..transitions.diffusion_v2 import (
    DiffusionV2ScheduleConfig,
    DiffusionV2Transition,
)
from ..transitions.transitions import GaussianTransition


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

TransitionDiffusionV2Conf = builds(
    DiffusionV2Transition,
    populate_full_signature=True,
    latent_dim="${experiment.latent_dim}",
    j="${experiment.j}",
    emb_time_dim="${experiment.emb_time_dim}",
    covariate_dim="${experiment.covariate_dim}",
    unet=CSDIUnetConf(),
    schedule=DiffusionV2ScheduleConfig(),
)


# ---------------------------------------------------------------------------
# ZenStore (shared singleton — all experiment modules append to this store).
# ---------------------------------------------------------------------------

store = ZenStore(name="ddssm")

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
)

InitPriorGaussianConf = builds(
    GaussianInitPrior,
    builds_bases=(GaussianInitPriorConf,),
    populate_full_signature=True,
    latent_dim="${experiment.latent_dim}",
    j="${experiment.j}",
    emb_time_dim="${experiment.emb_time_dim}",
    covariate_dim="${experiment.covariate_dim}",
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
    DDSSMTrainer, populate_full_signature=True, zen_partial=True,
)


def _experiment_conf(
    *,
    data_conf,
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
    transition_conf=None,
    encoder_conf=None,
    decoder_conf=None,
    z_init_conf=None,
):
    """Compose an Experiment config from its parts.

    Centralizes the wiring so each preset is a one-liner pointing at
    the right Confs.

    ``transition_conf``, ``encoder_conf``, ``decoder_conf`` and
    ``z_init_conf`` are all optional.  When omitted the experiment uses
    the top-level config-group selection (defaults: ``transition=gaussian``,
    ``encoder=gaussian``, ``decoder=gaussian``, ``z_init=gaussian``), so
    callers can switch e.g. ``experiment=harmonic transition=diffusion
    encoder=gaussian`` from the CLI without touching Python.  Pass an
    explicit conf only when the preset must lock in a specific
    implementation.
    """
    resolved_transition = transition_conf if transition_conf is not None else "${transition}"
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
