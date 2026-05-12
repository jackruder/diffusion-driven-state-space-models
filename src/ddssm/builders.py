"""Centralized hydra-zen builders for every model part.

One place to import from when assembling an :class:`Experiment` config
in a notebook, org src block, or Python script. Each name here is a
``builds(...)`` config — call it like a function with keyword
overrides::

    from ddssm.builders import DiffTransition, Schedule, Unet
    t = DiffTransition(
        unet=Unet(channels=64, n_layers=4),
        schedule=Schedule(sigma_min=0.01, S_k=20),
    )

Shape kwargs (``data_dim``, ``latent_dim``, ``j``, ``emb_time_dim``,
``covariate_dim``, ``use_mask``) default to OmegaConf ``MISSING`` so
users can construct a slot builder without specifying them up front;
:func:`experiments._make.experiment` fills them in based on the
top-level experiment shape, eliminating any need for
``${experiment.*}`` interpolation.
"""

from __future__ import annotations

from hydra_zen import builds
from omegaconf import MISSING

# Runtime classes — actual constructors targeted by ``builds()``.
from .data.datamodule import KDDDataModule, NullDataModule, SyntheticDataModule
from .decoder import GaussianDecoder
from .diffnets import (
    ContextProducer,
    CSDIUnet,
    DiffResidualBlockConfig,
    FeatureMixerConfig,
    MLPContextProducer,
    MLPCSDIUnet,
    ResidualBlockConfig,
    TimeMixerConfig,
)
from .dssd import DDSSM_base, DDSSMHyperParamsConf, REWOConf  # dataclasses
from .encoder import GaussianEncoder, GaussianInitPrior
from .eval.runner import EvalSpec
from .experiment import Experiment, ObjectiveSpec, TrainableModules, TrainingScalars
from .futsum import GRUFutureSummary, TransformerFutureSummary
from .gaussians import GaussianHead
from .train import DDSSMTrainer
from .transitions.diffusion import DiffusionScheduleConfig, DiffusionTransition
from .transitions.diffusion_v2 import (
    DiffusionV2ScheduleConfig,
    DiffusionV2Transition,
)
from .transitions.transitions import GaussianTransition
from .variance.runner import ProbeCell, ProbeMetricSpec, ProbePlotSpec, ProbeSpec
from .viz.runner import PlotSpec, VizSpec


# ---------------------------------------------------------------------------
# Mixer + residual block builders.
#
# The runtime objects (``TimeMixerConfig`` etc.) are plain dataclasses, but
# we wrap them in hydra-zen ``builds(...)`` so callers/experiments can
# override any sub-field from a single overrides string, e.g.
# ``encoder.context.residual_block.feature.nheads=8``.
# Instantiating one of these configs returns an instance of the underlying
# dataclass — that's what ``ContextProducer``/``CSDIUnet`` expect.
# ---------------------------------------------------------------------------

TimeMixer = builds(TimeMixerConfig, populate_full_signature=True)
FeatureMixer = builds(FeatureMixerConfig, populate_full_signature=True)

ResidualBlock = builds(
    ResidualBlockConfig,
    populate_full_signature=True,
    time=TimeMixer(),
    feature=FeatureMixer(n_layers=2),
)

DiffResidualBlock = builds(
    DiffResidualBlockConfig,
    populate_full_signature=True,
    time=TimeMixer(),
    feature=FeatureMixer(),
)

# ---------------------------------------------------------------------------
# Schedules (plain dataclasses already; re-export under short names).
# ---------------------------------------------------------------------------

Schedule = DiffusionScheduleConfig
ScheduleV2 = DiffusionV2ScheduleConfig

# ---------------------------------------------------------------------------
# Head, context, U-Net, future-summary builders.
# All are ``zen_partial=True`` because the enclosing module supplies the
# shape kwargs (combined_dim, side_dim, …) at construction time.
# ---------------------------------------------------------------------------

Head = builds(GaussianHead, populate_full_signature=True, zen_partial=True)

Context = builds(
    ContextProducer,
    channels=8,
    num_layers=2,
    residual_block=ResidualBlock(),
    populate_full_signature=True,
    zen_partial=True,
)

MLPContext = builds(
    MLPContextProducer,
    channels=8,
    num_layers=2,
    residual_block=ResidualBlock(),
    populate_full_signature=True,
    zen_partial=True,
)

Unet = builds(
    CSDIUnet,
    channels=64,
    n_layers=4,
    embedding_dim=128,
    residual_block=DiffResidualBlock(),
    populate_full_signature=True,
    zen_partial=True,
)

MLPUnet = builds(
    MLPCSDIUnet,
    channels=64,
    n_layers=2,
    embedding_dim=128,
    residual_block=DiffResidualBlock(),
    populate_full_signature=True,
    zen_partial=True,
)

GRUFutSum = builds(
    GRUFutureSummary,
    summary_dim=64,
    num_layers=2,
    populate_full_signature=True,
    zen_partial=True,
)

TransformerFutSum = builds(
    TransformerFutureSummary,
    summary_dim=64,
    num_layers=2,
    populate_full_signature=True,
    zen_partial=True,
)


# ---------------------------------------------------------------------------
# Encoder / Decoder / InitPrior / Transition builders.
# Shapes are caller-supplied; the inner Context/Head/etc. default to the
# builders above so a one-liner like ``Encoder(data_dim=1, latent_dim=4,
# j=1, emb_time_dim=16, use_mask=False)`` is fully instantiable.
# ---------------------------------------------------------------------------

# All shape-related kwargs are MISSING by default so
# ``experiments._make.experiment`` can fill them in one place.

_SHAPE_ENC = dict(
    data_dim=MISSING, latent_dim=MISSING, j=MISSING,
    emb_time_dim=MISSING, covariate_dim=MISSING, use_mask=MISSING,
)
_SHAPE_DEC = dict(
    data_dim=MISSING, latent_dim=MISSING, j=MISSING,
    emb_time_dim=MISSING, covariate_dim=MISSING,
)
_SHAPE_LAT = dict(
    latent_dim=MISSING, j=MISSING,
    emb_time_dim=MISSING, covariate_dim=MISSING,
)

Encoder = builds(
    GaussianEncoder,
    populate_full_signature=True,
    **_SHAPE_ENC,
    context=Context(),
    gaussian_head=Head(clamp_logvar_min=-10.0),
    fut_summary=GRUFutSum(),
)

Decoder = builds(
    GaussianDecoder,
    populate_full_signature=True,
    **_SHAPE_DEC,
    context=Context(),
    gaussian_head=Head(),
)

ZInit = builds(
    GaussianInitPrior,
    populate_full_signature=True,
    **_SHAPE_LAT,
    context=Context(),
    aux_context=Context(),
    gaussian_head=Head(clamp_logvar_min=-10.0),
    aux_posterior_head=Head(clamp_logvar_min=-10.0),
)

GaussTransition = builds(
    GaussianTransition,
    populate_full_signature=True,
    **_SHAPE_LAT,
    context=Context(),
    gaussian_head=Head(),
)

DiffTransition = builds(
    DiffusionTransition,
    populate_full_signature=True,
    **_SHAPE_LAT,
    unet=Unet(),
    schedule=Schedule(),
)

DiffV2Transition = builds(
    DiffusionV2Transition,
    populate_full_signature=True,
    **_SHAPE_LAT,
    unet=Unet(),
    schedule=ScheduleV2(),
)


# ---------------------------------------------------------------------------
# Data modules.
# ---------------------------------------------------------------------------

Synthetic = builds(SyntheticDataModule, populate_full_signature=True)
KDD = builds(KDDDataModule, populate_full_signature=True)
Null = builds(NullDataModule, populate_full_signature=True)


# ---------------------------------------------------------------------------
# Model, hyperparameters, training.
# ``DDSSM`` takes already-instantiated encoder/decoder/z_init/transition
# Confs from the caller; no interpolation.
# ---------------------------------------------------------------------------

Hparams = builds(DDSSMHyperParamsConf, populate_full_signature=True)
Rewo = builds(REWOConf, populate_full_signature=True)

Trainable = builds(TrainableModules, populate_full_signature=True)
Training = builds(
    TrainingScalars,
    populate_full_signature=True,
    trainable=None,
)
Objective = builds(ObjectiveSpec, populate_full_signature=True)

DDSSM = builds(
    DDSSM_base,
    populate_full_signature=True,
    data_dim=MISSING, latent_dim=MISSING, j=MISSING,
)

# ``build_trainer`` is a partial: the experiment fills in model/device/
# logging paths at run time.
TrainerPartial = builds(DDSSMTrainer, populate_full_signature=True, zen_partial=True)


# ---------------------------------------------------------------------------
# Eval / viz / variance specs.
# ---------------------------------------------------------------------------

Eval = builds(EvalSpec, populate_full_signature=True)
Plot = builds(PlotSpec, populate_full_signature=True)
Viz = builds(VizSpec, populate_full_signature=True)
ProbeCellB = builds(ProbeCell, populate_full_signature=True)
ProbeMetric = builds(ProbeMetricSpec, populate_full_signature=True)
ProbePlot = builds(ProbePlotSpec, populate_full_signature=True)
Probe = builds(ProbeSpec, populate_full_signature=True)


# ---------------------------------------------------------------------------
# Experiment composer (raw builds() — caller passes every slot explicitly).
# Most users will go through ``experiments._make.experiment`` which
# wraps this with shape-baking convenience.
# ---------------------------------------------------------------------------

ExperimentC = builds(Experiment, populate_full_signature=True)


__all__ = [
    # Mixer / residual-block builders (instantiate to runtime dataclasses)
    "TimeMixer", "FeatureMixer", "ResidualBlock", "DiffResidualBlock",
    # Schedules
    "Schedule", "ScheduleV2",
    # Architectural builders
    "Head", "Context", "MLPContext", "Unet", "MLPUnet",
    "GRUFutSum", "TransformerFutSum",
    # Module-slot builders
    "Encoder", "Decoder", "ZInit",
    "GaussTransition", "DiffTransition", "DiffV2Transition",
    # Data modules
    "Synthetic", "KDD", "Null",
    # Model + training
    "DDSSM", "Hparams", "Rewo",
    "Trainable", "Training", "Objective",
    "TrainerPartial",
    # Eval / viz / variance
    "Eval", "Plot", "Viz",
    "Probe", "ProbeCellB", "ProbeMetric", "ProbePlot",
    # Experiment composer
    "ExperimentC",
]
