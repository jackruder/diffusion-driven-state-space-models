"""Centralized hydra-zen builders for every model part.

One place to import from when assembling an :class:`Experiment` config
in a notebook, org src block, or Python script. Each name here is a
``builds(...)`` config — call it like a function with keyword
overrides::

    from ddssm.experiment.builders import (
        DiffTransition,
        DiffSchedule,
        Unet,
    )

    t = DiffTransition(
        unet=Unet(
            channels=64,
            n_layers=4,
        ),
        schedule=DiffSchedule(
            S_k=20
        ),
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

from ddssm.adapters import DDSSMAdapter
from ddssm.nn.futsum import GRUFutureSummary, TransformerFutureSummary
from ddssm.data.mocap import MocapDataModule
from ddssm.experiment import (
    SBatch as _SBatchDC,
    Experiment,
    Objectives as _ObjectivesDC,
    ObjectiveSpec,
    TrainingScalars,
)
from ddssm.model.dssd import DDSSM_base, DDSSMHyperParamsConf  # dataclasses
from ddssm.nn.fusions import DKSFusion, GatedFusion, ConcatLinearFusion
from ddssm.viz.runner import VizSpec, PlotSpec
from ddssm.eval.runner import EvalSpec
from ddssm.nn.diffnets import (
    CSDIUnet,
    MLPCSDIUnet,
    ContextProducer,
    TimeMixerConfig,
    FeatureMixerConfig,
    MLPContextProducer,
    ResidualBlockConfig,
    DiffResidualBlockConfig,
)
from ddssm.nn.combiners import CompoundCombiner
from ddssm.nn.gaussians import GaussianHead
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder
from ddssm.nn.dist_heads import GaussianDistHead

# Runtime classes — actual constructors targeted by ``builds()``.
from ddssm.nn.aggregators import (
    GRUAggregator,
    MLPAggregator,
    IdentityAggregator,
    AttentionAggregator,
    ContextProducerAggregator,
)
from ddssm.training.train import DDSSMTrainer
from ddssm.training.stages import TrainableConf
from ddssm.data.datamodule import (
    KDDDataModule,
    NullDataModule,
    GluonTSDataModule,
    SyntheticDataModule,
)
from ddssm.variance.runner import ProbeCell, ProbeSpec, ProbePlotSpec, ProbeMetricSpec
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.model.centering.baselines import (
    ZeroBaseline,
    PersistenceBaseline,
)
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)
from ddssm.model.transitions.transitions import GaussianTransition

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

DiffSchedule = DiffusionScheduleConfig

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


# ---------------------------------------------------------------------------
# Encoder aggregators / fusions / distribution heads.
#
# An encoder is built from three slots:
#   combiner = CompoundCombiner(aggregator=..., fusion=...)
#   dist_head = GaussianDistHead
# Each builder is ``zen_partial=True`` so the encoder (or another module)
# supplies shape kwargs at construction time.
# ---------------------------------------------------------------------------

IdentityAggregatorB = builds(
    IdentityAggregator,
    populate_full_signature=True,
    zen_partial=True,
)

GRUAggregatorB = builds(
    GRUAggregator,
    num_gru_layers=1,
    populate_full_signature=True,
    zen_partial=True,
)

MLPAggregatorB = builds(
    MLPAggregator,
    num_layers=2,
    populate_full_signature=True,
    zen_partial=True,
)

AttentionAggregatorB = builds(
    AttentionAggregator,
    nheads=4,
    num_attn_layers=1,
    ff_mult=4,
    dropout=0.0,
    populate_full_signature=True,
    zen_partial=True,
)

ContextAggregatorB = builds(
    ContextProducerAggregator,
    channels=8,
    num_layers=2,
    residual_block=ResidualBlock(),
    populate_full_signature=True,
    zen_partial=True,
)

ConcatLinearFusionB = builds(
    ConcatLinearFusion,
    populate_full_signature=True,
    zen_partial=True,
)

DKSFusionB = builds(
    DKSFusion,
    populate_full_signature=True,
    zen_partial=True,
)

GatedFusionB = builds(
    GatedFusion,
    populate_full_signature=True,
    zen_partial=True,
)


def Combiner(*, aggregator, fusion=None):
    """Compose an aggregator + fusion into a ``CompoundCombiner`` partial."""
    if fusion is None:
        fusion = ConcatLinearFusionB()
    return builds(
        CompoundCombiner,
        aggregator=aggregator,
        fusion=fusion,
        populate_full_signature=True,
        zen_partial=True,
    )


GaussianDistHeadB = builds(
    GaussianDistHead,
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
# Encoder / Decoder / Transition builders.
# Shapes are caller-supplied; the inner Context/Head/etc. default to the
# builders above so a one-liner like ``Encoder(data_dim=1, latent_dim=4,
# j=1, emb_time_dim=16, use_mask=False)`` is fully instantiable.
# ---------------------------------------------------------------------------

# All shape-related kwargs are MISSING by default so
# ``experiments._make.experiment`` can fill them in one place.

_SHAPE_ENC = dict(
    data_dim=MISSING,
    latent_dim=MISSING,
    j=MISSING,
    emb_time_dim=MISSING,
    covariate_dim=MISSING,
    use_mask=MISSING,
)
_SHAPE_DEC = dict(
    data_dim=MISSING,
    latent_dim=MISSING,
    j=MISSING,
    emb_time_dim=MISSING,
    covariate_dim=MISSING,
)
_SHAPE_LAT = dict(
    latent_dim=MISSING,
    j=MISSING,
    emb_time_dim=MISSING,
    covariate_dim=MISSING,
)

Encoder = builds(
    GaussianEncoder,
    populate_full_signature=True,
    **_SHAPE_ENC,
    combiner=Combiner(aggregator=ContextAggregatorB(), fusion=ConcatLinearFusionB()),
    dist_head=GaussianDistHeadB(),
    fut_summary=GRUFutSum(),
)

Decoder = builds(
    GaussianDecoder,
    populate_full_signature=True,
    **_SHAPE_DEC,
    context=Context(),
    gaussian_head=Head(),
)

GaussTransition = builds(
    GaussianTransition,
    populate_full_signature=True,
    **_SHAPE_LAT,
    context=Context(),
    gaussian_head=Head(),
)

# ---------------------------------------------------------------------------
# Model-v2 baseline-centering builders.
# ---------------------------------------------------------------------------

ZeroBaselineB = builds(
    ZeroBaseline,
    populate_full_signature=True,
    latent_dim=MISSING,
    j=MISSING,
)
PersistenceBaselineB = builds(
    PersistenceBaseline,
    populate_full_signature=True,
    latent_dim=MISSING,
    j=MISSING,
)
AuxPosteriorB = builds(
    AuxPosterior,
    populate_full_signature=True,
    latent_dim=MISSING,
    j=MISSING,
)
SigmaDataBufferB = builds(
    SigmaDataBuffer,
    populate_full_signature=True,
    T_max=MISSING,
)
DiffTransition = builds(
    DiffusionTransition,
    populate_full_signature=True,
    baseline=MISSING,
    latent_dim=MISSING,
    j=MISSING,
    emb_time_dim=MISSING,
    T_max=MISSING,
    unet=Unet(),
    schedule=DiffSchedule(),
)


# ---------------------------------------------------------------------------
# Data modules.
# ---------------------------------------------------------------------------

Synthetic = builds(SyntheticDataModule, populate_full_signature=True)
KDD = builds(KDDDataModule, populate_full_signature=True)
GluonTS = builds(GluonTSDataModule, populate_full_signature=True)
Mocap = builds(MocapDataModule, populate_full_signature=True)
Null = builds(NullDataModule, populate_full_signature=True)


# ---------------------------------------------------------------------------
# Model, hyperparameters, training.
# ``DDSSM`` takes already-instantiated encoder/decoder/transition
# Confs from the caller; no interpolation.
# ---------------------------------------------------------------------------

Hparams = builds(DDSSMHyperParamsConf, populate_full_signature=True)

Training = builds(
    TrainingScalars,
    populate_full_signature=True,
)
Objective = builds(ObjectiveSpec, populate_full_signature=True)
Objectives = builds(_ObjectivesDC, populate_full_signature=True)
SBatch = builds(_SBatchDC, populate_full_signature=True)

DDSSM = builds(
    DDSSM_base,
    populate_full_signature=True,
    data_dim=MISSING,
    latent_dim=MISSING,
    j=MISSING,
)

# ``build_trainer`` is a partial: the experiment fills in model/device/
# logging paths at run time.
TrainerPartial = builds(DDSSMTrainer, populate_full_signature=True, zen_partial=True)

# Optional freeze/unfreeze mask, plugged into ``Training(trainable=Trainable(...))``.
Trainable = builds(TrainableConf, populate_full_signature=True)


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

# Adapter wrapper conf. ``experiments._make.experiment`` wraps a DDSSM model
# conf in this so ``Experiment.model`` is a ``ModelAdapter`` (not a bare
# ``DDSSM_base``). ``module`` / ``config`` / ``build_trainer`` are curried in
# by that factory; the import edge is one-directional (adapters never import
# experiment at runtime), so this cannot cycle.
DDSSMAdapterC = builds(DDSSMAdapter, populate_full_signature=True)


__all__ = [
    # Mixer / residual-block builders (instantiate to runtime dataclasses)
    "TimeMixer",
    "FeatureMixer",
    "ResidualBlock",
    "DiffResidualBlock",
    # Schedules
    "DiffSchedule",
    # Architectural builders
    "Head",
    "Context",
    "MLPContext",
    "Unet",
    "MLPUnet",
    "GRUFutSum",
    "TransformerFutSum",
    # Encoder building blocks: aggregator + fusion + dist-head
    "IdentityAggregatorB",
    "GRUAggregatorB",
    "MLPAggregatorB",
    "AttentionAggregatorB",
    "ContextAggregatorB",
    "ConcatLinearFusionB",
    "DKSFusionB",
    "GatedFusionB",
    "Combiner",
    "GaussianDistHeadB",
    # Module-slot builders
    "Encoder",
    "Decoder",
    "GaussTransition",
    # Model-v2 baseline-centering builders
    "ZeroBaselineB",
    "PersistenceBaselineB",
    "AuxPosteriorB",
    "SigmaDataBufferB",
    "DiffTransition",
    # Data modules
    "Synthetic",
    "KDD",
    "GluonTS",
    "Null",
    # Model + training
    "DDSSM",
    "Hparams",
    "Training",
    "Objective",
    "Objectives",
    "SBatch",
    "TrainerPartial",
    "Trainable",
    # Eval / viz / variance
    "Eval",
    "Plot",
    "Viz",
    "Probe",
    "ProbeCellB",
    "ProbeMetric",
    "ProbePlot",
    # Experiment composer
    "ExperimentC",
    "DDSSMAdapterC",
]
