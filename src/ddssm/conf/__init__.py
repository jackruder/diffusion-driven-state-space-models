"""Public DDSSM configuration surface.

Importing :mod:`ddssm.conf` registers the Hydra config groups.  Experiment
presets live in ``ddssm.conf.experiments.*`` so source-block configs can import
only the preset module they need.
"""

from __future__ import annotations

from ..dssd import DDSSMConf, DDSSMHyperParamsConf, REWOConf
from ..train import DDSSMTrainerConf
from ._infra import (
    ContextProducerCSDIConf,
    CSDIUnetGroupConf,
    DDSSMTrainerPartial,
    DecoderGaussianConf,
    EncoderGaussianConf,
    FeatureMixerTransformerConf,
    InitPriorGaussianConf,
    KDDDataModuleConf,
    NullDataModuleConf,
    ObjectiveSpecConf,
    StageLrsConf,
    StageSchedulerConf,
    StageSpecConf,
    StageTrainableConf,
    StagesConf,
    SyntheticDataModuleConf,
    TimeMixerConvConf,
    TrainableJointConf,
    TrainableModulesConf,
    TrainableReconOnlyConf,
    TrainableTransOnlyConf,
    TrainingScalarsConf,
    TransitionDiffusionConf,
    TransitionDiffusionV2Conf,
    TransitionGaussianConf,
    build_experiment_conf,
    store,
)
from ._registry import register_configs

register_configs()

__all__ = [
    "DDSSMConf",
    "DDSSMHyperParamsConf",
    "DDSSMTrainerConf",
    "DDSSMTrainerPartial",
    "REWOConf",
    "TransitionGaussianConf",
    "TransitionDiffusionConf",
    "TransitionDiffusionV2Conf",
    "ContextProducerCSDIConf",
    "CSDIUnetGroupConf",
    "TimeMixerConvConf",
    "FeatureMixerTransformerConf",
    "EncoderGaussianConf",
    "DecoderGaussianConf",
    "InitPriorGaussianConf",
    "NullDataModuleConf",
    "SyntheticDataModuleConf",
    "KDDDataModuleConf",
    "TrainableModulesConf",
    "TrainableJointConf",
    "TrainableReconOnlyConf",
    "TrainableTransOnlyConf",
    "TrainingScalarsConf",
    "ObjectiveSpecConf",
    "StageLrsConf",
    "StageTrainableConf",
    "StageSchedulerConf",
    "StageSpecConf",
    "StagesConf",
    "build_experiment_conf",
    "register_configs",
    "store",
]
