"""Public DDSSM configuration surface.

Importing :mod:`ddssm.conf` registers the Hydra config groups.  Experiment
presets live in ``ddssm.conf.experiments.*`` so source-block configs can import
only the preset module they need.
"""

from __future__ import annotations

from ..dssd import REWOConf, DDSSMConf, DDSSMHyperParamsConf
from ..train import DDSSMTrainerConf
from ._infra import (
    StagesConf,
    StageLrsConf,
    StageSpecConf,
    CSDIUnetGroupConf,
    KDDDataModuleConf,
    ObjectiveSpecConf,
    TimeMixerConvConf,
    NullDataModuleConf,
    StageSchedulerConf,
    StageTrainableConf,
    TrainableJointConf,
    DDSSMTrainerPartial,
    DecoderGaussianConf,
    EncoderGaussianConf,
    TrainingScalarsConf,
    TrainableModulesConf,
    InitPriorGaussianConf,
    TrainableReconOnlyConf,
    TrainableTransOnlyConf,
    TransitionGaussianConf,
    ContextProducerCSDIConf,
    SyntheticDataModuleConf,
    TransitionDiffusionConf,
    TransitionDiffusionV2Conf,
    FeatureMixerTransformerConf,
    store,
    build_experiment_conf,
)
from ._registry import register_configs

register_configs()  # noqa: RUF067

__all__ = [
    "CSDIUnetGroupConf",
    "ContextProducerCSDIConf",
    "DDSSMConf",
    "DDSSMHyperParamsConf",
    "DDSSMTrainerConf",
    "DDSSMTrainerPartial",
    "DecoderGaussianConf",
    "EncoderGaussianConf",
    "FeatureMixerTransformerConf",
    "InitPriorGaussianConf",
    "KDDDataModuleConf",
    "NullDataModuleConf",
    "ObjectiveSpecConf",
    "REWOConf",
    "StageLrsConf",
    "StageSchedulerConf",
    "StageSpecConf",
    "StageTrainableConf",
    "StagesConf",
    "SyntheticDataModuleConf",
    "TimeMixerConvConf",
    "TrainableJointConf",
    "TrainableModulesConf",
    "TrainableReconOnlyConf",
    "TrainableTransOnlyConf",
    "TrainingScalarsConf",
    "TransitionDiffusionConf",
    "TransitionDiffusionV2Conf",
    "TransitionGaussianConf",
    "build_experiment_conf",
    "register_configs",
    "store",
]
