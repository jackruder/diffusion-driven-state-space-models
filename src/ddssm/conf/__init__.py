"""DDSSM Hydra configuration package.

Structure
---------
conf/
    _infra.py        — ZenStore, transition/data-module/trainer Confs,
                       TrainableModulesConf, Stages dataclasses, helpers
    _eval_viz.py     — eval/viz family defaults (SynthEvalConf, etc.)
    experiments/
        component.py — synthetic_gauss, synthetic_diffusion
        synthetic.py — harmonic_*, bimodal_*, robot_*
        kdd.py       — kdd_gauss, kdd_diffusion

Importing this package (``import ddssm.conf``) is sufficient to register
all experiment presets with Hydra's ConfigStore.  All public names from
``_infra``, ``_eval_viz``, and the ``experiments`` subpackage are
re-exported here so that external code using ``from ddssm.conf import X``
continues to work without change.
"""

from __future__ import annotations

# Re-export upstream Confs that callers import via ``ddssm.conf``
from ..dssd import REWOConf, DDSSMConf, DDSSMHyperParamsConf
from ..train import DDSSMTrainerConf

# 1. Infrastructure (creates ``store`` and registers transition/model/trainer groups)
from ._infra import (
    StagesConf,
    StageLrsConf,
    StageSpecConf,
    LambdaRampConf,
    KDDDataModuleConf,
    ObjectiveSpecConf,
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
    DecoderGaussianMLPConf,
    EncoderGaussianMLPConf,
    TrainableReconOnlyConf,
    TrainableTransOnlyConf,
    TransitionGaussianConf,
    SyntheticDataModuleConf,
    TransitionDiffusionConf,
    InitPriorGaussianMLPConf,
    TransitionDiffusionV2Conf,
    TransitionGaussianMLPConf,
    TransitionDiffusionMLPConf,
    TransitionDiffusionV2MLPConf,
    store,
    _experiment_conf,
    load_yaml_config,
)

# 2. Eval/viz family defaults
from ._eval_viz import (
    KDDVizConf,
    KDDEvalConf,
    SynthVizConf,
    SynthEvalConf,
    BimodalVizConf,
    Robot2DVizConf,
    BimodalEvalConf,
    HarmonicVizConf,
    Robot2DEvalConf,
    HarmonicEvalConf,
)

# 3. Experiment presets (triggers all store(...) registrations)
from .experiments import (
    BimodalExperimentConf,
    Robot2DExperimentConf,
    HarmonicExperimentConf,
    KDDGaussExperimentConf,
    KDDDiffusionExperimentConf,
    SyntheticGaussExperimentConf,
    SyntheticDiffusionExperimentConf,
)

# 4. Materialise all registered configs into Hydra's ConfigStore.
#    This must run after every store(...) call in the experiments subpackage.
store.add_to_hydra_store(overwrite_ok=True)


__all__ = [
    # Re-exported upstream Confs
    "DDSSMConf",
    "DDSSMHyperParamsConf",
    "DDSSMTrainerConf",
    "DDSSMTrainerPartial",
    "REWOConf",
    # Transitions
    "TransitionGaussianConf",
    "TransitionGaussianMLPConf",
    "TransitionDiffusionConf",
    "TransitionDiffusionMLPConf",
    "TransitionDiffusionV2Conf",
    "TransitionDiffusionV2MLPConf",
    # Encoder / Decoder / InitPrior groups
    "EncoderGaussianConf",
    "EncoderGaussianMLPConf",
    "DecoderGaussianConf",
    "DecoderGaussianMLPConf",
    "InitPriorGaussianConf",
    "InitPriorGaussianMLPConf",
    # Data modules
    "NullDataModuleConf",
    "SyntheticDataModuleConf",
    "KDDDataModuleConf",
    # Experiment building blocks
    "TrainableModulesConf",
    "TrainableJointConf",
    "TrainableReconOnlyConf",
    "TrainableTransOnlyConf",
    "TrainingScalarsConf",
    "ObjectiveSpecConf",
    # Eval/viz family defaults
    "SynthEvalConf",
    "SynthVizConf",
    "KDDEvalConf",
    "KDDVizConf",
    "HarmonicEvalConf",
    "HarmonicVizConf",
    "BimodalEvalConf",
    "BimodalVizConf",
    "Robot2DEvalConf",
    "Robot2DVizConf",
    # Component / smoke-test experiments
    "SyntheticGaussExperimentConf",
    "SyntheticDiffusionExperimentConf",
    # KDD experiments
    "KDDGaussExperimentConf",
    "KDDDiffusionExperimentConf",
    # synthetic confs
    "HarmonicExperimentConf",
    "BimodalExperimentConf",
    "Robot2DExperimentConf",
    # Stages dataclasses
    "StageLrsConf",
    "StageTrainableConf",
    "StageSchedulerConf",
    "LambdaRampConf",
    "StageSpecConf",
    "StagesConf",
    # Utilities
    "load_yaml_config",
    "store",
    "_experiment_conf",
]
