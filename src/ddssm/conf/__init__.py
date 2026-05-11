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

# 1. Infrastructure (creates ``store`` and registers transition/model/trainer groups)
from ._infra import (
    DDSSMTrainerPartial,
    DecoderGaussianConf,
    DecoderGaussianMLPConf,
    EncoderGaussianConf,
    EncoderGaussianMLPConf,
    InitPriorGaussianConf,
    InitPriorGaussianMLPConf,
    KDDDataModuleConf,
    LambdaRampConf,
    NullDataModuleConf,
    ObjectiveSpecConf,
    StageLrsConf,
    StageSchedulerConf,
    StageSpecConf,
    StageTrainableConf,
    StagesConf,
    SyntheticDataModuleConf,
    TrainableJointConf,
    TrainableModulesConf,
    TrainableReconOnlyConf,
    TrainableTransOnlyConf,
    TrainingScalarsConf,
    TransitionDiffusionConf,
    TransitionDiffusionMLPConf,
    TransitionDiffusionV2Conf,
    TransitionDiffusionV2MLPConf,
    TransitionGaussianConf,
    TransitionGaussianMLPConf,
    ProbeCellConf,
    ProbeMetricSpecConf,
    ProbePlotSpecConf,
    ProbeSpecConf,
    _experiment_conf,
    load_yaml_config,
    store,
)

# Re-export upstream Confs that callers import via ``ddssm.conf``
from ..dssd import DDSSMConf, DDSSMHyperParamsConf, REWOConf
from ..train import DDSSMTrainerConf

# 2. Eval/viz family defaults
from ._eval_viz import (
    BimodalEvalConf,
    BimodalVizConf,
    HarmonicEvalConf,
    HarmonicVizConf,
    KDDEvalConf,
    KDDVizConf,
    Robot2DEvalConf,
    Robot2DVizConf,
    SynthEvalConf,
    SynthVizConf,
)
from ._variance import (
    BimodalCleanVarianceConf,
    BimodalNoisyVarianceConf,
    LGSSMVarianceConf,
    NonlinearBimodalLiftVarianceConf,
)

# 3. Experiment presets (triggers all store(...) registrations)
from .experiments import (
    KDDDiffusionExperimentConf,
    KDDGaussExperimentConf,
    SyntheticDiffusionExperimentConf,
    SyntheticGaussExperimentConf,
    HarmonicExperimentConf,
    BimodalExperimentConf,
    Robot2DExperimentConf,
    VarianceProbeBimodalCleanExperimentConf,
    VarianceProbeBimodalNoisyExperimentConf,
    VarianceProbeLGSSMExperimentConf,
    VarianceProbeNonlinearBimodalLiftExperimentConf,
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
    "ProbeCellConf",
    "ProbeMetricSpecConf",
    "ProbePlotSpecConf",
    "ProbeSpecConf",
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
    "LGSSMVarianceConf",
    "BimodalCleanVarianceConf",
    "BimodalNoisyVarianceConf",
    "NonlinearBimodalLiftVarianceConf",
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
    "VarianceProbeLGSSMExperimentConf",
    "VarianceProbeBimodalCleanExperimentConf",
    "VarianceProbeBimodalNoisyExperimentConf",
    "VarianceProbeNonlinearBimodalLiftExperimentConf",
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
