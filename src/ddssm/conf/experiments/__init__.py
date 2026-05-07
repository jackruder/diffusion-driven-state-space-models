"""Experiment preset registry.

Importing this package (or any of its submodules) triggers ``store(...)``
registrations for every experiment preset.  ``conf/__init__.py`` imports
this package before calling ``store.add_to_hydra_store()`` so all presets
are materialized into Hydra's ConfigStore in one shot.

Submodules map to ``verifications.org`` sections
-------------------------------------------------
- ``component``  → Component Tests (synthetic_gauss, synthetic_diffusion)
- ``synthetic``  → Synthetic Experiments (harmonic_*, bimodal_*, robot_*)
- ``kdd``        → Real-Data Experiment: KDD Cup 2018 (kdd_gauss, kdd_diffusion)
"""

from __future__ import annotations

from . import component, kdd, synthetic
from .component import SyntheticDiffusionExperimentConf, SyntheticGaussExperimentConf
from .kdd import KDDDiffusionExperimentConf, KDDGaussExperimentConf
from .synthetic import (
    BimodalDiffExperimentConf,
    BimodalGaussExperimentConf,
    HarmonicDiffExperimentConf,
    HarmonicDiffJ2ExperimentConf,
    HarmonicGaussExperimentConf,
    HarmonicGaussJ2ExperimentConf,
    HarmonicNoisyDiffExperimentConf,
    HarmonicNoisyGaussExperimentConf,
    RobotDiff2DExperimentConf,
    RobotGauss2DExperimentConf,
)

__all__ = [
    # Component / smoke-test
    "SyntheticGaussExperimentConf",
    "SyntheticDiffusionExperimentConf",
    # KDD Cup 2018
    "KDDGaussExperimentConf",
    "KDDDiffusionExperimentConf",
    # Harmonic
    "HarmonicGaussExperimentConf",
    "HarmonicDiffExperimentConf",
    "HarmonicGaussJ2ExperimentConf",
    "HarmonicDiffJ2ExperimentConf",
    # Harmonic-noisy
    "HarmonicNoisyGaussExperimentConf",
    "HarmonicNoisyDiffExperimentConf",
    # Bimodal
    "BimodalGaussExperimentConf",
    "BimodalDiffExperimentConf",
    # Robot navigation 2D
    "RobotGauss2DExperimentConf",
    "RobotDiff2DExperimentConf",
]
