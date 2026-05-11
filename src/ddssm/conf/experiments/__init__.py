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

from . import kdd, component, synthetic
from .kdd import KDDGaussExperimentConf, KDDDiffusionExperimentConf
from .component import SyntheticGaussExperimentConf, SyntheticDiffusionExperimentConf
from .synthetic import (
    BimodalExperimentConf,
    Robot2DExperimentConf,
    HarmonicExperimentConf,
)

__all__ = [
    "BimodalExperimentConf",
    "HarmonicExperimentConf",
    "KDDDiffusionExperimentConf",
    "KDDGaussExperimentConf",
    "Robot2DExperimentConf",
    "SyntheticDiffusionExperimentConf",
    "SyntheticGaussExperimentConf", "component", "kdd", "synthetic",
]
