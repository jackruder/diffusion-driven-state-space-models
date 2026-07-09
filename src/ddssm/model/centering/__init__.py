"""Baseline-centering machinery for the model-v2 DDSSM redesign.

This package collects the modules introduced by ``model-v2.org``'s
"Baseline Centering" section: a parameter-free deterministic baseline
``μ_p(z_{t-1})`` (Zero / Persistence, with a fixed unit prior variance
``σ_p² = 1``) and an EMA buffer tracking the per-step centered-residual
variance ``σ_data²(t)``.

These pieces are pure leaves: they do not import any DDSSM transition
or model module, and are unit-testable in isolation.  The diffusion
transition (:mod:`ddssm.model.transitions.diffusion`) and
:class:`ddssm.model.dssd.DDSSM_base` consume them by reference.
"""

from __future__ import annotations

from ddssm.model.centering.baselines import (
    BaseBaseline,
    ZeroBaseline,
    PersistenceBaseline,
)
from ddssm.model.centering.sigma_data import SigmaDataBuffer

__all__ = [
    "BaseBaseline",
    "PersistenceBaseline",
    "SigmaDataBuffer",
    "ZeroBaseline",
]
