"""Baseline-centering machinery for the model-v2 DDSSM redesign.

This package collects the modules introduced by ``model-v2.org``'s
"Baseline Centering" section: a deterministic baseline ``μ_p(z_{t-1})``
(four forms — Zero / Identity / Linear / MLP), an EMA buffer tracking
the per-step centered-residual variance ``σ_data²(t)``, regularizers
on ``log σ_p²`` and the baseline-anchor, and the stage-1 → stage-2
handoff.

These pieces are pure leaves: they do not import any DDSSM transition
or model module, and are unit-testable in isolation.  The
transitions (:mod:`ddssm.transitions.baseline_gaussian`,
:mod:`ddssm.transitions.diffusion_v3`) and :class:`ddssm.dssd.DDSSM_base`
consume them by reference.
"""

from __future__ import annotations

from .baselines import (
    BaseBaseline,
    IdentityBaseline,
    LinearBaseline,
    MLPBaseline,
    ZeroBaseline,
)
from .handoff import CenteringHandoffConf, perform_centering_handoff
from .regularizers import r_mu_p_loss, r_sigma_p_loss
from .sigma_data import SigmaDataBuffer

__all__ = [
    "BaseBaseline",
    "CenteringHandoffConf",
    "IdentityBaseline",
    "LinearBaseline",
    "MLPBaseline",
    "SigmaDataBuffer",
    "ZeroBaseline",
    "perform_centering_handoff",
    "r_mu_p_loss",
    "r_sigma_p_loss",
]
