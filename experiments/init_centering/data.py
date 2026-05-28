"""Data-module configs for the init-centering preset family.

Re-exports the ablation datasets from the library presets
(:mod:`ddssm.data.presets`) — the harder-than-LGSSM synthetic targets
the grilling settled on:

- :data:`NonlinBimodalLift1D` — D=1, latent d=1.
- :data:`NonlinBimodalLiftMV` — D=8, latent d=4.

``Harmonic`` is retained for the smoke presets that point at it.
"""

from __future__ import annotations

from ddssm.data.presets import (
    Harmonic,
    NonlinBimodalLift1D,
    NonlinBimodalLiftMV,
)

__all__ = ["Harmonic", "NonlinBimodalLift1D", "NonlinBimodalLiftMV"]
