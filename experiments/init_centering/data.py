"""Data-module configs for the init-centering preset family.

Re-exports the two ablation datasets registered in
``experiments.synthetic.data`` (the harder-than-LGSSM synthetic
targets the grilling settled on):

- :data:`NonlinBimodalLift1D` — D=1, latent d=1.
- :data:`NonlinBimodalLiftMV` — D=8, latent d=4.

``Harmonic`` is retained because the legacy ``init_centering_smoke`` /
``init_centering_pilot`` presets still point at it.
"""

from __future__ import annotations

from experiments.synthetic.data import (
    Harmonic,
    NonlinBimodalLift1D,
    NonlinBimodalLiftMV,
)

__all__ = ["Harmonic", "NonlinBimodalLift1D", "NonlinBimodalLiftMV"]
