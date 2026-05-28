"""Register the library dataset presets into the Hydra ``data_store``.

The dataset configs themselves are library code in
:mod:`ddssm.data.presets`; this module only publishes them to the
``data`` config group so ``data=NAME`` CLI overrides resolve. Imported
by :mod:`experiments` before the experiment families.
"""

from __future__ import annotations

from ddssm.data.presets import (
    LGSSM,
    Harmonic,
    Bimodal,
    BimodalNoisy,
    Robot2D,
    NonlinBimodalLift1D,
    NonlinBimodalLiftMV,
)

from conf.registry import data_store

data_store(LGSSM, name="lgssm")
data_store(Harmonic, name="harmonic")
data_store(Bimodal, name="bimodal")
data_store(BimodalNoisy, name="bimodal_noisy")
data_store(Robot2D, name="robot2d")
data_store(NonlinBimodalLift1D, name="nonlin_bimodal_lift_1d")
data_store(NonlinBimodalLiftMV, name="nonlin_bimodal_lift_mv")
