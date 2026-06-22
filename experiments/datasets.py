"""Register the library dataset presets into the Hydra ``data_store``.

The dataset configs themselves are library code in
:mod:`ddssm.data.presets`; this module only publishes them to the
``data`` config group. The store is packaged at ``experiment.data``
(see :mod:`ddssm.experiment.stores`), so ``+data=NAME`` overrides the dataset baked
into the chosen experiment preset, e.g.::

    python -m ddssm.app experiment=init_smoke_simple +data=harmonic

(use ``+data=`` to append; a bare ``data=`` errors because ``data`` isn't
in the defaults list). Imported by :mod:`experiments` before the
experiment families.
"""

from __future__ import annotations

from ddssm.data.presets import (
    Taxi,
    Wiki,
    LGSSM,
    Solar,
    Bimodal,
    Robot2D,
    Harmonic,
    Traffic,
    KDDFull,
    KDDBeijing,
    KDDStation,
    Electricity,
    BimodalNoisy,
    NonlinBimodalLift1D,
    NonlinBimodalLiftMV,
    LorenzDirect,
)
from ddssm.experiment.stores import data_store
from experiments.lorenz.data import LorenzDirect

# Synthetic (sequence-format) datasets.
data_store(LGSSM, name="lgssm")
data_store(Harmonic, name="harmonic")
data_store(Bimodal, name="bimodal")
data_store(BimodalNoisy, name="bimodal_noisy")
data_store(Robot2D, name="robot2d")
data_store(NonlinBimodalLift1D, name="nonlin_bimodal_lift_1d")
data_store(NonlinBimodalLiftMV, name="nonlin_bimodal_lift_mv")
# Synthetic (Lorenz) datasets.
data_store(LorenzDirect, name="lorenz")

# GluonTS repository datasets (windowed; fetched lazily on first loader access).
data_store(Solar, name="solar")
data_store(Electricity, name="electricity")
data_store(Traffic, name="traffic")
data_store(Taxi, name="taxi")
data_store(Wiki, name="wiki")

# KDD Cup 2018 PM2.5 (windowed; preprocessed .pt payloads under data/).
data_store(KDDFull, name="kdd")
data_store(KDDBeijing, name="kdd_beijing")
data_store(KDDStation, name="kdd_station")
