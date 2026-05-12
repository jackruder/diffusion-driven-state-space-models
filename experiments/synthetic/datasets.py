"""Synthetic data-module configs (closed-form modes).

Generated on the fly by :class:`SyntheticDataModule`; every dataset
here shares ``T=32`` and ``batch_size=32``.
"""

from __future__ import annotations

from ddssm.builders import Synthetic

from conf.registry import data_store


LGSSM = Synthetic(mode="lgssm", D=1, T=32, N_per_split=512, batch_size=32)
Harmonic = Synthetic(mode="harmonic", D=1, T=32, N_per_split=1024, batch_size=32)
Bimodal = Synthetic(mode="bimodal", D=1, T=32, N_per_split=1024, batch_size=32)
BimodalNoisy = Synthetic(mode="bimodal-noisy", D=1, T=32,
                         N_per_split=1024, batch_size=32)
Robot2D = Synthetic(mode="robot-basis-pursuit", D=2, T=32,
                    N_per_split=1024, batch_size=32)

data_store(LGSSM, name="lgssm")
data_store(Harmonic, name="harmonic")
data_store(Bimodal, name="bimodal")
data_store(BimodalNoisy, name="bimodal_noisy")
data_store(Robot2D, name="robot2d")
