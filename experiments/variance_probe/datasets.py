"""Datasets for variance-probe experiments (smaller N_per_split)."""

from __future__ import annotations

from ddssm.builders import Synthetic

from conf.registry import data_store


ProbeLGSSM = Synthetic(mode="lgssm", D=1, T=64,
                       N_per_split=256, batch_size=32)
ProbeBimodal = Synthetic(mode="bimodal", D=1, T=64,
                         N_per_split=256, batch_size=32)
ProbeBimodalNoisy = Synthetic(mode="bimodal-noisy", D=1, T=64,
                              N_per_split=256, batch_size=32)
NonlinearBimodalLift = Synthetic(
    mode="nonlinear-bimodal-lift", D=4, T=64,
    N_per_split=256, batch_size=32,
)

data_store(ProbeLGSSM, name="probe_lgssm")
data_store(ProbeBimodal, name="probe_bimodal")
data_store(ProbeBimodalNoisy, name="probe_bimodal_noisy")
data_store(NonlinearBimodalLift, name="nonlinear_bimodal_lift")
