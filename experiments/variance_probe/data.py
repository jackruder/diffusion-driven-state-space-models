"""Datasets for variance-probe experiments (smaller ``N_per_split``)."""

from __future__ import annotations

from ddssm.builders import Synthetic

from conf.registry import data_store


T = 32
N_PER_SPLIT = 256
BATCH_SIZE = 32


ProbeLGSSM = Synthetic(
    mode="lgssm", D=1, T=T, N_per_split=N_PER_SPLIT, batch_size=BATCH_SIZE,
)
ProbeBimodal = Synthetic(
    mode="bimodal", D=1, T=T, N_per_split=N_PER_SPLIT, batch_size=BATCH_SIZE,
)
ProbeBimodalNoisy = Synthetic(
    mode="bimodal-noisy", D=1, T=T, N_per_split=N_PER_SPLIT, batch_size=BATCH_SIZE,
)
NonlinearBimodalLift = Synthetic(
    mode="nonlinear-bimodal-lift", D=4, T=T,
    N_per_split=N_PER_SPLIT, batch_size=BATCH_SIZE,
)


data_store(ProbeLGSSM, name="probe_lgssm")
data_store(ProbeBimodal, name="probe_bimodal")
data_store(ProbeBimodalNoisy, name="probe_bimodal_noisy")
data_store(NonlinearBimodalLift, name="nonlinear_bimodal_lift")
