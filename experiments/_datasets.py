"""Named data-module configs — one per dataset.

Registered under ``group="data"``. Reach them via
``python -m ddssm.app data=<name>`` or
``from experiments._datasets import Harmonic``.

Defined in :file:`verifications.org`; tangled here.
"""

from __future__ import annotations

from ddssm.builders import KDD, Synthetic

from experiments._registry import data_store


# ---------------------------------------------------------------------------
# 1D synthetic time series — generated on the fly from a closed-form mode.
# ---------------------------------------------------------------------------

LGSSM = Synthetic(mode="lgssm", D=1, T=64, N_per_split=512, batch_size=32)
Harmonic = Synthetic(mode="harmonic", D=1, T=64, N_per_split=1024, batch_size=32)
Bimodal = Synthetic(mode="bimodal", D=1, T=64, N_per_split=1024, batch_size=32)
BimodalNoisy = Synthetic(mode="bimodal-noisy", D=1, T=64, N_per_split=1024, batch_size=32)


# ---------------------------------------------------------------------------
# 2D robot trajectories.
# ---------------------------------------------------------------------------

Robot2D = Synthetic(mode="robot-basis-pursuit", D=2, T=64,
                    N_per_split=1024, batch_size=32)


# ---------------------------------------------------------------------------
# Lift to D=4 (nonlinear-bimodal-lift for variance probe).
# ---------------------------------------------------------------------------

NonlinearBimodalLift = Synthetic(
    mode="nonlinear-bimodal-lift", D=4, T=64,
    N_per_split=256, batch_size=32,
)


# ---------------------------------------------------------------------------
# Variance-probe versions of the 1D modes (smaller dataset, fewer epochs).
# ---------------------------------------------------------------------------

ProbeLGSSM = Synthetic(mode="lgssm", D=1, T=64, N_per_split=256, batch_size=32)
ProbeBimodal = Synthetic(mode="bimodal", D=1, T=64, N_per_split=256, batch_size=32)
ProbeBimodalNoisy = Synthetic(mode="bimodal-noisy", D=1, T=64,
                              N_per_split=256, batch_size=32)


# ---------------------------------------------------------------------------
# Real benchmark: KDD Cup 2018 PM2.5.
# ---------------------------------------------------------------------------

KDDData = KDD(batch_size=128, eval_step_size=24)


data_store(LGSSM, name="lgssm")
data_store(Harmonic, name="harmonic")
data_store(Bimodal, name="bimodal")
data_store(BimodalNoisy, name="bimodal_noisy")
data_store(Robot2D, name="robot2d")
data_store(NonlinearBimodalLift, name="nonlinear_bimodal_lift")
data_store(ProbeLGSSM, name="probe_lgssm")
data_store(ProbeBimodal, name="probe_bimodal")
data_store(ProbeBimodalNoisy, name="probe_bimodal_noisy")
data_store(KDDData, name="kdd")


__all__ = [
    "LGSSM", "Harmonic", "Bimodal", "BimodalNoisy",
    "Robot2D", "NonlinearBimodalLift",
    "ProbeLGSSM", "ProbeBimodal", "ProbeBimodalNoisy",
    "KDDData",
]
