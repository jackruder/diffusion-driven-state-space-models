"""Library dataset-module presets (closed-form synthetic modes).

Reusable ``SyntheticDataModule`` configs, generated on the fly. Lives in
the library (not under an experiment family) so any experiment can
import them — the init-centering ablation reuses
``NonlinBimodalLift1D/MV``, and ``verifications.org`` exercises the
harmonic / bimodal / robot modes.

Every dataset shares ``T=32`` and ``batch_size=32``; change them once at
the top. Registration into the Hydra ``data_store`` (so ``data=NAME``
CLI overrides resolve) happens in :mod:`experiments.datasets`.
"""

from __future__ import annotations

from ddssm.data.synthetic import NLBL_MV_OBS_D, PENDULUM_OBS_D
from ddssm.experiment.builders import KDD, Mocap, GluonTS, Synthetic

T = 32
BATCH_SIZE = 32
N_PER_SPLIT = 1024
N_PER_SPLIT_LGSSM = 512  # LGSSM smoke runs use a smaller split.


LGSSM = Synthetic(
    mode="lgssm",
    D=1,
    T=T,
    N_per_split=N_PER_SPLIT_LGSSM,
    batch_size=BATCH_SIZE,
)
Harmonic = Synthetic(
    mode="harmonic",
    D=1,
    T=T,
    N_per_split=N_PER_SPLIT,
    batch_size=BATCH_SIZE,
)
HarmonicMixed = Synthetic(
    mode="harmonic-mixed",
    D=1,
    T=T,
    N_per_split=N_PER_SPLIT,
    batch_size=BATCH_SIZE,
)
Bimodal = Synthetic(
    mode="bimodal",
    D=1,
    T=T,
    N_per_split=N_PER_SPLIT,
    batch_size=BATCH_SIZE,
)
BimodalNoisy = Synthetic(
    mode="bimodal-noisy",
    D=1,
    T=T,
    N_per_split=N_PER_SPLIT,
    batch_size=BATCH_SIZE,
)
Robot2D = Synthetic(
    mode="robot-basis-pursuit",
    D=2,
    T=T,
    N_per_split=N_PER_SPLIT,
    batch_size=BATCH_SIZE,
)
# Init-centering ablation datasets. Both expose GT latents so
# ``gt_latent_jsd`` works headline-side; see
# :mod:`ddssm.eval.synthetic_kernels` for the matching closed-form
# transition kernels.
NonlinBimodalLift1D = Synthetic(
    mode="nonlinear-bimodal-lift",
    D=1,
    T=T,
    N_per_split=N_PER_SPLIT,
    batch_size=BATCH_SIZE,
    expose_gt_latents=True,
)
NonlinBimodalLiftMV = Synthetic(
    mode="nonlinear-bimodal-lift-mv",
    D=NLBL_MV_OBS_D,
    T=T,
    N_per_split=N_PER_SPLIT,
    batch_size=BATCH_SIZE,
    expose_gt_latents=True,
)
# 32×32 rendered stochastic damped pendulum (DVBF/RKN-style benchmark).
# Latent (θ, ω) SDE, emission = Gaussian blob at bob position, flattened
# to D = 1024. GT latents exposed.
Pendulum = Synthetic(
    mode="pendulum",
    D=PENDULUM_OBS_D,
    T=T,
    N_per_split=N_PER_SPLIT,
    batch_size=BATCH_SIZE,
    expose_gt_latents=True,
)

# Real windowed datasets. These keep their own (larger) window/batch defaults
# from the builders — T above applies only to the synthetic sequence modes.
# GluonTS repository datasets are fetched lazily on first loader access.
Solar = GluonTS(name="solar")
Electricity = GluonTS(name="electricity")
Traffic = GluonTS(name="traffic")
Taxi = GluonTS(name="taxi")
Wiki = GluonTS(name="wiki")
# KDD Cup 2018 PM2.5 — preprocessed .pt payloads under ``data/``.
KDDFull = KDD(filepath="data/kdd.pt")
KDDBeijing = KDD(filepath="data/kdd-beijing-pm25.pt")
KDDStation = KDD(filepath="data/kdd-aotizhongxin_aq.pt")

# CMU MoCap subject-35 walking — 16/3/4 sequences × 300 × 50, from the
# Wang-2007 preprocessing shared by Yildiz, Li, Course & Nair, Bartosh.
# ``mocap35.mat`` is fetched on demand into ``data/`` on first construction.
Mocap35 = Mocap(filepath="data/mocap35.mat")


__all__ = [
    "LGSSM",
    "Bimodal",
    "BimodalNoisy",
    "Electricity",
    "Harmonic",
    "HarmonicMixed",
    "KDDBeijing",
    "KDDFull",
    "KDDStation",
    "Mocap35",
    "NonlinBimodalLift1D",
    "NonlinBimodalLiftMV",
    "Pendulum",
    "Robot2D",
    "Solar",
    "Taxi",
    "Traffic",
    "Wiki",
]
