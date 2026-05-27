"""Synthetic data-module configs (closed-form modes).

Generated on the fly by :class:`SyntheticDataModule`; every dataset
here shares ``T=32`` and ``batch_size=32``. To bump the sequence
length or batch size globally, change them once at the top.
"""

from __future__ import annotations

from ddssm.builders import Synthetic
from ddssm.data.synthetic import NLBL_MV_OBS_D

from conf.registry import data_store


T = 32
BATCH_SIZE = 32
N_PER_SPLIT = 1024
N_PER_SPLIT_LGSSM = 512  # LGSSM smoke runs use a smaller split.


LGSSM = Synthetic(
    mode="lgssm", D=1, T=T,
    N_per_split=N_PER_SPLIT_LGSSM, batch_size=BATCH_SIZE,
)
Harmonic = Synthetic(
    mode="harmonic", D=1, T=T,
    N_per_split=N_PER_SPLIT, batch_size=BATCH_SIZE,
)
Bimodal = Synthetic(
    mode="bimodal", D=1, T=T,
    N_per_split=N_PER_SPLIT, batch_size=BATCH_SIZE,
)
BimodalNoisy = Synthetic(
    mode="bimodal-noisy", D=1, T=T,
    N_per_split=N_PER_SPLIT, batch_size=BATCH_SIZE,
)
Robot2D = Synthetic(
    mode="robot-basis-pursuit", D=2, T=T,
    N_per_split=N_PER_SPLIT, batch_size=BATCH_SIZE,
)
# Init-centering ablation datasets. Both expose GT latents so
# ``gt_latent_jsd`` works headline-side; see
# :mod:`ddssm.eval.synthetic_kernels` for the matching closed-form
# transition kernels.
NonlinBimodalLift1D = Synthetic(
    mode="nonlinear-bimodal-lift", D=1, T=T,
    N_per_split=N_PER_SPLIT, batch_size=BATCH_SIZE,
    expose_gt_latents=True,
)
NonlinBimodalLiftMV = Synthetic(
    mode="nonlinear-bimodal-lift-mv", D=NLBL_MV_OBS_D, T=T,
    N_per_split=N_PER_SPLIT, batch_size=BATCH_SIZE,
    expose_gt_latents=True,
)


data_store(LGSSM, name="lgssm")
data_store(Harmonic, name="harmonic")
data_store(Bimodal, name="bimodal")
data_store(BimodalNoisy, name="bimodal_noisy")
data_store(Robot2D, name="robot2d")
data_store(NonlinBimodalLift1D, name="nonlin_bimodal_lift_1d")
data_store(NonlinBimodalLiftMV, name="nonlin_bimodal_lift_mv")
