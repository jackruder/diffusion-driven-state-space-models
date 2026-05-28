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

from ddssm.builders import Synthetic
from ddssm.data.synthetic import NLBL_MV_OBS_D


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


__all__ = [
    "LGSSM", "Harmonic", "Bimodal", "BimodalNoisy", "Robot2D",
    "NonlinBimodalLift1D", "NonlinBimodalLiftMV",
]
