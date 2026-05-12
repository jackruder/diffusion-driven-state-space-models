"""KDD transitions (Gaussian + Diffusion).

The diffusion variant uses a full CSDI U-Net at the default channel
count (64 channels, 4 layers, 128-dim time embedding) with the
:class:`~experiments.kdd.arch.KDDDiffResBlock` residual block. The
Gaussian variant uses the same context producer as the encoder/decoder.
"""

from __future__ import annotations

from ddssm.builders import DiffTransition, GaussTransition, Schedule, Unet

from conf.registry import transition_store

from experiments.kdd.arch import (
    KDDContext, KDDDiffResBlock, KDDPlainHead,
)


_UNET = Unet(
    channels=64,
    n_layers=4,
    embedding_dim=128,
    residual_block=KDDDiffResBlock,
)
_SCHEDULE = Schedule()

KDDGauss = GaussTransition(
    latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
    hidden_dim=64,
    context=KDDContext,
    gaussian_head=KDDPlainHead,
)
KDDDiff = DiffTransition(
    latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
    unet=_UNET, schedule=_SCHEDULE,
)

transition_store(KDDGauss, name="kdd_gauss")
transition_store(KDDDiff, name="kdd_diff")
