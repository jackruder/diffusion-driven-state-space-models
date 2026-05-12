"""KDD transitions (Gaussian + Diffusion)."""

from __future__ import annotations

from ddssm.builders import DiffTransition, GaussTransition, Schedule, Unet

from conf.registry import transition_store


_UNET = Unet()
_SCHEDULE = Schedule()

KDDGauss = GaussTransition(
    latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
)
KDDDiff = DiffTransition(
    latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
    unet=_UNET, schedule=_SCHEDULE,
)

transition_store(KDDGauss, name="kdd_gauss")
transition_store(KDDDiff, name="kdd_diff")
