"""DiffusionV2 transitions for variance-probe experiments."""

from __future__ import annotations

from ddssm.builders import DiffV2Transition, Unet

from conf.registry import transition_store

from experiments.variance_probe.schedules import V2


# Shared CSDI U-Net (no need to register a separate unet for these).
_UNET = Unet()

DiffV2_1D = DiffV2Transition(
    latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
    unet=_UNET, schedule=V2,
)

DiffV2_Medium = DiffV2Transition(
    latent_dim=8, j=1, emb_time_dim=16, covariate_dim=0,
    unet=_UNET, schedule=V2,
)

transition_store(DiffV2_1D, name="diffv2_1d")
transition_store(DiffV2_Medium, name="diffv2_medium")
