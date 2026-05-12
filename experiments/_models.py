"""Named :class:`DDSSM` configs — one per (shape, transition) combination.

Each model bakes shape (``data_dim``, ``latent_dim``, ``j``,
``emb_time_dim``, ``covariate_dim``) into every leaf and ships with
default :class:`Hparams`. Experiments override hparams when needed via
``dataclasses.replace`` or :func:`experiments._make.override`.

Registered under ``group="model"``. Reach them via
``python -m ddssm.app model=<name>`` or
``from experiments._models import SmallGauss``.

Defined in :file:`verifications.org`; tangled here.
"""

from __future__ import annotations

from ddssm.builders import (
    DDSSM, Decoder, DiffTransition, DiffV2Transition, Encoder,
    GaussTransition, Hparams, Schedule, ScheduleV2, Unet, ZInit,
)

from experiments._registry import model_store


# ---------------------------------------------------------------------------
# 1D synthetic (D=1, d=4, j=1, emb_time_dim=16) — covers
# {lgssm, harmonic, bimodal, bimodal-noisy}.
# ---------------------------------------------------------------------------

def _ddssm_1d(transition):
    return DDSSM(
        data_dim=1, latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
        encoder=Encoder(data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
                        covariate_dim=0, use_mask=False),
        decoder=Decoder(data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
                        covariate_dim=0),
        z_init=ZInit(latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0),
        transition=transition,
        hyperparams=Hparams(),
        use_observation_mask=False,
    )


SmallGauss = _ddssm_1d(GaussTransition(
    latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
))
SmallDiff = _ddssm_1d(DiffTransition(
    latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
    unet=Unet(), schedule=Schedule(),
))
ProbeSmall = _ddssm_1d(DiffV2Transition(
    latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
    unet=Unet(), schedule=ScheduleV2(),
))


# ---------------------------------------------------------------------------
# 2D robot navigation (D=2, d=6, j=2, emb_time_dim=16).
# ---------------------------------------------------------------------------

def _ddssm_robot2d(transition):
    return DDSSM(
        data_dim=2, latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
        encoder=Encoder(data_dim=2, latent_dim=6, j=2, emb_time_dim=16,
                        covariate_dim=0, use_mask=False),
        decoder=Decoder(data_dim=2, latent_dim=6, j=2, emb_time_dim=16,
                        covariate_dim=0),
        z_init=ZInit(latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0),
        transition=transition,
        hyperparams=Hparams(),
        use_observation_mask=False,
    )


Robot2DGauss = _ddssm_robot2d(GaussTransition(
    latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
))
Robot2DDiff = _ddssm_robot2d(DiffTransition(
    latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
    unet=Unet(), schedule=Schedule(),
))


# ---------------------------------------------------------------------------
# KDD Cup 2018 (D=6, d=8, j=1, emb_time_dim=32, covariate_dim=3).
# ---------------------------------------------------------------------------

def _ddssm_kdd(transition):
    return DDSSM(
        data_dim=6, latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
        encoder=Encoder(data_dim=6, latent_dim=8, j=1, emb_time_dim=32,
                        covariate_dim=3, use_mask=False),
        decoder=Decoder(data_dim=6, latent_dim=8, j=1, emb_time_dim=32,
                        covariate_dim=3),
        z_init=ZInit(latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3),
        transition=transition,
        hyperparams=Hparams(),
        use_observation_mask=False,
    )


KDDGauss = _ddssm_kdd(GaussTransition(
    latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
))
KDDDiff = _ddssm_kdd(DiffTransition(
    latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
    unet=Unet(), schedule=Schedule(),
))


# ---------------------------------------------------------------------------
# Variance-probe D=4 (nonlinear-bimodal-lift only).
# ---------------------------------------------------------------------------

ProbeMedium = DDSSM(
    data_dim=4, latent_dim=8, j=1, emb_time_dim=16, covariate_dim=0,
    encoder=Encoder(data_dim=4, latent_dim=8, j=1, emb_time_dim=16,
                    covariate_dim=0, use_mask=False),
    decoder=Decoder(data_dim=4, latent_dim=8, j=1, emb_time_dim=16,
                    covariate_dim=0),
    z_init=ZInit(latent_dim=8, j=1, emb_time_dim=16, covariate_dim=0),
    transition=DiffV2Transition(
        latent_dim=8, j=1, emb_time_dim=16, covariate_dim=0,
        unet=Unet(), schedule=ScheduleV2(),
    ),
    hyperparams=Hparams(),
    use_observation_mask=False,
)


model_store(SmallGauss, name="small_gauss")
model_store(SmallDiff, name="small_diff")
model_store(ProbeSmall, name="probe_small")
model_store(Robot2DGauss, name="robot2d_gauss")
model_store(Robot2DDiff, name="robot2d_diff")
model_store(KDDGauss, name="kdd_gauss")
model_store(KDDDiff, name="kdd_diff")
model_store(ProbeMedium, name="probe_medium")


__all__ = [
    "SmallGauss", "SmallDiff", "ProbeSmall",
    "Robot2DGauss", "Robot2DDiff",
    "KDDGauss", "KDDDiff", "ProbeMedium",
]
