"""Composed model configs for the variance-probe family.

Variance-probe runs measure score-net variance on toy synthetic data:
the diffusion net's only job is to be the *subject* of the
measurement, not to extract maximum capacity. We reuse the synthetic
family's encoder / decoder / z_init (shape ``Small1D`` or
``ProbeMedium``) and the tiny MLP score-net, swapping in DiffusionV2's
noise schedule.

Edit one shape constant in the corresponding namespace class below and
the change propagates into the transition + DDSSM compositions.
"""

from __future__ import annotations

from ddssm.builders import DDSSM, DiffV2Transition, Hparams, ScheduleV2

from conf.registry import model_store, schedule_store, transition_store

from experiments.synthetic.model import MLPTiny, ProbeMedium, Small1D


# ---------------------------------------------------------------------------
# DiffusionV2 schedule.
# ---------------------------------------------------------------------------

V2 = ScheduleV2()


# ---------------------------------------------------------------------------
# Shape: ProbeSmall (D=1, latent_dim=4, j=1).
#
# Reuses the synthetic ``Small1D`` encoder/decoder/z_init shape.
# ---------------------------------------------------------------------------


class ProbeSmall:
    data_dim = 1
    latent_dim = 4
    j = 1
    emb_time_dim = 16
    covariate_dim = 0

    transition = DiffV2Transition(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        unet=MLPTiny, schedule=V2,
    )

    model = DDSSM(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        encoder=Small1D.encoder, decoder=Small1D.decoder, z_init=Small1D.z_init,
        transition=transition,
        hyperparams=Hparams(),
        use_observation_mask=False,
    )


# ---------------------------------------------------------------------------
# Shape: ProbeMedium (D=4, latent_dim=8, j=1, nonlinear-bimodal-lift).
#
# Reuses the synthetic ``ProbeMedium`` encoder/decoder/z_init shape.
# ---------------------------------------------------------------------------


class ProbeMediumModel:
    data_dim = 4
    latent_dim = 8
    j = 1
    emb_time_dim = 16
    covariate_dim = 0

    transition = DiffV2Transition(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        unet=MLPTiny, schedule=V2,
    )

    model = DDSSM(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        encoder=ProbeMedium.encoder, decoder=ProbeMedium.decoder, z_init=ProbeMedium.z_init,
        transition=transition,
        hyperparams=Hparams(),
        use_observation_mask=False,
    )


# ---------------------------------------------------------------------------
# Store registrations.
# ---------------------------------------------------------------------------

schedule_store(V2, name="v2")
transition_store(ProbeSmall.transition, name="diffv2_1d")
transition_store(ProbeMediumModel.transition, name="diffv2_medium")
model_store(ProbeSmall.model, name="probe_small")
model_store(ProbeMediumModel.model, name="probe_medium")
