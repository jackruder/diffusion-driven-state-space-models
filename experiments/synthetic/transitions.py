"""Transitions for the synthetic-data family.

Four variants: Gaussian + Diffusion, in each of the 1D and 2D shapes.
Architectural knobs (context, head) live in :mod:`experiments.synthetic.arch`.
"""

from __future__ import annotations

from ddssm.builders import DiffTransition, GaussTransition, Head

from conf.registry import transition_store

from experiments.synthetic.arch import SmallContext
from experiments.synthetic.schedules import Default as DefaultSchedule
from experiments.synthetic.unets import CSDI, MLPTiny


# Gauss transition uses the unclamped head — its variance is regularised
# implicitly by the encoder/decoder KL trade-off.
_GAUSS_HEAD = Head()


Gauss1D = GaussTransition(
    latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
    hidden_dim=64,
    context=SmallContext,
    gaussian_head=_GAUSS_HEAD,
)
# Diff1D uses the tiny MLP score-net so its size stays close to
# Gauss1D — Gauss-vs-Diff comparisons should differ in the *modelling*
# story (score net vs. Gaussian head), not in raw capacity. Swap to
# ``CSDI`` for an apples-to-CSDI comparison via override.
Diff1D = DiffTransition(
    latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
    unet=MLPTiny, schedule=DefaultSchedule,
)
GaussRobot2D = GaussTransition(
    latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
    hidden_dim=64,
    context=SmallContext,
    gaussian_head=_GAUSS_HEAD,
)
# Robot 2D keeps the full CSDI U-Net — real spatial structure benefits
# from the conv backbone.
DiffRobot2D = DiffTransition(
    latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
    unet=CSDI, schedule=DefaultSchedule,
)

transition_store(Gauss1D, name="gauss_1d")
transition_store(Diff1D, name="diff_1d")
transition_store(GaussRobot2D, name="gauss_robot2d")
transition_store(DiffRobot2D, name="diff_robot2d")
