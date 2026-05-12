"""z-init priors for the synthetic-data family.

Architectural knobs (context producers + Gaussian heads for the prior
and aux-posterior) live in :mod:`experiments.synthetic.arch` so they're
visible at the experiment site.
"""

from __future__ import annotations

from ddssm.builders import ZInit

from conf.registry import z_init_store

from experiments.synthetic.arch import SmallContext, SmallHead


Small1D = ZInit(
    latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
    hidden_dim=64, pad_mask_emb_dim=8,
    context=SmallContext, aux_context=SmallContext,
    gaussian_head=SmallHead, aux_posterior_head=SmallHead,
)

Robot2D = ZInit(
    latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
    hidden_dim=64, pad_mask_emb_dim=8,
    context=SmallContext, aux_context=SmallContext,
    gaussian_head=SmallHead, aux_posterior_head=SmallHead,
)

ProbeMedium = ZInit(
    latent_dim=8, j=1, emb_time_dim=16, covariate_dim=0,
    hidden_dim=64, pad_mask_emb_dim=8,
    context=SmallContext, aux_context=SmallContext,
    gaussian_head=SmallHead, aux_posterior_head=SmallHead,
)

z_init_store(Small1D, name="small_1d")
z_init_store(Robot2D, name="robot2d")
z_init_store(ProbeMedium, name="probe_medium")
