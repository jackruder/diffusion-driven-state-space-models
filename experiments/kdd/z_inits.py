"""KDD z-init prior."""

from __future__ import annotations

from ddssm.builders import ZInit

from conf.registry import z_init_store

from experiments.kdd.arch import KDDClampedHead, KDDContext


KDD = ZInit(
    latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
    hidden_dim=64, pad_mask_emb_dim=8,
    context=KDDContext, aux_context=KDDContext,
    gaussian_head=KDDClampedHead, aux_posterior_head=KDDClampedHead,
)
z_init_store(KDD, name="kdd")
