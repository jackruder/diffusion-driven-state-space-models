"""KDD encoder (D=6, j=1, covariate_dim=3, emb_time_dim=32)."""

from __future__ import annotations

from ddssm.builders import Encoder

from conf.registry import encoder_store

from experiments.kdd.arch import KDDClampedHead, KDDContext, KDDFutSum


KDD = Encoder(
    data_dim=6, latent_dim=8, j=1, emb_time_dim=32,
    covariate_dim=3, static_covariate_dim=0,
    use_mask=False,
    hidden_dim=64,
    fut_mask_emb_dim=8,
    pad_mask_emb_dim=8,
    context=KDDContext,
    gaussian_head=KDDClampedHead,
    fut_summary=KDDFutSum,
)
encoder_store(KDD, name="kdd")
