"""KDD decoder (D=6, covariate_dim=3, emb_time_dim=32)."""

from __future__ import annotations

from ddssm.builders import Decoder

from conf.registry import decoder_store

from experiments.kdd.arch import KDDContext, KDDPlainHead


KDD = Decoder(
    data_dim=6, latent_dim=8, j=1, emb_time_dim=32,
    covariate_dim=3, static_covariate_dim=0,
    hidden_dim=64, mask_emb_dim=8,
    context=KDDContext,
    gaussian_head=KDDPlainHead,
)
decoder_store(KDD, name="kdd")
