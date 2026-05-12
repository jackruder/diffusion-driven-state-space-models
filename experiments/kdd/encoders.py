"""KDD encoder (D=6, j=1, covariate_dim=3, emb_time_dim=32)."""

from __future__ import annotations

from ddssm.builders import Encoder

from conf.registry import encoder_store


KDD = Encoder(
    data_dim=6, latent_dim=8, j=1, emb_time_dim=32,
    covariate_dim=3, use_mask=False,
)
encoder_store(KDD, name="kdd")
