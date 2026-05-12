"""KDD decoder (D=6, covariate_dim=3, emb_time_dim=32)."""

from __future__ import annotations

from ddssm.builders import Decoder

from conf.registry import decoder_store


KDD = Decoder(data_dim=6, latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3)
decoder_store(KDD, name="kdd")
