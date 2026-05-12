"""Decoders for the synthetic-data family."""

from __future__ import annotations

from ddssm.builders import Decoder

from conf.registry import decoder_store


Small1D = Decoder(
    data_dim=1, latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
)

Robot2D = Decoder(
    data_dim=2, latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
)

ProbeMedium = Decoder(
    data_dim=4, latent_dim=8, j=1, emb_time_dim=16, covariate_dim=0,
)

decoder_store(Small1D, name="small_1d")
decoder_store(Robot2D, name="robot2d")
decoder_store(ProbeMedium, name="probe_medium")
