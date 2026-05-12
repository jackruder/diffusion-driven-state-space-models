"""Decoders for the synthetic-data family.

Architectural knobs (context producer, gaussian head) live in
:mod:`experiments.synthetic.arch` so they're visible at the experiment
site rather than relying on silent defaults in :mod:`ddssm.builders`.
"""

from __future__ import annotations

from ddssm.builders import Decoder, Head

from conf.registry import decoder_store

from experiments.synthetic.arch import SmallContext


# Decoder uses an unclamped Gaussian head (a logvar prior keeps it
# well-behaved during training); see :class:`~ddssm.decoder.GaussianDecoder`.
_DECODER_HEAD = Head()


Small1D = Decoder(
    data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
    covariate_dim=0, static_covariate_dim=0,
    hidden_dim=64, mask_emb_dim=8,
    context=SmallContext,
    gaussian_head=_DECODER_HEAD,
)

Robot2D = Decoder(
    data_dim=2, latent_dim=6, j=2, emb_time_dim=16,
    covariate_dim=0, static_covariate_dim=0,
    hidden_dim=64, mask_emb_dim=8,
    context=SmallContext,
    gaussian_head=_DECODER_HEAD,
)

ProbeMedium = Decoder(
    data_dim=4, latent_dim=8, j=1, emb_time_dim=16,
    covariate_dim=0, static_covariate_dim=0,
    hidden_dim=64, mask_emb_dim=8,
    context=SmallContext,
    gaussian_head=_DECODER_HEAD,
)

decoder_store(Small1D, name="small_1d")
decoder_store(Robot2D, name="robot2d")
decoder_store(ProbeMedium, name="probe_medium")
