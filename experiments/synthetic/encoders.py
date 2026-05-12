"""Encoders for the synthetic-data family.

Two shapes:

* ``small_1d`` — D=1, latent_dim=4, j=1 (used by harmonic, bimodal, lgssm).
* ``robot2d`` — D=2, latent_dim=6, j=2 (robot navigation).

Architectural knobs (context producer, gaussian head, future summary)
are imported from :mod:`experiments.synthetic.arch` so they're visible
at the experiment site rather than relying on silent defaults in
:mod:`ddssm.builders`.
"""

from __future__ import annotations

from ddssm.builders import Encoder

from conf.registry import encoder_store

from experiments.synthetic.arch import SmallContext, SmallHead, TinyGRU


Small1D = Encoder(
    data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
    covariate_dim=0, static_covariate_dim=0,
    use_mask=False,
    hidden_dim=64,
    fut_mask_emb_dim=8,
    pad_mask_emb_dim=8,
    context=SmallContext,
    gaussian_head=SmallHead,
    fut_summary=TinyGRU,
)

Robot2D = Encoder(
    data_dim=2, latent_dim=6, j=2, emb_time_dim=16,
    covariate_dim=0, static_covariate_dim=0,
    use_mask=False,
    hidden_dim=64,
    fut_mask_emb_dim=8,
    pad_mask_emb_dim=8,
    context=SmallContext,
    gaussian_head=SmallHead,
    fut_summary=TinyGRU,
)

ProbeMedium = Encoder(
    data_dim=4, latent_dim=8, j=1, emb_time_dim=16,
    covariate_dim=0, static_covariate_dim=0,
    use_mask=False,
    hidden_dim=64,
    fut_mask_emb_dim=8,
    pad_mask_emb_dim=8,
    context=SmallContext,
    gaussian_head=SmallHead,
    fut_summary=TinyGRU,
)

encoder_store(Small1D, name="small_1d")
encoder_store(Robot2D, name="robot2d")
encoder_store(ProbeMedium, name="probe_medium")
