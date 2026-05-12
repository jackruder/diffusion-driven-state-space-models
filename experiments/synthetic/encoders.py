"""Encoders for the synthetic-data family.

Two shapes:

* ``small_1d`` — D=1, latent_dim=4, j=1 (used by harmonic, bimodal, lgssm).
* ``robot2d`` — D=2, latent_dim=6, j=2 (robot navigation).

All use a tiny single-layer GRU future-summary
(``summary_dim=16, num_layers=1``) — the dominant per-step cost in
training (it's sequential over T) and overkill for toy synthetic
data at this latent size.
"""

from __future__ import annotations

from ddssm.builders import Encoder, GRUFutSum

from conf.registry import encoder_store


# Shared tiny GRU summary — single layer, 16-dim hidden state.
_TINY_GRU = GRUFutSum(summary_dim=16, num_layers=1)


Small1D = Encoder(
    data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
    covariate_dim=0, use_mask=False,
    fut_summary=_TINY_GRU,
)

Robot2D = Encoder(
    data_dim=2, latent_dim=6, j=2, emb_time_dim=16,
    covariate_dim=0, use_mask=False,
    fut_summary=_TINY_GRU,
)

ProbeMedium = Encoder(
    data_dim=4, latent_dim=8, j=1, emb_time_dim=16,
    covariate_dim=0, use_mask=False,
    fut_summary=_TINY_GRU,
)

encoder_store(Small1D, name="small_1d")
encoder_store(Robot2D, name="robot2d")
encoder_store(ProbeMedium, name="probe_medium")
