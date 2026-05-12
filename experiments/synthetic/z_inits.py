"""z-init priors for the synthetic-data family."""

from __future__ import annotations

from ddssm.builders import ZInit

from conf.registry import z_init_store


Small1D = ZInit(latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0)
Robot2D = ZInit(latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0)
ProbeMedium = ZInit(latent_dim=8, j=1, emb_time_dim=16, covariate_dim=0)

z_init_store(Small1D, name="small_1d")
z_init_store(Robot2D, name="robot2d")
z_init_store(ProbeMedium, name="probe_medium")
