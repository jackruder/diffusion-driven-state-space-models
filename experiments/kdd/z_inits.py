"""KDD z-init prior."""

from __future__ import annotations

from ddssm.builders import ZInit

from conf.registry import z_init_store


KDD = ZInit(latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3)
z_init_store(KDD, name="kdd")
