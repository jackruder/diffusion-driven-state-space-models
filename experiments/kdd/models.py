"""Composed :class:`DDSSM` models for the KDD family."""

from __future__ import annotations

from ddssm.builders import DDSSM, Hparams

from conf.registry import model_store

from experiments.kdd.decoders import KDD as DecKDD
from experiments.kdd.encoders import KDD as EncKDD
from experiments.kdd.transitions import KDDDiff, KDDGauss
from experiments.kdd.z_inits import KDD as ZKDD


KDDGaussModel = DDSSM(
    data_dim=6, latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
    encoder=EncKDD, decoder=DecKDD, z_init=ZKDD,
    transition=KDDGauss,
    hyperparams=Hparams(),
    use_observation_mask=False,
)

KDDDiffModel = DDSSM(
    data_dim=6, latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
    encoder=EncKDD, decoder=DecKDD, z_init=ZKDD,
    transition=KDDDiff,
    hyperparams=Hparams(),
    use_observation_mask=False,
)

model_store(KDDGaussModel, name="kdd_gauss")
model_store(KDDDiffModel, name="kdd_diff")
