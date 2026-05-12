"""Composed :class:`DDSSM` models for the synthetic-data family.

Each model wires up an encoder + decoder + z-init + transition from
their named registries and registers the composition to
``model_store``. Hparams are a default ``Hparams()`` instance — the
experiment-level :func:`experiment` helper replaces this with the
preset's hparams at composition time.
"""

from __future__ import annotations

from ddssm.builders import DDSSM, Hparams

from conf.registry import model_store

from experiments.synthetic.encoders import Small1D as EncSmall1D
from experiments.synthetic.encoders import Robot2D as EncRobot2D
from experiments.synthetic.decoders import Small1D as DecSmall1D
from experiments.synthetic.decoders import Robot2D as DecRobot2D
from experiments.synthetic.z_inits import Small1D as ZSmall1D
from experiments.synthetic.z_inits import Robot2D as ZRobot2D
from experiments.synthetic.transitions import (
    Gauss1D, Diff1D, GaussRobot2D, DiffRobot2D,
)


SmallGauss = DDSSM(
    data_dim=1, latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
    encoder=EncSmall1D, decoder=DecSmall1D, z_init=ZSmall1D,
    transition=Gauss1D,
    hyperparams=Hparams(),
    use_observation_mask=False,
)

SmallDiff = DDSSM(
    data_dim=1, latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
    encoder=EncSmall1D, decoder=DecSmall1D, z_init=ZSmall1D,
    transition=Diff1D,
    hyperparams=Hparams(),
    use_observation_mask=False,
)

Robot2DGauss = DDSSM(
    data_dim=2, latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
    encoder=EncRobot2D, decoder=DecRobot2D, z_init=ZRobot2D,
    transition=GaussRobot2D,
    hyperparams=Hparams(),
    use_observation_mask=False,
)

Robot2DDiff = DDSSM(
    data_dim=2, latent_dim=6, j=2, emb_time_dim=16, covariate_dim=0,
    encoder=EncRobot2D, decoder=DecRobot2D, z_init=ZRobot2D,
    transition=DiffRobot2D,
    hyperparams=Hparams(),
    use_observation_mask=False,
)

model_store(SmallGauss, name="small_gauss")
model_store(SmallDiff, name="small_diff")
model_store(Robot2DGauss, name="robot2d_gauss")
model_store(Robot2DDiff, name="robot2d_diff")
