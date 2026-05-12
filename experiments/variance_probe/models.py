"""Composed :class:`DDSSM` models for variance-probe experiments.

Reuses encoder/decoder/z_init from :mod:`experiments.synthetic` for
the 1D shape; the D=4 ``ProbeMedium`` shape has its own
encoder/decoder/z_init (registered by :mod:`experiments.synthetic`).
"""

from __future__ import annotations

from ddssm.builders import DDSSM, Hparams

from conf.registry import model_store

from experiments.synthetic.encoders import (
    ProbeMedium as EncProbeMedium, Small1D as EncSmall1D,
)
from experiments.synthetic.decoders import (
    ProbeMedium as DecProbeMedium, Small1D as DecSmall1D,
)
from experiments.synthetic.z_inits import (
    ProbeMedium as ZProbeMedium, Small1D as ZSmall1D,
)
from experiments.variance_probe.transitions import DiffV2_1D, DiffV2_Medium


ProbeSmall = DDSSM(
    data_dim=1, latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
    encoder=EncSmall1D, decoder=DecSmall1D, z_init=ZSmall1D,
    transition=DiffV2_1D,
    hyperparams=Hparams(),
    use_observation_mask=False,
)

ProbeMedium = DDSSM(
    data_dim=4, latent_dim=8, j=1, emb_time_dim=16, covariate_dim=0,
    encoder=EncProbeMedium, decoder=DecProbeMedium, z_init=ZProbeMedium,
    transition=DiffV2_Medium,
    hyperparams=Hparams(),
    use_observation_mask=False,
)

model_store(ProbeSmall, name="probe_small")
model_store(ProbeMedium, name="probe_medium")
