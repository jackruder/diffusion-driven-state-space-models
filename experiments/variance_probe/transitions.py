"""DiffusionV2 transitions for variance-probe experiments.

These are diagnostic runs on toy synthetic data — the score net's only
job is to be the *subject* of the variance measurement, not to extract
maximum capacity. Use the tiny MLP score-net (~10k params) so total
model size stays close to the synthetic Gaussian baseline.
"""

from __future__ import annotations

from ddssm.builders import DiffV2Transition

from conf.registry import transition_store

from experiments.synthetic.unets import MLPTiny
from experiments.variance_probe.schedules import V2


DiffV2_1D = DiffV2Transition(
    latent_dim=4, j=1, emb_time_dim=16, covariate_dim=0,
    unet=MLPTiny, schedule=V2,
)

DiffV2_Medium = DiffV2Transition(
    latent_dim=8, j=1, emb_time_dim=16, covariate_dim=0,
    unet=MLPTiny, schedule=V2,
)

transition_store(DiffV2_1D, name="diffv2_1d")
transition_store(DiffV2_Medium, name="diffv2_medium")
