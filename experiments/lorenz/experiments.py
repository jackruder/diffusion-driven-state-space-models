"""Named Lorenz experiment presets.

lorenz_smoke — the canonical entry point for validating the pipeline
on Lorenz 63 data. Uses the MLP / pinned / per-t cell (the same cell
as init_smoke_high_surface, which exercises every code path), sized for
3D direct observation with T=64.

The central question: can the diffusion transition learn the bimodal
lobe-switching residual that a Gaussian baseline cannot model?

Run:
    python -m ddssm.app experiment=lorenz_smoke
Inspect config without running:
    python -m ddssm.app experiment=lorenz_smoke --cfg job
"""

from __future__ import annotations

from experiments._make import experiment
from ddssm.experiment.stores import experiment_store
from experiments.init_centering.model import SmokeModel
from experiments.lorenz.data import LorenzDirect
from experiments.lorenz.hparams import LorenzHparams, LorenzStages, Training800

lorenz_smoke = experiment(
    data=LorenzDirect,
    model=SmokeModel(
        baseline_form="mlp",
        baseline_mode="pinned",
        tracking_mode="per_t",
        latent_dim=4,
        data_dim=3,
        T_max=64,
    ),
    hparams=LorenzHparams,
    training=Training800,
    stages=LorenzStages,
)
experiment_store(lorenz_smoke, name="lorenz_smoke")
