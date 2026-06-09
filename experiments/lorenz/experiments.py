"""Named Lorenz experiment presets.

lorenz_smoke         — canonical: MLP/pinned/per-t, 800-step Gaussian pretrain
                       + 2000-step diffusion. Starting point for the family.

lorenz_gaussian_full — Gaussian-only for 2800 total steps (same budget as smoke);
                       the apples-to-apples baseline. Runs only stage_1 with the
                       Gaussian transition so we can compare final ELBO.

lorenz_frozen_enc    — same as lorenz_smoke but encoder is frozen during stage 2.
                       Tests whether encoder gradient conflicts cause the
                       reconstruction spikes observed in lorenz_smoke.

lorenz_long_stage2   — same as lorenz_smoke but stage 2 extended to 5000 steps.
                       Tests whether the stage-2 plateau is slow convergence.

lorenz_low_lr        — same as lorenz_smoke but stage-2 trans_lr lowered to 1e-4.
                       Tests whether LR instability is causing the reconstruction
                       spikes without sacrificing encoder adaptability.

Run:
    python -m ddssm.app experiment=lorenz_smoke
    python -m ddssm.app experiment=lorenz_gaussian_full
    python -m ddssm.app experiment=lorenz_frozen_enc
    python -m ddssm.app experiment=lorenz_long_stage2
    python -m ddssm.app experiment=lorenz_low_lr
Inspect config without running:
    python -m ddssm.app experiment=lorenz_smoke --cfg job
"""

from __future__ import annotations

from experiments._make import experiment
from ddssm.experiment.stores import experiment_store
from experiments.init_centering.model import SmokeModel
from experiments.lorenz.data import LorenzDirect
from experiments.lorenz.hparams import (
    LorenzHparams,
    LorenzStages,
    LorenzStagesGaussianOnly,
    LorenzStagesFrozenEnc,
    LorenzStagesLongStage2,
    LorenzStagesLowLr,
    Training800,
)

_model_kwargs = dict(
    baseline_form="mlp",
    baseline_mode="pinned",
    tracking_mode="per_t",
    latent_dim=4,
    data_dim=3,
    T_max=64,
)

lorenz_smoke = experiment(
    data=LorenzDirect,
    model=SmokeModel(**_model_kwargs),
    hparams=LorenzHparams,
    training=Training800,
    stages=LorenzStages,
)
experiment_store(lorenz_smoke, name="lorenz_smoke")

lorenz_gaussian_full = experiment(
    data=LorenzDirect,
    model=SmokeModel(**_model_kwargs),
    hparams=LorenzHparams,
    training=Training800,
    stages=LorenzStagesGaussianOnly,
)
experiment_store(lorenz_gaussian_full, name="lorenz_gaussian_full")

lorenz_frozen_enc = experiment(
    data=LorenzDirect,
    model=SmokeModel(**_model_kwargs),
    hparams=LorenzHparams,
    training=Training800,
    stages=LorenzStagesFrozenEnc,
)
experiment_store(lorenz_frozen_enc, name="lorenz_frozen_enc")

lorenz_long_stage2 = experiment(
    data=LorenzDirect,
    model=SmokeModel(**_model_kwargs),
    hparams=LorenzHparams,
    training=Training800,
    stages=LorenzStagesLongStage2,
)
experiment_store(lorenz_long_stage2, name="lorenz_long_stage2")

lorenz_low_lr = experiment(
    data=LorenzDirect,
    model=SmokeModel(**_model_kwargs),
    hparams=LorenzHparams,
    training=Training800,
    stages=LorenzStagesLowLr,
)
experiment_store(lorenz_low_lr, name="lorenz_low_lr")