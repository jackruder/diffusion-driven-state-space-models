"""Hparams and multi-stage config for the Lorenz experiment family.

Reuses the standard SmokeHparams and StagesB factory from init_centering,
but increases the stage budgets to accommodate 3D nonlinear dynamics:

- n_pretrain=800: ~4× the smoke run (200), giving the MLP baseline
  time to converge on the coupled quadratic Lorenz dynamics.
- n_stage2=2000: ~2× the smoke run (1000), giving the diffusion score
  net time to learn the bimodal lobe-switching residual in 3D.

Additional stage configs for ablation experiments:

- LorenzStagesGaussianOnly: single-stage Gaussian-only run at full budget
  (2800 steps = 800 + 2000); provides the apples-to-apples comparison
  against the two-stage diffusion run.

- LorenzStagesFrozenEnc: two-stage run with the encoder frozen during
  stage 2; tests whether encoder gradient conflicts cause the
  reconstruction spikes observed in lorenz_smoke.
"""

from __future__ import annotations

from dataclasses import replace

from hydra_zen import builds

from ddssm.training.stages import StagesConf
from experiments.init_centering.hparams import (
    LR,
    SmokeHparams,
    StagesB,
    Training800,
    _build_init_centering_stages,
)

LorenzHparams = SmokeHparams

# Canonical Lorenz stages: 800-step Gaussian pretrain + 2000-step diffusion
LorenzStages = StagesB(n_pretrain=800, n_stage2=2000)


def _build_lorenz_stages_gaussian_only(n_pretrain: int = 2800) -> StagesConf:
    """Single-stage Gaussian-only run at full budget for baseline comparison."""
    stages = _build_init_centering_stages(n_pretrain=n_pretrain, n_stage2=1)
    stages.run = ["stage_1"]
    return stages


LorenzStagesGaussianOnly = builds(
    _build_lorenz_stages_gaussian_only,
    populate_full_signature=True,
)


def _build_lorenz_stages_frozen_enc(
    n_pretrain: int = 800,
    n_stage2: int = 2000,
) -> StagesConf:
    """Two-stage run with encoder frozen in stage 2 to prevent gradient conflicts."""
    stages = _build_init_centering_stages(n_pretrain=n_pretrain, n_stage2=n_stage2)
    assert stages.stage_2 is not None
    stages.stage_2.trainable.encoder = False
    return stages


LorenzStagesFrozenEnc = builds(
    _build_lorenz_stages_frozen_enc,
    populate_full_signature=True,
)

# Extended stage-2 budget — tests whether the stage-2 plateau in lorenz_smoke
# is slow convergence or genuine stalling.
LorenzStagesLongStage2 = StagesB(n_pretrain=800, n_stage2=5000)


def _build_lorenz_stages_low_lr(
    n_pretrain: int = 800,
    n_stage2: int = 2000,
    stage2_trans_lr: float = 1e-4,
) -> StagesConf:
    """Two-stage run with a lower stage-2 trans_lr to reduce score-net instability."""
    stages = _build_init_centering_stages(n_pretrain=n_pretrain, n_stage2=n_stage2)
    assert stages.stage_2 is not None
    new_lrs = replace(stages.stage_2.lrs, trans_lr=stage2_trans_lr)
    stages.stage_2 = replace(stages.stage_2, lrs=new_lrs)
    return stages


LorenzStagesLowLr = builds(
    _build_lorenz_stages_low_lr,
    populate_full_signature=True,
)

def _build_lorenz_stages_sweep(
    n_pretrain: int = 800,
    n_stage2: int = 2000,
    stage2_trans_lr: float = 3e-4,
    base_lr: float = LR,
    dec_mult: float = 1.0,
    trans_mult: float = 1.0,
    stage_1_warmup_frac: float = 0.25,
    stage_2_warmup_frac: float = 0.10,
) -> StagesConf:
    """Stage builder for Lorenz Optuna sweeps — exposes all HPO-relevant knobs."""
    stages = _build_init_centering_stages(
        n_pretrain=n_pretrain,
        n_stage2=n_stage2,
        base_lr=base_lr,
        dec_mult=dec_mult,
        trans_mult=trans_mult,
        stage_1_warmup_frac=stage_1_warmup_frac,
        stage_2_warmup_frac=stage_2_warmup_frac,
    )
    assert stages.stage_2 is not None
    new_lrs = replace(stages.stage_2.lrs, trans_lr=stage2_trans_lr)
    stages.stage_2 = replace(stages.stage_2, lrs=new_lrs)
    return stages


LorenzStagesSweep = builds(_build_lorenz_stages_sweep, populate_full_signature=True)


def _build_lorenz_stages_sweep_frozen_enc(
    n_pretrain: int = 800,
    n_stage2: int = 2000,
    stage2_trans_lr: float = 3e-4,
    base_lr: float = LR,
    dec_mult: float = 1.0,
    trans_mult: float = 1.0,
    stage_1_warmup_frac: float = 0.25,
    stage_2_warmup_frac: float = 0.10,
) -> StagesConf:
    """Like _build_lorenz_stages_sweep but with encoder frozen in stage 2."""
    stages = _build_lorenz_stages_sweep(
        n_pretrain=n_pretrain,
        n_stage2=n_stage2,
        stage2_trans_lr=stage2_trans_lr,
        base_lr=base_lr,
        dec_mult=dec_mult,
        trans_mult=trans_mult,
        stage_1_warmup_frac=stage_1_warmup_frac,
        stage_2_warmup_frac=stage_2_warmup_frac,
    )
    assert stages.stage_2 is not None
    stages.stage_2.trainable.encoder = False
    return stages


LorenzStagesSweepFrozenEnc = builds(
    _build_lorenz_stages_sweep_frozen_enc,
    populate_full_signature=True,
)


# Keep LR accessible for callers that want to derive stage2_trans_lr relative
# to the base (e.g. stage2_trans_lr = LR / 5).
__all__ = [
    "LR",
    "LorenzHparams",
    "LorenzStages",
    "LorenzStagesGaussianOnly",
    "LorenzStagesFrozenEnc",
    "LorenzStagesLongStage2",
    "LorenzStagesLowLr",
    "LorenzStagesSweep",
    "LorenzStagesSweepFrozenEnc",
    "Training800",
    "_build_lorenz_stages_sweep",
    "_build_lorenz_stages_sweep_frozen_enc",
]
