"""Hparams + multi-stage config for the gluonts_forecast family.

The persistence baseline is param-free → always **pinned** (no R_μp / learnable
branch). Budgets are FIXED per dataset (pinned from the pilot), NOT swept; the
lean sweep tunes ``base_lr``/``dec_mult``/``trans_mult`` (+ ``latent_dim`` on the
model, ``batch_size`` on hparams) only. ``validate_every`` MUST be > 0 so the
validation ELBO (the sweep objective) is logged to ``metrics.csv``. EarlyStop is
ON for stage_1 so we don't over-spend the recon pretrain budget.
"""

from __future__ import annotations

from hydra_zen import builds

from ddssm.model.losses import FullELBO
from ddssm.training.stages import (
    StagesConf,
    StageLrsConf,
    EarlyStopSpec,
    StageSpecConf,
    LambdaRampConf,
    StageTrainableConf,
    make_lambda_cosine,
)
from ddssm.experiment.builders import Hparams, Training, CenteringHandoff

BASE_LR = 5e-4


GluonHparams = Hparams(
    S=1,
    batch_size=64,
    grad_accum_steps=1,
    ema_decay=0.999,
    enc_lr=BASE_LR,
    dec_lr=BASE_LR,
    trans_lr=BASE_LR,
)

# validate_every MUST be > 0: Experiment.train only builds the val_loader when
# ``training.validate_every > 0`` (experiment.py), and the val ELBO is the sweep
# objective. The per-stage ``val_every`` then controls the in-stage cadence.
GluonTraining = Training(
    steps=20000,
    log_every=100,
    validate_every=500,
    checkpoint_every=2000,
    amp=True,
)


def _build_gluonts_stages(
    *,
    n_pretrain: int = 3000,
    n_stage2: int = 30000,
    sigma_pert: float = 1e-3,
    lambda_sigma_p: float = 1e-2,
    base_lr: float = BASE_LR,
    dec_mult: float = 1.0,
    trans_mult: float = 1.0,
    enc_lr: float | None = None,
    dec_lr: float | None = None,
    trans_lr: float | None = None,
    stage_1_lambda_start: float = 0.001,
    stage_2_lambda_start: float = 0.1,
    # λ ramp END (default 1.0). Set start=end=0.0 to pin the rate/KL weight to a
    # constant 0 — a pure autoencoder (Phase-1 encoder-capacity probe), no ramp.
    stage_1_lambda_end: float = 1.0,
    stage_2_lambda_end: float = 1.0,
    stage_1_warmup_frac: float = 0.3,
    stage_2_warmup_frac: float = 0.1,
    log_every: int = 100,
    validate_every: int = 500,
    checkpoint_every: int = 2000,
    early_stop_enabled: bool = True,
    early_stop_window: int = 500,
    early_stop_min_improvement: float = 1e-4,
    early_stop_warmup_steps: int = 500,
    # Stage run order; default both. Pass ["stage_1"] for a stage-1-only run
    # (the pure-AE Phase-1 path — stage_2 is still built but never executed).
    run: list[str] | None = None,
    # Freeze enc+dec in stage_2 so ONLY the transition trains against a STATIC
    # latent frame — isolates transition learning from the encoder co-evolution
    # trap (a good frame is learned in stage_1, then pinned).
    stage_2_freeze_frame: bool = False,
) -> StagesConf:
    """Two-stage recon→joint orchestration for a gluonts cell (persistence/pinned)."""
    effective_enc_lr = enc_lr if enc_lr is not None else base_lr
    effective_dec_lr = dec_lr if dec_lr is not None else base_lr * dec_mult
    effective_trans_lr = trans_lr if trans_lr is not None else base_lr * trans_mult
    lrs = StageLrsConf(
        enc_lr=effective_enc_lr,
        dec_lr=effective_dec_lr,
        trans_lr=effective_trans_lr,
    )
    # Persistence μ_p is param-free → baseline never trainable (pinned).
    trainable = StageTrainableConf(
        encoder=True,
        decoder=True,
        transition=True,
        baseline=False,
    )
    # Stage-2 mask: optionally freeze the learned frame (enc+dec) so only the
    # transition trains against a static latent target.
    stage2_trainable = (
        StageTrainableConf(
            encoder=False,
            decoder=False,
            transition=True,
            baseline=False,
        )
        if stage_2_freeze_frame
        else trainable
    )
    es = EarlyStopSpec(
        enabled=early_stop_enabled,
        window=early_stop_window,
        min_improvement=early_stop_min_improvement,
        warmup_steps=early_stop_warmup_steps,
    )
    stage1_lambda = LambdaRampConf(
        start=float(stage_1_lambda_start),
        end=float(stage_1_lambda_end),
        steps=max(1, int(round(stage_1_warmup_frac * n_pretrain))),
        delay=0,
    )
    stage2_lambda = LambdaRampConf(
        start=float(stage_2_lambda_start),
        end=float(stage_2_lambda_end),
        steps=max(1, int(round(stage_2_warmup_frac * n_stage2))),
        delay=0,
    )
    stage1_loss = FullELBO(
        rate_lambda=make_lambda_cosine(
            stage1_lambda,
            total_steps=int(n_pretrain),
            default_end=float(stage_1_lambda_end),
        ),
        lambda_sigma_p=lambda_sigma_p,
        lambda_mu_p=0.0,
    )
    # Persistence baseline: R_μp anchor (λ_μp) is moot — μ_p has no parameters.
    stage2_loss = FullELBO(
        rate_lambda=make_lambda_cosine(
            stage2_lambda,
            total_steps=int(n_stage2),
            default_end=float(stage_2_lambda_end),
        ),
        lambda_sigma_p=0.0,
        lambda_mu_p=0.0,
    )
    return StagesConf(
        stage_1=StageSpecConf(
            steps=int(n_pretrain),
            trainable=trainable,
            lrs=lrs,
            lambda_ramp=stage1_lambda,
            log_every=log_every,
            val_every=validate_every,
            checkpoint_every=checkpoint_every,
            early_stop=es if early_stop_enabled else None,
            # Fires *after* stage 1 (only when stage 2 will run): a
            # stage-2-only run gets no handoff, hence zero μ_p snapshot/pin
            # or encoder perturbation.
            centering_handoff=CenteringHandoff(sigma_pert=float(sigma_pert)),
            loss=stage1_loss,
        ),
        stage_2=StageSpecConf(
            steps=int(n_stage2),
            trainable=stage2_trainable,
            lrs=lrs,
            lambda_ramp=stage2_lambda,
            log_every=log_every,
            val_every=validate_every,
            checkpoint_every=checkpoint_every,
            loss=stage2_loss,
        ),
        run=list(run) if run is not None else ["stage_1", "stage_2"],
    )


# hydra-zen wrapper so the preset / Optuna sweep can override fields by name.
GluonStages = builds(_build_gluonts_stages, populate_full_signature=True)


__all__ = [
    "GluonHparams",
    "GluonStages",
    "GluonTraining",
    "_build_gluonts_stages",
]
