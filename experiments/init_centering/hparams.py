"""Hparams + training scalars + multi-stage config for the init-centering preset.

Phase B promotes the cell axes (``baseline_mode``, ``tracking_mode``)
and the two sweep knobs (``sigma_pert``, ``n_pretrain``) to
first-class fields on the hparams dataclass and exposes a parametric
:class:`StagesConf` factory so the Optuna sweep (Phase C) can sample
them via CLI override.

Default values reproduce the canonical cell from
``init-experiment.org:275`` — MLP / Pinned / per-t EMA — with the
trainable-mask interlock baked in:

* Stage 1 trains ``baseline=True``.
* Stage 2 under Pinned: ``baseline=False`` (consistent with the
  imperative freeze in :func:`perform_centering_handoff`).
* Stage 2 under Learnable: ``baseline=True`` so the R_μp anchor
  regulariser is the only thing constraining μ_p's drift.
"""

from __future__ import annotations

from typing import Literal

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

LR = 5e-4
LAMBDA_WARMUP = 50  # short: smoke runs are ~800 total steps


SmokeHparams = Hparams(
    S=1,
    batch_size=16,
    grad_accum_steps=1,
    ema_decay=0.997,
    enc_lr=LR,
    dec_lr=LR,
    trans_lr=LR,
)
# Stage-1 λ_σp moved to the per-stage loss object in
# ``_build_init_centering_stages`` (ADR-0004).
LAMBDA_SIGMA_P = 1e-2


# Single-fit fallback (used by Experiment.train *only* if model.config.stages
# is None — the smoke preset configures stages so this is informational only).
Training800 = Training(steps=800, log_every=25, checkpoint_every=200, amp=True)


def _build_init_centering_stages(
    *,
    baseline_mode: Literal["pinned", "learnable"] = "pinned",
    n_pretrain: int = 200,
    n_stage2: int = 1000,
    sigma_pert: float = 1e-2,
    # Regulariser strengths (Optuna-swept). ``lambda_sigma_p`` is the
    # stage-1 log-variance anchor λ_σp; it feeds the stage-1 loss object
    # directly (ADR-0004 — it is NOT read off Hparams). ``anchor_lambda``
    # is the stage-2 R_μp strength λ_μp; ``None`` ⇒ 0.0 under Pinned (the
    # term is moot when μ_p is frozen) / 1e-2 under Learnable.
    lambda_sigma_p: float = LAMBDA_SIGMA_P,
    anchor_lambda: float | None = None,
    # LRs are parametrised as ``base_lr`` (encoder LR) with per-group
    # multipliers for decoder + transition. This replaces the prior
    # 3-independent-log-uniforms sweep with a 1-base + 2-multiplier
    # search, exploiting the correlation between the LRs. Pass
    # ``enc_lr`` etc. explicitly to override the derived values.
    base_lr: float = LR,
    dec_mult: float = 1.0,
    trans_mult: float = 1.0,
    enc_lr: float | None = None,
    dec_lr: float | None = None,
    trans_lr: float | None = None,
    # Per-stage λ-warmup parameters (CONTEXT.md § "lambda_warmup
    # redesign"). Each stage runs a cosine ramp on its OWN stage-local
    # step counter; the ramp covers ``warmup_frac × stage.steps`` and
    # rises from ``lambda_start`` to 1.0. Stage 2's default ``λ_start``
    # is 0.1 (higher than stage 1's 0.001) because the model is
    # pretrained, not random — we only need to ease through the
    # loss-form change, not relearn the rate-distortion tradeoff.
    stage_1_lambda_start: float = 0.001,
    stage_2_lambda_start: float = 0.1,
    stage_1_lambda_end: float = 1.0,
    stage_2_lambda_end: float = 1.0,
    stage_1_warmup_frac: float = 0.25,
    stage_2_warmup_frac: float = 0.10,
    log_every: int = 25,
    checkpoint_every: int = 200,
    early_stop_enabled: bool = False,
    early_stop_window: int = 50,
    early_stop_min_improvement: float = 1e-4,
    early_stop_warmup_steps: int = 100,
    stage_2_freeze_enc_dec: bool = False,
    # Which stages to execute, in order. Default runs both. Override
    # to ``["stage_2"]`` to skip the closed-form-Gaussian pretrain and
    # train the diffusion-centered model from random init (no stage-1
    # baseline anchor — the centering_handoff still snapshots the
    # random-init baseline, harmless under baseline_mode="pinned").
    run_stages: list[str] | None = None,
) -> StagesConf:
    """Build the two-stage orchestration spec for an init-centering cell.

    The stage-2 baseline trainable mask is wired in lockstep with
    ``baseline_mode`` so the declarative mask and the
    :func:`perform_centering_handoff` imperative freeze agree.
    """
    # ``anchor_lambda`` (stage-2 λ_μp) defaults by mode: the R_μp term is
    # moot when μ_p is frozen (Pinned ⇒ 0.0); Learnable ⇒ the
    # doc-recommended 1e-2 (model-v2.org § Baseline-mode variants).
    if anchor_lambda is None:
        anchor_lambda = 0.0 if baseline_mode == "pinned" else 1e-2

    # Derive per-group LRs from (base_lr, multipliers) unless explicit
    # overrides are supplied.
    effective_enc_lr = enc_lr if enc_lr is not None else base_lr
    effective_dec_lr = dec_lr if dec_lr is not None else base_lr * dec_mult
    effective_trans_lr = trans_lr if trans_lr is not None else base_lr * trans_mult
    lrs = StageLrsConf(
        enc_lr=effective_enc_lr,
        dec_lr=effective_dec_lr,
        trans_lr=effective_trans_lr,
    )
    stage1_trainable = StageTrainableConf(
        encoder=True,
        decoder=True,
        transition=True,
        baseline=True,
    )
    stage2_baseline_trainable = baseline_mode == "learnable"
    stage2_enc = not stage_2_freeze_enc_dec
    stage2_dec = not stage_2_freeze_enc_dec
    stage2_trainable = StageTrainableConf(
        encoder=stage2_enc,
        decoder=stage2_dec,
        transition=True,
        baseline=stage2_baseline_trainable,
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
    # Per-stage loss objects (ADR-0004). Each owns its own λ schedule +
    # regulariser weights; the model returns the components unweighted.
    # Stage 1 carries the log-variance anchor λ_σp; stage 2 carries the
    # R_μp anchor λ_μp (``anchor_lambda``) — both live here on the loss
    # object, not on the model or Hparams.
    stage1_loss = FullELBO(
        rate_lambda=make_lambda_cosine(
            stage1_lambda,
            total_steps=int(n_pretrain),
            default_end=1.0,
        ),
        lambda_sigma_p=lambda_sigma_p,
        lambda_mu_p=0.0,
    )
    stage2_loss = FullELBO(
        rate_lambda=make_lambda_cosine(
            stage2_lambda,
            total_steps=int(n_stage2),
            default_end=1.0,
        ),
        lambda_sigma_p=0.0,
        lambda_mu_p=anchor_lambda,
    )
    return StagesConf(
        stage_1=StageSpecConf(
            steps=int(n_pretrain),
            trainable=stage1_trainable,
            lrs=lrs,
            lambda_ramp=stage1_lambda,
            log_every=log_every,
            val_every=0,
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
            val_every=0,
            checkpoint_every=checkpoint_every,
            loss=stage2_loss,
        ),
        run=list(run_stages) if run_stages is not None else ["stage_1", "stage_2"],
    )


# hydra-zen wrapper so the preset / Optuna sweep can override fields by name.
StagesB = builds(
    _build_init_centering_stages,
    populate_full_signature=True,
)


__all__ = [
    "SmokeHparams",
    "StagesB",
    "Training800",
    "_build_init_centering_stages",
]
