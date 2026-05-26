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

from ddssm.stages import (
    LambdaRampConf,
    StagesConf,
    StageLrsConf,
    EarlyStopSpec,
    StageSpecConf,
    StageTrainableConf,
)
from ddssm.builders import Hparams, Training, CenteringHandoff

LR = 5e-4
LAMBDA_WARMUP = 50  # short: smoke runs are ~800 total steps


SmokeHparams = Hparams(
    S=1,
    batch_size=16,
    grad_accum_steps=1,
    lambda_schedule="cosine",
    lambda_start=0.001,
    lambda_end=1.0,
    lambda_warmup_steps=LAMBDA_WARMUP,
    enc_lr=LR,
    dec_lr=LR,
    zinit_lr=LR,
    trans_lr=LR,
    lambda_sigma_p=1e-2,  # stage-1 log-variance anchor (per model-v2.org
    # § State-conditional prior variance; suggested 1e-2 starting point).
)


# Single-fit fallback (used by Experiment.train *only* if model.config.stages
# is None — the smoke preset configures stages so this is informational only).
Training800 = Training(steps=800, log_every=25, checkpoint_every=200, amp=False)


def _build_init_centering_stages(
    *,
    baseline_mode: Literal["pinned", "learnable"] = "pinned",
    n_pretrain: int = 200,
    n_stage2: int = 1000,
    sigma_pert: float = 1e-2,
    # LRs are parametrised as ``base_lr`` (encoder LR) with per-group
    # multipliers for decoder + transition. This replaces the prior
    # 3-independent-log-uniforms sweep with a 1-base + 2-multiplier
    # search, exploiting the correlation between the LRs. ``zinit_lr``
    # stays separate (not swept by default). Pass ``enc_lr`` etc.
    # explicitly to override the derived values.
    base_lr: float = LR,
    dec_mult: float = 1.0,
    trans_mult: float = 1.0,
    enc_lr: float | None = None,
    dec_lr: float | None = None,
    zinit_lr: float = LR,
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
    stage_1_warmup_frac: float = 0.25,
    stage_2_warmup_frac: float = 0.10,
    log_every: int = 25,
    checkpoint_every: int = 200,
    early_stop_enabled: bool = False,
    early_stop_window: int = 50,
    early_stop_min_improvement: float = 1e-4,
    early_stop_warmup_steps: int = 100,
) -> StagesConf:
    """Build the two-stage orchestration spec for an init-centering cell.

    The stage-2 baseline trainable mask is wired in lockstep with
    ``baseline_mode`` so the declarative mask and the
    :func:`perform_centering_handoff` imperative freeze agree.
    """
    # Derive per-group LRs from (base_lr, multipliers) unless explicit
    # overrides are supplied.
    effective_enc_lr = enc_lr if enc_lr is not None else base_lr
    effective_dec_lr = dec_lr if dec_lr is not None else base_lr * dec_mult
    effective_trans_lr = trans_lr if trans_lr is not None else base_lr * trans_mult
    lrs = StageLrsConf(
        enc_lr=effective_enc_lr,
        dec_lr=effective_dec_lr,
        zinit_lr=zinit_lr,
        trans_lr=effective_trans_lr,
    )
    stage1_trainable = StageTrainableConf(
        encoder=True, decoder=True, z_init=False, transition=True, baseline=True,
    )
    stage2_baseline_trainable = baseline_mode == "learnable"
    stage2_trainable = StageTrainableConf(
        encoder=True, decoder=True, z_init=False, transition=True,
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
        end=1.0,
        steps=max(1, int(round(stage_1_warmup_frac * n_pretrain))),
        delay=0,
    )
    stage2_lambda = LambdaRampConf(
        start=float(stage_2_lambda_start),
        end=1.0,
        steps=max(1, int(round(stage_2_warmup_frac * n_stage2))),
        delay=0,
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
        ),
        stage_2=StageSpecConf(
            steps=int(n_stage2),
            trainable=stage2_trainable,
            lrs=lrs,
            lambda_ramp=stage2_lambda,
            log_every=log_every,
            val_every=0,
            checkpoint_every=checkpoint_every,
            centering_handoff=CenteringHandoff(sigma_pert=float(sigma_pert)),
        ),
        run=["stage_1", "stage_2"],
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
