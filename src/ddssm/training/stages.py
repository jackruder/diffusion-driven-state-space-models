"""Reusable single-phase training config pieces.

These small dataclasses configure a single ``trainer.fit(...)`` run:
per-module trainable masks, learning rates, an ELBO-plateau early-stop
spec, and a cosine rate-λ ramp. The multi-stage ``StageOrchestrator``
and its handoff hook were removed when staged training was retired
(training is now a single phase keyed on ``training.steps``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable


@dataclass
class StageLrsConf:
    """Per-module learning rates passed into ``trainer._rebuild_optimizer``."""

    enc_lr: float = 5e-4
    dec_lr: float = 5e-4
    trans_lr: float = 5e-4


@dataclass
class StageTrainableConf:
    """Per-module ``requires_grad`` mask.

    Matches the slot names used by :meth:`DDSSMTrainer._set_trainable`
    (encoder / decoder / transition).  The aux posterior is part of the
    *encoder* family (via DDSSM_base's ``aux_posterior`` slot) and shares
    the encoder flag.
    """

    encoder: bool = True
    decoder: bool = True
    transition: bool = True


@dataclass
class EarlyStopSpec:
    """ELBO-plateau early-stop spec.

    The trainer maintains a rolling window of ``loss/total`` values
    (one entry per logged train step).  Once at least ``window``
    entries are available *and* ``global_step >= warmup_steps``, the
    trainer compares the mean of the older half of the window against
    the mean of the newer half; if the relative drop
    ``(old_mean - new_mean) / max(|old_mean|, eps)`` is below
    ``min_improvement``, training exits early.
    """

    enabled: bool = False
    window: int = 50
    min_improvement: float = 1e-4
    warmup_steps: int = 100


@dataclass
class LambdaRampConf:
    """Cosine rate-λ ramp spec consumed by :func:`make_lambda_cosine`.

    The ramp runs from ``start`` to ``end`` over ``steps`` relative
    steps after an initial ``delay``.
    """

    end: float | None = 1.0
    delay: int = 0
    steps: int | None = None
    start: float = 0.001


def make_lambda_cosine(
    spec: LambdaRampConf, total_steps: int, default_end: float
) -> Callable[[int], float]:
    """Build a cosine λ-ramp schedule from a ``LambdaRamp`` spec.

    Args:
        spec: ``LambdaRampConf`` with ``start``, ``end``, ``delay``, ``steps``.
        total_steps: Fallback total step count when ``spec.steps`` is ``None``.
        default_end: Fallback end value when ``spec.end`` is ``None``.

    Returns:
        A callable ``f(step_idx: int) -> float`` returning λ at a given
        1-based step.
    """
    end = spec.end if spec.end is not None else default_end
    ramp_T = spec.steps if spec.steps is not None else total_steps
    delay = max(0, int(spec.delay))

    def f(step_idx: int) -> float:
        t = max(0, step_idx - delay)
        T = max(1, ramp_T - delay)
        u = min(1.0, t / T)
        return float(end + 0.5 * (spec.start - end) * (1.0 + math.cos(math.pi * u)))

    return f
