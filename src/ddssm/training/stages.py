"""Reusable single-phase training config pieces.

These small dataclasses configure a single ``trainer.fit(...)`` run:
per-module trainable masks, learning rates, an ELBO-plateau early-stop
spec, and a cosine rate-λ ramp. The multi-stage ``StageOrchestrator``
and its handoff hook were removed when staged training was retired
(training is now a single phase keyed on ``training.steps``).
"""

from __future__ import annotations

import math
from dataclasses import field, dataclass
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


@dataclass
class LrScheduleConf:
    """Per-role LR schedule spec consumed by :func:`make_lr_lambda`.

    All numeric fields may be ``None``; use
    :func:`resolve_lr_schedule_defaults` to fill them before calling
    :func:`make_lr_lambda`.
    """

    warmup_steps: int | None = None
    decay_start: int | None = None
    decay_end: int | None = None
    final_scale: float | None = None
    shape: str = "cosine"  # "cosine" | "const"


@dataclass
class LrScheduleGroupConf:
    """Per-role LR schedule group (φθ encoder/decoder, ψ transition).

    Args:
        phith: Schedule for the encoder/decoder (φ, θ) optimiser.
        psi: Schedule for the transition (ψ) optimiser.
    """

    phith: LrScheduleConf = field(default_factory=LrScheduleConf)
    psi: LrScheduleConf = field(default_factory=LrScheduleConf)


def make_lr_lambda(spec: LrScheduleConf) -> Callable[[int], float]:
    """Build a step-index → scale callable from a fully-resolved ``LrScheduleConf``.

    The returned callable takes a 0-based step index and returns a
    multiplicative LR scale in ``[0, 1]``.

    Phases (evaluated in order):
    1. **Warmup** – linear 0→1 over ``[0, warmup_steps)``.
       If ``warmup_steps == 0`` the warmup phase is skipped and step 0
       enters the hold phase at scale 1.0.
    2. **Hold** – constant 1.0 over ``[warmup_steps, decay_start)``.
    3. **Decay** (shape-dependent):
       - ``"cosine"``: cosine from 1.0 to ``final_scale`` over
         ``[decay_start, decay_end]``.  Steps past ``decay_end`` clamp
         to ``final_scale``.
       - ``"const"``: always 1.0 (``final_scale`` is ignored).

    Args:
        spec: A fully-resolved :class:`LrScheduleConf` (no ``None`` fields).

    Returns:
        A callable ``f(step: int) -> float``.

    Raises:
        ValueError: If ``spec.shape`` is not ``"cosine"`` or ``"const"``.
    """
    if spec.shape not in {"cosine", "const"}:
        raise ValueError(
            f"LrScheduleConf.shape must be 'cosine' or 'const', got {spec.shape!r}"
        )

    warmup_steps = spec.warmup_steps  # int, non-None after resolution
    decay_start = spec.decay_start
    decay_end = spec.decay_end
    final_scale = spec.final_scale
    shape = spec.shape

    def f(step: int) -> float:
        # 1. Warmup phase
        if warmup_steps is not None and warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        # 2. Const shape: never decays
        if shape == "const":
            return 1.0
        # 3. Hold phase
        if decay_start is not None and step < decay_start:
            return 1.0
        # 4. Past decay window
        if decay_end is not None and step >= decay_end:
            return float(final_scale)  # type: ignore[arg-type]
        # 5. Cosine decay interior
        width = decay_end - decay_start  # type: ignore[operator]
        u = (step - decay_start) / width  # type: ignore[operator]
        return float(
            final_scale  # type: ignore[operator]
            + (1.0 - final_scale)  # type: ignore[operator]
            * 0.5
            * (1.0 + math.cos(math.pi * u))
        )

    return f


def resolve_lr_schedule_defaults(
    group: LrScheduleGroupConf,
    lambda_ramp: LambdaRampConf | None,
    training_steps: int,
) -> LrScheduleGroupConf:
    """Fill ``None`` fields in *group* using the standard default table.

    Returns NEW :class:`LrScheduleConf` objects (the input is never mutated).

    Default table (``T = training_steps``, ``λ_end = lambda_ramp.delay + lambda_ramp.steps``):

    .. code-block:: text

        role   warmup_steps  decay_start               decay_end  final_scale
        phith  0             λ_end                     T          0.05
        psi    λ_end // 4    λ_end + (T - λ_end) // 2  T          0.20

    ``λ_end`` is computed only when at least one of the fields
    ``phith.decay_start``, ``psi.warmup_steps``, or ``psi.decay_start``
    is ``None``.  If it *is* needed and ``lambda_ramp`` is ``None`` or
    ``lambda_ramp.steps`` is ``None``, a :class:`ValueError` is raised
    (rather than silently degenerating to ``T``).

    Args:
        group: Input :class:`LrScheduleGroupConf` (possibly with ``None`` fields).
        lambda_ramp: :class:`LambdaRampConf` used to compute ``λ_end``.
        training_steps: Total training step budget ``T``.

    Returns:
        A new :class:`LrScheduleGroupConf` with all fields filled.

    Raises:
        ValueError: If ``λ_end`` is needed but cannot be computed, or if the
            resolved anchors violate ``0 <= warmup_steps <= decay_start
            <= decay_end <= T`` for either role.
    """
    T = training_steps

    # Determine whether λ_end is required
    needs_lambda_end = (
        group.phith.decay_start is None
        or group.psi.warmup_steps is None
        or group.psi.decay_start is None
    )

    lambda_end: int | None = None
    if needs_lambda_end:
        if lambda_ramp is None:
            raise ValueError(
                "resolve_lr_schedule_defaults: lambda_ramp is None but λ_end is"
                " needed to fill missing decay_start / warmup_steps fields."
                " Pass a LambdaRampConf with steps set."
            )
        if lambda_ramp.steps is None:
            raise ValueError(
                "resolve_lr_schedule_defaults: lambda_ramp.steps is None but"
                " λ_end is needed to fill missing decay_start / warmup_steps"
                " fields.  Setting steps=None would silently degenerate λ_end"
                " to T and eliminate all decay — set lambda_ramp.steps explicitly."
            )
        lambda_end = lambda_ramp.delay + lambda_ramp.steps

    def _fill_phith(src: LrScheduleConf) -> LrScheduleConf:
        assert lambda_end is not None or src.decay_start is not None
        return LrScheduleConf(
            warmup_steps=src.warmup_steps if src.warmup_steps is not None else 0,
            decay_start=(
                src.decay_start if src.decay_start is not None else lambda_end
            ),
            decay_end=src.decay_end if src.decay_end is not None else T,
            final_scale=src.final_scale if src.final_scale is not None else 0.05,
            shape=src.shape,
        )

    def _fill_psi(src: LrScheduleConf) -> LrScheduleConf:
        assert lambda_end is not None or (
            src.warmup_steps is not None and src.decay_start is not None
        )
        le = lambda_end  # local alias for clarity
        return LrScheduleConf(
            warmup_steps=(
                src.warmup_steps
                if src.warmup_steps is not None
                else (le // 4)  # type: ignore[operator]
            ),
            decay_start=(
                src.decay_start
                if src.decay_start is not None
                else (le + (T - le) // 2)  # type: ignore[operator]
            ),
            decay_end=src.decay_end if src.decay_end is not None else T,
            final_scale=src.final_scale if src.final_scale is not None else 0.20,
            shape=src.shape,
        )

    filled_phith = _fill_phith(group.phith)
    filled_psi = _fill_psi(group.psi)

    # Validate ordering for each role
    for role_name, conf in (("phith", filled_phith), ("psi", filled_psi)):
        ws = conf.warmup_steps
        ds = conf.decay_start
        de = conf.decay_end
        if not (0 <= ws <= ds <= de <= T):  # type: ignore[operator]
            raise ValueError(
                f"resolve_lr_schedule_defaults: invalid anchor ordering for"
                f" role '{role_name}': 0 <= warmup_steps({ws}) <="
                f" decay_start({ds}) <= decay_end({de}) <= T({T}) violated."
            )

    return LrScheduleGroupConf(phith=filled_phith, psi=filled_psi)


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
