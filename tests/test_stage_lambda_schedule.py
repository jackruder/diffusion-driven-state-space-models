"""Unit tests for the per-stage λ schedule (task #29)."""

from __future__ import annotations

import pytest

from ddssm.training.stages import LambdaRampConf, make_lambda_cosine


def test_cosine_ramp_starts_at_start() -> None:
    spec = LambdaRampConf(start=0.001, end=1.0, steps=100, delay=0)
    f = make_lambda_cosine(spec, total_steps=100, default_end=1.0)
    # The schedule is cosine: 0.5(1-cos(πu)) where u = t/T; t=0 ⇒ u=0 ⇒ factor=0
    assert f(0) == pytest.approx(0.001, abs=1e-6)


def test_cosine_ramp_ends_at_end() -> None:
    spec = LambdaRampConf(start=0.001, end=1.0, steps=100, delay=0)
    f = make_lambda_cosine(spec, total_steps=100, default_end=1.0)
    assert f(100) == pytest.approx(1.0, abs=1e-6)
    # Plateaus past the warmup.
    assert f(200) == pytest.approx(1.0, abs=1e-6)


def test_cosine_ramp_midpoint_halfway_between() -> None:
    """Cosine schedule reaches the midpoint of (start, end) at half-warmup."""
    spec = LambdaRampConf(start=0.0, end=1.0, steps=100, delay=0)
    f = make_lambda_cosine(spec, total_steps=100, default_end=1.0)
    # u=0.5 ⇒ cos(π/2)=0 ⇒ factor = 0.5(1-0) = 0.5
    assert f(50) == pytest.approx(0.5, abs=1e-6)


def test_stage_2_higher_start_starts_higher() -> None:
    """Stage 2's default ``λ_start=0.1`` keeps the floor above stage 1's 0.001."""
    s1 = LambdaRampConf(start=0.001, end=1.0, steps=50, delay=0)
    s2 = LambdaRampConf(start=0.1, end=1.0, steps=60, delay=0)
    f1 = make_lambda_cosine(s1, total_steps=50, default_end=1.0)
    f2 = make_lambda_cosine(s2, total_steps=60, default_end=1.0)
    assert f2(0) > f1(0)
    assert f2(0) == pytest.approx(0.1, abs=1e-6)


def test_init_centering_factory_wires_stage_lambda_ramps() -> None:
    """``_build_init_centering_stages`` must populate per-stage ``lambda_ramp``."""
    from experiments.init_centering.hparams import _build_init_centering_stages

    stages = _build_init_centering_stages(
        n_pretrain=200,
        n_stage2=600,
        stage_1_warmup_frac=0.25,
        stage_2_warmup_frac=0.10,
        stage_1_lambda_start=0.001,
        stage_2_lambda_start=0.1,
    )
    assert stages.stage_1.lambda_ramp.steps == 50  # 0.25 * 200
    assert stages.stage_1.lambda_ramp.start == pytest.approx(0.001)
    assert stages.stage_1.lambda_ramp.end == pytest.approx(1.0)

    assert stages.stage_2.lambda_ramp.steps == 60  # 0.10 * 600
    assert stages.stage_2.lambda_ramp.start == pytest.approx(0.1)
    assert stages.stage_2.lambda_ramp.end == pytest.approx(1.0)


def test_warmup_frac_rounds_to_at_least_one_step() -> None:
    """Tiny fractions still produce at least 1-step warmup (avoid divide-by-zero)."""
    from experiments.init_centering.hparams import _build_init_centering_stages

    stages = _build_init_centering_stages(
        n_pretrain=10,
        n_stage2=10,
        stage_1_warmup_frac=0.001,
        stage_2_warmup_frac=0.001,
    )
    assert stages.stage_1.lambda_ramp.steps >= 1
    assert stages.stage_2.lambda_ramp.steps >= 1
