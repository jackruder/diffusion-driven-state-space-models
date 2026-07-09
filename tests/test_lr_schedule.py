"""Unit tests for LR-schedule config, math, and trainer integration.

Covers:
- ``make_lr_lambda``: warmup, hold, cosine decay, const shape, edge cases.
- ``resolve_lr_schedule_defaults``: table defaults, user-set field survival,
  error on missing lambda_ramp info, error on inverted anchors, immutability.
"""

from __future__ import annotations

import math

import pytest

from ddssm.training.stages import (
    LambdaRampConf,
    LrScheduleConf,
    LrScheduleGroupConf,
    make_lr_lambda,
    resolve_lr_schedule_defaults,
)

# ---------------------------------------------------------------------------
# make_lr_lambda
# ---------------------------------------------------------------------------


class TestMakeLrLambdaWarmup:
    """Warmup phase semantics."""

    def test_warmup_zero_gives_one_at_step_zero(self) -> None:
        """warmup_steps=0 must return 1.0 at step 0 (no division by zero)."""
        spec = LrScheduleConf(
            warmup_steps=0,
            decay_start=100,
            decay_end=400,
            final_scale=0.05,
            shape="cosine",
        )
        f = make_lr_lambda(spec)
        assert f(0) == pytest.approx(1.0, abs=1e-9)

    def test_warmup_linear_at_step_zero(self) -> None:
        """s=0 during warmup (warmup_steps>0) returns 0/warmup_steps = 0.0."""
        spec = LrScheduleConf(
            warmup_steps=100,
            decay_start=200,
            decay_end=600,
            final_scale=0.05,
            shape="cosine",
        )
        f = make_lr_lambda(spec)
        assert f(0) == pytest.approx(0.0, abs=1e-9)

    def test_warmup_linear_midpoint(self) -> None:
        """S = warmup_steps // 2 returns 0.5 for warmup_steps=100."""
        spec = LrScheduleConf(
            warmup_steps=100,
            decay_start=200,
            decay_end=600,
            final_scale=0.05,
            shape="cosine",
        )
        f = make_lr_lambda(spec)
        assert f(50) == pytest.approx(0.5, abs=1e-9)

    def test_warmup_at_warmup_steps_is_one(self) -> None:
        """S = warmup_steps itself: warmup_steps / warmup_steps = 1.0."""
        spec = LrScheduleConf(
            warmup_steps=100,
            decay_start=200,
            decay_end=600,
            final_scale=0.05,
            shape="cosine",
        )
        f = make_lr_lambda(spec)
        assert f(100) == pytest.approx(1.0, abs=1e-9)


class TestMakeLrLambdaHold:
    """Hold interval between warmup and decay."""

    def test_hold_interval_returns_one(self) -> None:
        """Steps in [warmup_steps, decay_start) should return 1.0."""
        spec = LrScheduleConf(
            warmup_steps=50,
            decay_start=200,
            decay_end=600,
            final_scale=0.05,
            shape="cosine",
        )
        f = make_lr_lambda(spec)
        # Explicitly check several hold steps
        for s in (50, 100, 150, 199):
            assert f(s) == pytest.approx(1.0, abs=1e-9), f"step={s}"


class TestMakeLrLambdaDecay:
    """Cosine decay phase."""

    def test_decay_strictly_decreasing(self) -> None:
        """LR should be monotonically decreasing across the decay window."""
        spec = LrScheduleConf(
            warmup_steps=0,
            decay_start=100,
            decay_end=400,
            final_scale=0.05,
            shape="cosine",
        )
        f = make_lr_lambda(spec)
        steps = range(100, 401, 10)
        values = [f(s) for s in steps]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], (
                f"Not non-increasing at step {list(steps)[i]}"
            )

    def test_final_scale_at_decay_end(self) -> None:
        """f(decay_end) should equal final_scale."""
        spec = LrScheduleConf(
            warmup_steps=0,
            decay_start=100,
            decay_end=400,
            final_scale=0.05,
            shape="cosine",
        )
        f = make_lr_lambda(spec)
        assert f(400) == pytest.approx(0.05, abs=1e-9)

    def test_final_scale_beyond_decay_end(self) -> None:
        """Steps past decay_end clamp to final_scale."""
        spec = LrScheduleConf(
            warmup_steps=0,
            decay_start=100,
            decay_end=400,
            final_scale=0.05,
            shape="cosine",
        )
        f = make_lr_lambda(spec)
        assert f(500) == pytest.approx(0.05, abs=1e-9)
        assert f(10000) == pytest.approx(0.05, abs=1e-9)

    def test_cosine_midpoint(self) -> None:
        """At the midpoint of decay the cosine formula gives the exact value."""
        fs = 0.1
        spec = LrScheduleConf(
            warmup_steps=0,
            decay_start=0,
            decay_end=200,
            final_scale=fs,
            shape="cosine",
        )
        f = make_lr_lambda(spec)
        # s=100, u = (100-0)/(200-0) = 0.5
        # cos(π*0.5) = 0; result = fs + (1-fs)*0.5*(1+0) = fs + (1-fs)*0.5
        expected = fs + (1 - fs) * 0.5 * (1 + math.cos(math.pi * 0.5))
        assert f(100) == pytest.approx(expected, abs=1e-9)

    def test_zero_width_decay_window(self) -> None:
        """decay_end == decay_start: step >= decay_start returns final_scale."""
        spec = LrScheduleConf(
            warmup_steps=0,
            decay_start=200,
            decay_end=200,
            final_scale=0.1,
            shape="cosine",
        )
        f = make_lr_lambda(spec)
        assert f(200) == pytest.approx(0.1, abs=1e-9)
        assert f(300) == pytest.approx(0.1, abs=1e-9)


class TestMakeLrLambdaConst:
    """shape='const' never decays."""

    def test_const_after_warmup_is_one(self) -> None:
        """Const shape returns 1.0 well past any decay_end."""
        spec = LrScheduleConf(
            warmup_steps=50,
            decay_start=100,
            decay_end=300,
            final_scale=0.05,
            shape="const",
        )
        f = make_lr_lambda(spec)
        assert f(1000) == pytest.approx(1.0, abs=1e-9)

    def test_const_hold_is_one(self) -> None:
        """Const shape returns 1.0 in the hold region."""
        spec = LrScheduleConf(
            warmup_steps=0,
            decay_start=100,
            decay_end=300,
            final_scale=0.05,
            shape="const",
        )
        f = make_lr_lambda(spec)
        assert f(150) == pytest.approx(1.0, abs=1e-9)


class TestMakeLrLambdaErrors:
    """Error cases."""

    def test_unknown_shape_raises_value_error(self) -> None:
        """Unsupported shape raises ValueError at build time, not per-step."""
        spec = LrScheduleConf(
            warmup_steps=0,
            decay_start=100,
            decay_end=400,
            final_scale=0.05,
            shape="linear",  # not supported
        )
        with pytest.raises(ValueError, match="shape"):
            make_lr_lambda(spec)


# ---------------------------------------------------------------------------
# resolve_lr_schedule_defaults
# ---------------------------------------------------------------------------


# Shared fixture values: delay=50, steps=150 → λ_end=200; T=800
_RAMP = LambdaRampConf(end=1.0, delay=50, steps=150, start=0.001)
_T = 800
# Expected defaults derived from the table (λ_end = 200):
_PHI_WARMUP = 0
_PHI_DECAY_START = 200  # λ_end
_PHI_DECAY_END = 800  # T
_PHI_FINAL = 0.05
_PSI_WARMUP = 50  # λ_end // 4 = 200 // 4
_PSI_DECAY_START = 500  # λ_end + (T - λ_end) // 2 = 200 + 600//2
_PSI_DECAY_END = 800  # T
_PSI_FINAL = 0.20


class TestResolveDefaultsAllNone:
    """All-None group fields filled exactly per the spec table."""

    def setup_method(self) -> None:
        group = LrScheduleGroupConf()  # all fields default (None where relevant)
        self.resolved = resolve_lr_schedule_defaults(group, _RAMP, _T)

    def test_phith_warmup_steps(self) -> None:
        assert self.resolved.phith.warmup_steps == _PHI_WARMUP

    def test_phith_decay_start(self) -> None:
        assert self.resolved.phith.decay_start == _PHI_DECAY_START

    def test_phith_decay_end(self) -> None:
        assert self.resolved.phith.decay_end == _PHI_DECAY_END

    def test_phith_final_scale(self) -> None:
        assert self.resolved.phith.final_scale == pytest.approx(_PHI_FINAL)

    def test_psi_warmup_steps(self) -> None:
        assert self.resolved.psi.warmup_steps == _PSI_WARMUP

    def test_psi_decay_start(self) -> None:
        assert self.resolved.psi.decay_start == _PSI_DECAY_START

    def test_psi_decay_end(self) -> None:
        assert self.resolved.psi.decay_end == _PSI_DECAY_END

    def test_psi_final_scale(self) -> None:
        assert self.resolved.psi.final_scale == pytest.approx(_PSI_FINAL)


class TestResolveDefaultsUserSetFieldsSurvive:
    """Non-None user-set fields must not be overwritten."""

    def test_phith_user_decay_start_survives(self) -> None:
        group = LrScheduleGroupConf(
            phith=LrScheduleConf(decay_start=300),
        )
        resolved = resolve_lr_schedule_defaults(group, _RAMP, _T)
        assert resolved.phith.decay_start == 300

    def test_psi_user_warmup_steps_survives(self) -> None:
        group = LrScheduleGroupConf(
            psi=LrScheduleConf(warmup_steps=10),
        )
        resolved = resolve_lr_schedule_defaults(group, _RAMP, _T)
        assert resolved.psi.warmup_steps == 10

    def test_phith_user_final_scale_survives(self) -> None:
        group = LrScheduleGroupConf(
            phith=LrScheduleConf(final_scale=0.01),
        )
        resolved = resolve_lr_schedule_defaults(group, _RAMP, _T)
        assert resolved.phith.final_scale == pytest.approx(0.01)

    def test_psi_user_decay_end_survives(self) -> None:
        group = LrScheduleGroupConf(
            psi=LrScheduleConf(decay_end=700),
        )
        # Must also satisfy the ordering constraint: set decay_start low enough.
        group.psi.decay_start = 300
        resolved = resolve_lr_schedule_defaults(group, _RAMP, _T)
        assert resolved.psi.decay_end == 700


class TestResolveDefaultsErrors:
    """Error cases for resolve_lr_schedule_defaults."""

    def test_lambda_ramp_none_raises(self) -> None:
        """ValueError when lambda_ramp is None and λ_end is needed."""
        group = LrScheduleGroupConf()  # phith.decay_start is None → λ_end needed
        with pytest.raises(ValueError):
            resolve_lr_schedule_defaults(group, None, _T)

    def test_lambda_ramp_steps_none_raises(self) -> None:
        """ValueError when lambda_ramp.steps is None and λ_end is needed."""
        ramp = LambdaRampConf(end=1.0, delay=50, steps=None, start=0.001)
        group = LrScheduleGroupConf()
        with pytest.raises(ValueError):
            resolve_lr_schedule_defaults(group, ramp, _T)

    def test_inverted_anchors_phith_raises(self) -> None:
        """decay_start > decay_end for phith should raise ValueError."""
        group = LrScheduleGroupConf(
            phith=LrScheduleConf(decay_start=700, decay_end=300),
        )
        with pytest.raises(ValueError):
            resolve_lr_schedule_defaults(group, _RAMP, _T)

    def test_inverted_anchors_psi_raises(self) -> None:
        """warmup_steps > decay_start for psi should raise ValueError."""
        group = LrScheduleGroupConf(
            psi=LrScheduleConf(warmup_steps=600, decay_start=100),
        )
        with pytest.raises(ValueError):
            resolve_lr_schedule_defaults(group, _RAMP, _T)


class TestResolveDefaultsImmutability:
    """Input conf objects must not be mutated."""

    def test_input_group_not_mutated(self) -> None:
        group = LrScheduleGroupConf()
        phith_id = id(group.phith)
        psi_id = id(group.psi)
        resolve_lr_schedule_defaults(group, _RAMP, _T)
        # Objects must be the same instances (not replaced in-place)
        assert id(group.phith) == phith_id
        assert id(group.psi) == psi_id
        # Fields must still be None
        assert group.phith.warmup_steps is None
        assert group.psi.decay_start is None


# ---------------------------------------------------------------------------
# Hparams surface: DDSSMHyperParamsConf + _default_hyperparams carry the two
# new optional fields, defaulting to None so existing configs are unaffected.
# ---------------------------------------------------------------------------


class TestHparamsSurface:
    def test_dataclass_defaults_are_none(self):
        from ddssm.model.dssd import DDSSMHyperParamsConf

        hp = DDSSMHyperParamsConf()
        assert hp.lambda_ramp is None
        assert hp.lr_schedule is None

    def test_default_hyperparams_namespace_has_none_fields(self):
        from ddssm.model.dssd import _default_hyperparams

        ns = _default_hyperparams()
        assert ns.lambda_ramp is None
        assert ns.lr_schedule is None

    def test_dataclass_accepts_configured_values(self):
        from ddssm.model.dssd import DDSSMHyperParamsConf

        ramp = LambdaRampConf(delay=50, steps=150)
        sched = LrScheduleGroupConf()
        hp = DDSSMHyperParamsConf(lambda_ramp=ramp, lr_schedule=sched)
        assert hp.lambda_ramp is ramp
        assert hp.lr_schedule is sched


# ---------------------------------------------------------------------------
# Trainer integration: _build_default_loss, _install_lr_schedule, fit() wiring,
# per-role LR logging, checkpoint round-trip and contract guards.
#
# These tests import torch and touch the trainer; they need the diffusion
# transition (make_vhp_model) to exercise the ψ role. Test invocation on this
# host requires LD_LIBRARY_PATH and TORCHDYNAMO_DISABLE=1 (see CLAUDE.md).
# ---------------------------------------------------------------------------


import os
import sys
from pathlib import Path

import torch as _torch


def _tests_dir_on_path():
    d = str(Path(__file__).parent)
    if d not in sys.path:
        sys.path.insert(0, d)


@pytest.fixture(scope="module")
def _eager_models():
    """Force torch.compile off so backward passes work without g++."""
    old = os.environ.get("DDSSM_TORCH_COMPILE")
    os.environ["DDSSM_TORCH_COMPILE"] = "0"
    yield
    if old is None:
        os.environ.pop("DDSSM_TORCH_COMPILE", None)
    else:
        os.environ["DDSSM_TORCH_COMPILE"] = old


def _vhp_model():
    _tests_dir_on_path()
    from tests.test_integration.conftest import make_vhp_model

    _torch.manual_seed(0)
    return make_vhp_model()


def _small_model():
    _tests_dir_on_path()
    from tests.test_trainer import make_small_model

    return make_small_model()


def _default_group() -> LrScheduleGroupConf:
    return LrScheduleGroupConf()


def _default_ramp() -> LambdaRampConf:
    return LambdaRampConf(end=1.0, delay=0, steps=40, start=0.001)


class TestBuildDefaultLoss:
    def test_no_ramp_gives_constant_lambda_one(self, tmp_path, _eager_models):
        from ddssm.training.train import DDSSMTrainer

        trainer = DDSSMTrainer(
            model=_small_model(),
            device=_torch.device("cpu"),
            tensorboard_dir=str(tmp_path / "tb"),
            quiet=True,
        )
        loss = trainer._build_default_loss(total_steps=100)
        assert loss.lambda_at(0) == pytest.approx(1.0)
        assert loss.lambda_at(50) == pytest.approx(1.0)
        assert loss.lambda_at(100) == pytest.approx(1.0)

    def test_with_ramp_traces_cosine(self, tmp_path, _eager_models):
        from ddssm.model.dssd import DDSSMHyperParamsConf
        from ddssm.training.train import DDSSMTrainer
        from ddssm.training.stages import make_lambda_cosine

        ramp = LambdaRampConf(end=1.0, delay=10, steps=80, start=0.001)
        hp = DDSSMHyperParamsConf(lambda_ramp=ramp)
        trainer = DDSSMTrainer(
            model=_small_model(),
            hparams=hp,
            device=_torch.device("cpu"),
            tensorboard_dir=str(tmp_path / "tb"),
            quiet=True,
        )
        loss = trainer._build_default_loss(total_steps=200)
        expected = make_lambda_cosine(ramp, total_steps=200, default_end=1.0)
        for s in (0, 5, 10, 30, 50, 90, 100, 199):
            assert loss.lambda_at(s) == pytest.approx(expected(s), abs=1e-8), (
                f"λ mismatch at step {s}"
            )


class TestInstallLrScheduleNoOp:
    def test_none_group_leaves_schedulers_empty(self, tmp_path, _eager_models):
        from ddssm.training.train import DDSSMTrainer

        trainer = DDSSMTrainer(
            model=_small_model(),
            device=_torch.device("cpu"),
            tensorboard_dir=str(tmp_path / "tb"),
            quiet=True,
        )
        trainer._install_lr_schedule(None, total_steps=100, lambda_ramp=None)
        assert trainer._schedulers == []


class TestInstallLrScheduleSingle:
    def test_single_optimizer_per_role_lambdas(self, tmp_path, _eager_models):
        from ddssm.model.dssd import DDSSMHyperParamsConf
        from ddssm.training.train import DDSSMTrainer
        from ddssm.training.stages import (
            make_lr_lambda,
            resolve_lr_schedule_defaults,
        )

        ramp = LambdaRampConf(end=1.0, delay=0, steps=40, start=0.001)
        group = LrScheduleGroupConf()
        hp = DDSSMHyperParamsConf(lambda_ramp=ramp, lr_schedule=group)
        trainer = DDSSMTrainer(
            model=_vhp_model(),
            hparams=hp,
            device=_torch.device("cpu"),
            tensorboard_dir=str(tmp_path / "tb"),
            quiet=True,
        )
        # __init__ built the AdamW with claim_psi=True (lr_schedule is set),
        # so we should see role-tagged groups on the single optimizer.
        roles = {g.get("role") for g in trainer.optimizer.param_groups}
        assert "phith" in roles
        assert "psi" in roles

        trainer._install_lr_schedule(
            group, total_steps=200, lambda_ramp=ramp
        )
        assert len(trainer._schedulers) == 1
        sched = trainer._schedulers[0]
        assert sched.optimizer is trainer.optimizer

        # Resolve defaults to compute expected per-role lambdas.
        resolved = resolve_lr_schedule_defaults(group, ramp, 200)
        phith_fn = make_lr_lambda(resolved.phith)
        psi_fn = make_lr_lambda(resolved.psi)

        # After LambdaLR ctor, last_epoch=0. The stored lr_lambdas must
        # match the group roles.
        for lrl, pg in zip(sched.lr_lambdas, trainer.optimizer.param_groups):
            role = pg.get("role", "phith")
            expected = phith_fn if role == "phith" else psi_fn
            for s in (0, 5, 10, 40, 100, 199):
                assert lrl(s) == pytest.approx(expected(s), abs=1e-9), (
                    f"lambda mismatch for role={role} at step {s}"
                )


class TestInstallLrScheduleSplit:
    def test_split_two_schedulers_independent_shapes(self, tmp_path, _eager_models):
        from ddssm.model.dssd import DDSSMHyperParamsConf
        from ddssm.training.train import DDSSMTrainer
        from ddssm.training.stages import (
            make_lr_lambda,
            resolve_lr_schedule_defaults,
        )

        ramp = LambdaRampConf(end=1.0, delay=0, steps=40, start=0.001)
        group = LrScheduleGroupConf()
        hp = DDSSMHyperParamsConf(lambda_ramp=ramp, lr_schedule=group)
        trainer = DDSSMTrainer(
            model=_vhp_model(),
            hparams=hp,
            device=_torch.device("cpu"),
            tensorboard_dir=str(tmp_path / "tb"),
            quiet=True,
        )
        trainer._install_split_topology()
        assert len(trainer._optimizers) == 2

        trainer._install_lr_schedule(
            group, total_steps=200, lambda_ramp=ramp
        )
        assert len(trainer._schedulers) == 2
        sched_phith, sched_psi = trainer._schedulers
        assert sched_phith.optimizer is trainer._optimizers[0]
        assert sched_psi.optimizer is trainer.opt_psi
        assert sched_psi.optimizer is trainer._optimizers[1]
        assert trainer.scheduler is sched_phith

        resolved = resolve_lr_schedule_defaults(group, ramp, 200)
        phith_fn = make_lr_lambda(resolved.phith)
        psi_fn = make_lr_lambda(resolved.psi)

        # phith side: every stored lambda equals phith_fn at probe steps.
        for lrl in sched_phith.lr_lambdas:
            for s in (0, 5, 10, 40, 100, 199):
                assert lrl(s) == pytest.approx(phith_fn(s), abs=1e-9)

        # psi side: every stored lambda equals psi_fn (NOT broadcast of
        # phith_fn — this is the regression the new install path fixes).
        for lrl in sched_psi.lr_lambdas:
            for s in (0, 5, 10, 40, 100, 199):
                assert lrl(s) == pytest.approx(psi_fn(s), abs=1e-9)

        # Sanity: at step 0, ψ warmup is active (warmup_steps=10) so ψ lr
        # is a fraction of base; φθ has no warmup so it's at base.
        # (Verifies the schedules are actually distinct.)
        assert psi_fn(0) < 1.0
        assert phith_fn(0) == pytest.approx(1.0)


class TestFitInstallsSchedule:
    def test_fit_installs_schedule_from_hparams(self, tmp_path, _eager_models):
        """fit() must install the LR schedule when hparams.lr_schedule is set."""
        from ddssm.model.dssd import DDSSMHyperParamsConf

        # Ramp fit within the 2-step budget: steps=1 → λ_end=1, ≤ T=2.
        ramp = LambdaRampConf(end=1.0, delay=0, steps=1, start=0.001)
        group = LrScheduleGroupConf()
        hp = DDSSMHyperParamsConf(
            lambda_ramp=ramp, lr_schedule=group, grad_accum_steps=1
        )
        trainer = _make_trainer_with_hparams(
            _vhp_model(), hp, split=False, tmp_path=tmp_path
        )
        # Confirm no scheduler pre-fit.
        assert trainer._schedulers == []

        _drive_fit(trainer, total_steps=2)

        # After fit, scheduler installed (single-optimizer).
        assert len(trainer._schedulers) == 1
        # last_epoch is 0 after LambdaLR ctor + 2 optimizer steps = 2
        # (schedulers step after each optimizer step, non-skipped).
        assert trainer._schedulers[0].last_epoch >= 1


class TestPerRoleLrLogging:
    def test_lr_phith_and_lr_psi_logged_in_split(self, tmp_path, _eager_models):
        """optim/lr_phith and optim/lr_psi appear in split-mode metrics."""
        from ddssm.model.dssd import DDSSMHyperParamsConf

        ramp = LambdaRampConf(end=1.0, delay=0, steps=1, start=0.001)
        group = LrScheduleGroupConf()
        hp = DDSSMHyperParamsConf(
            lambda_ramp=ramp, lr_schedule=group, grad_accum_steps=1
        )
        trainer = _make_trainer_with_hparams(
            _vhp_model(), hp, split=True, tmp_path=tmp_path
        )
        _drive_fit(trainer, total_steps=2)

        train_split = trainer.metrics._split("train")
        assert "optim/lr_phith" in train_split.meters, (
            f"optim/lr_phith not in metrics: {list(train_split.meters.keys())[:20]}"
        )
        assert "optim/lr_psi" in train_split.meters

    def test_lr_phith_and_lr_psi_logged_in_single(self, tmp_path, _eager_models):
        """Single-mode with role-tagged groups still emits both LR keys."""
        from ddssm.model.dssd import DDSSMHyperParamsConf

        ramp = LambdaRampConf(end=1.0, delay=0, steps=1, start=0.001)
        group = LrScheduleGroupConf()
        hp = DDSSMHyperParamsConf(
            lambda_ramp=ramp, lr_schedule=group, grad_accum_steps=1
        )
        trainer = _make_trainer_with_hparams(
            _vhp_model(), hp, split=False, tmp_path=tmp_path
        )
        _drive_fit(trainer, total_steps=2)

        train_split = trainer.metrics._split("train")
        assert "optim/lr_phith" in train_split.meters
        assert "optim/lr_psi" in train_split.meters


class TestCheckpointContractGuards:
    """The scheduler-contract guards in ``restore_from_checkpoint`` must
    hard-error on a scheduler-state mismatch. Both trainers here share
    identical hparams (same param-group topology) so the optimizer restore
    doesn't pre-empt the scheduler contract check; the only difference is
    whether ``_install_lr_schedule`` has been called on each side.
    """

    def _hp(self):
        from ddssm.model.dssd import DDSSMHyperParamsConf

        return DDSSMHyperParamsConf(
            # ``lr_schedule`` on hparams gates ``claim_psi=True`` in the
            # optimizer builder — both trainers must set it so their
            # param-group counts match and the optimizer restore doesn't
            # raise before we reach the scheduler contract.
            lambda_ramp=LambdaRampConf(end=1.0, delay=0, steps=20, start=0.001),
            lr_schedule=LrScheduleGroupConf(),
            grad_accum_steps=1,
        )

    def test_scheduled_ckpt_into_unscheduled_trainer_errors(
        self, tmp_path, _eager_models
    ):
        hp = self._hp()
        trainer = _make_trainer_with_hparams(
            _vhp_model(), hp, split=False, tmp_path=tmp_path / "sched"
        )
        # Install a scheduler manually so save writes scheduler_state.
        trainer._install_lr_schedule(
            hp.lr_schedule, total_steps=200, lambda_ramp=hp.lambda_ramp
        )
        assert trainer._schedulers
        ckpt_path = str(tmp_path / "sched_ckpt.pth")
        trainer.save_checkpoint(ckpt_path)

        # Second trainer with identical hparams but NO scheduler installed.
        trainer2 = _make_trainer_with_hparams(
            _vhp_model(), hp, split=False, tmp_path=tmp_path / "plain"
        )
        assert not trainer2._schedulers
        with pytest.raises(RuntimeError, match="scheduler"):
            trainer2.restore_from_checkpoint(ckpt_path)

    def test_unscheduled_ckpt_into_scheduled_trainer_errors(
        self, tmp_path, _eager_models
    ):
        hp = self._hp()
        trainer = _make_trainer_with_hparams(
            _vhp_model(), hp, split=False, tmp_path=tmp_path / "plain"
        )
        # Do NOT install a scheduler — the ckpt will carry scheduler_state=None.
        assert not trainer._schedulers
        ckpt_path = str(tmp_path / "plain_ckpt.pth")
        trainer.save_checkpoint(ckpt_path)

        trainer2 = _make_trainer_with_hparams(
            _vhp_model(), hp, split=False, tmp_path=tmp_path / "sched2"
        )
        trainer2._install_lr_schedule(
            hp.lr_schedule, total_steps=200, lambda_ramp=hp.lambda_ramp
        )
        assert trainer2._schedulers
        with pytest.raises(RuntimeError, match="scheduler"):
            trainer2.restore_from_checkpoint(ckpt_path)


# --- shared helpers for the trainer-integration tests ---------------------


def _make_trainer_with_hparams(model, hparams, *, split: bool, tmp_path):
    from ddssm.model.losses import FullELBO
    from ddssm.training.train import DDSSMTrainer

    trainer = DDSSMTrainer(
        model=model,
        hparams=hparams,
        device=_torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    trainer._active_loss = FullELBO(
        rate_lambda=lambda s: 1.0, use_split_loss=split
    )
    return trainer


def _drive_fit(trainer, *, total_steps: int):
    from torch.utils.data import Dataset, DataLoader

    class _DS(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            _torch.manual_seed(idx)
            return {
                "observed_data": _torch.randn(1, 5),
                "observation_mask": _torch.ones(1, 5),
                "timepoints": _torch.arange(5, dtype=_torch.long),
            }

    loader = DataLoader(_DS(), batch_size=2)
    trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=total_steps,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )
    return trainer
