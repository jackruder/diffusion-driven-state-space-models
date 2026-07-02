"""Tests for the checkpoint module (ADR-0005): payload schema, cross-check, EMA."""

from __future__ import annotations

from types import SimpleNamespace
import logging

import torch
import pytest
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ddssm.model.dssd import DDSSMHyperParamsConf
from ddssm.model.losses import FullELBO
from tests.test_trainer import make_small_model
from ddssm.training.train import DDSSMTrainer
from ddssm.training.checkpoint import Checkpoint, prepare_model, load_into_model
from ddssm.training.train_utils import make_warmup_cosine
from tests.test_integration.conftest import make_vhp_model


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.transition = nn.Linear(2, 2)


def _fake_trainer(model: _Toy, *, yaml: str | None = "m: 1") -> SimpleNamespace:
    """Minimal stand-in carrying the attributes Checkpoint.from_trainer reads."""
    return SimpleNamespace(
        model=model,
        optimizer=None,
        ema=SimpleNamespace(shadow=model.transition.state_dict()),
        ema_decay=0.999,
        global_step=7,
        grad_accum_steps=3,
        _model_config_yaml=yaml,
    )


def test_save_load_roundtrip(tmp_path):
    model = _Toy()
    ckpt = Checkpoint.from_trainer(_fake_trainer(model))
    path = str(tmp_path / "ckpt.pth")
    ckpt.save(path)

    loaded = Checkpoint.load(path, device=torch.device("cpu"))
    assert loaded.global_step == 7
    assert loaded.grad_accum_steps == 3
    assert loaded.model_config_yaml == "m: 1"
    assert loaded.ema_decay == pytest.approx(0.999)
    assert set(loaded.model_state) == set(model.state_dict())
    assert loaded.ema_state is not None


def test_load_into_model_applies_state(tmp_path):
    src = _Toy()
    path = str(tmp_path / "ckpt.pth")
    Checkpoint.from_trainer(_fake_trainer(src)).save(path)

    dst = _Toy()
    # Perturb dst so a successful load is observable.
    with torch.no_grad():
        for p in dst.parameters():
            p.add_(1.0)
    load_into_model(dst, path, device=torch.device("cpu"))
    for (k, a), (_, b) in zip(dst.state_dict().items(), src.state_dict().items()):
        assert torch.allclose(a, b), f"{k} did not load"


def test_cross_check_warns_on_drift(tmp_path, caplog):
    model = _Toy()
    path = str(tmp_path / "ckpt.pth")
    Checkpoint.from_trainer(_fake_trainer(model, yaml="hidden_dim: 64")).save(path)

    with caplog.at_level(logging.WARNING, logger="ddssm.training.checkpoint"):
        load_into_model(
            _Toy(),
            path,
            device=torch.device("cpu"),
            expected_model_config_yaml="hidden_dim: 80",
        )
    assert any("config drift" in r.message for r in caplog.records)


def test_cross_check_silent_when_match(tmp_path, caplog):
    model = _Toy()
    path = str(tmp_path / "ckpt.pth")
    Checkpoint.from_trainer(_fake_trainer(model, yaml="hidden_dim: 64")).save(path)

    with caplog.at_level(logging.WARNING, logger="ddssm.training.checkpoint"):
        load_into_model(
            _Toy(),
            path,
            device=torch.device("cpu"),
            expected_model_config_yaml="hidden_dim: 64",
        )
    assert not any("config drift" in r.message for r in caplog.records)


def test_load_ema_swaps_transition(tmp_path):
    model = _Toy()
    # Live transition weights = 0; EMA shadow = 1. The payload records both.
    with torch.no_grad():
        for p in model.transition.parameters():
            p.zero_()
    ema_shadow = {
        k: torch.ones_like(v) for k, v in model.transition.state_dict().items()
    }
    trainer = _fake_trainer(model)
    trainer.ema = SimpleNamespace(shadow=ema_shadow)
    path = str(tmp_path / "ckpt.pth")
    Checkpoint.from_trainer(trainer).save(path)

    # load_ema=False → live (zero) transition weights.
    live = _Toy()
    load_into_model(live, path, device=torch.device("cpu"), load_ema=False)
    assert torch.allclose(
        live.transition.weight, torch.zeros_like(live.transition.weight)
    )

    # load_ema=True → EMA (one) transition weights.
    ema = _Toy()
    load_into_model(ema, path, device=torch.device("cpu"), load_ema=True)
    assert torch.allclose(ema.transition.weight, torch.ones_like(ema.transition.weight))


def _ema_checkpoint(tmp_path) -> str:
    """Save a checkpoint whose transition live weights are 0 and EMA shadow is 1."""
    model = _Toy()
    with torch.no_grad():
        for p in model.transition.parameters():
            p.zero_()
    ema_shadow = {
        k: torch.ones_like(v) for k, v in model.transition.state_dict().items()
    }
    tr = _fake_trainer(model)
    tr.ema = SimpleNamespace(shadow=ema_shadow)
    path = str(tmp_path / "ema.pth")
    Checkpoint.from_trainer(tr).save(path)
    return path


def test_prepare_model_defaults_to_ema(tmp_path):
    """``prepare_model`` loads EMA shadows by default; opt out for live weights."""
    path = _ema_checkpoint(tmp_path)

    exp = SimpleNamespace(model=_Toy(), model_config_yaml=None)
    m = prepare_model(exp, checkpoint_path=path, device=torch.device("cpu"))
    assert torch.allclose(m.transition.weight, torch.ones_like(m.transition.weight))

    exp_live = SimpleNamespace(model=_Toy(), model_config_yaml=None)
    m_live = prepare_model(
        exp_live,
        checkpoint_path=path,
        device=torch.device("cpu"),
        load_ema=False,
    )
    assert torch.allclose(
        m_live.transition.weight, torch.zeros_like(m_live.transition.weight)
    )


def test_legacy_raw_state_dict_loads(tmp_path):
    """A bare state_dict (pre-payload checkpoint) still loads."""
    src = _Toy()
    path = str(tmp_path / "raw.pth")
    torch.save(src.state_dict(), path)

    loaded = Checkpoint.load(path, device=torch.device("cpu"))
    assert loaded.model_config_yaml is None
    dst = _Toy()
    load_into_model(dst, path, device=torch.device("cpu"))
    assert torch.allclose(dst.lin.weight, src.lin.weight)


# ---------------------------------------------------------------------------
# v2 schema: scaler_state + scheduler_state round-trip, back-compat,
# contract guard against silent enabled/disabled mismatch.
# ---------------------------------------------------------------------------


def test_scaler_scheduler_state_roundtrip(tmp_path):
    """v2 payload preserves non-None scaler / scheduler state on disk."""
    model = _Toy()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)
    # Advance the scheduler a few steps so its ``last_epoch`` differs from
    # the freshly-constructed default — guards against a silent reset on load.
    for _ in range(3):
        optimiser.step()
        scheduler.step()
    sched_state_pre = scheduler.state_dict()

    # Fabricate a "scaler state" dict — we can't enable a real GradScaler on
    # CPU-only CI, so we just verify the payload round-trips the dict bytes.
    scaler_state = {"scale": 1024.0, "growth_tracker": 5}

    payload_ckpt = Checkpoint(
        model_state=model.state_dict(),
        optimizer_state=optimiser.state_dict(),
        scaler_state=scaler_state,
        scheduler_state=sched_state_pre,
    )
    path = str(tmp_path / "v2.pth")
    payload_ckpt.save(path)

    loaded = Checkpoint.load(path, device=torch.device("cpu"))
    assert loaded.scaler_state == scaler_state
    # Scheduler state_dicts are plain dicts of python scalars — direct ==.
    assert loaded.scheduler_state == sched_state_pre


def test_v1_payload_back_compat_no_scaler_scheduler(tmp_path):
    """A hand-rolled v1 payload (no scaler/scheduler fields) loads cleanly."""
    model = _Toy()
    legacy_payload = {
        "_format": "ddssm_ckpt_v1",
        "model_config_yaml": None,
        "model_state": model.state_dict(),
        "optimizer_state": None,
        "ema_decay": 0.999,
        "ema_state": None,
        "global_step": 11,
        "grad_accum_steps": 1,
    }
    path = str(tmp_path / "v1.pth")
    torch.save(legacy_payload, path)

    loaded = Checkpoint.load(path, device=torch.device("cpu"))
    assert loaded.global_step == 11
    assert loaded.scaler_state is None
    assert loaded.scheduler_state is None


def test_restore_raises_when_saved_scaler_state_but_live_scaler_disabled(tmp_path):
    """v2 ckpt with scaler_state into a disabled-scaler trainer must raise."""
    # Build a real (minimal) trainer and write a v2 payload that includes
    # a synthetic scaler_state. The trainer's default ``self.scaler`` is
    # disabled, so the contract guard must fire on restore.
    import sys

    # ``tests/test_trainer.py`` already builds a small DDSSM model; reuse it
    # by adding the tests dir to sys.path. This keeps the regression test
    # self-contained without copy-pasting the model factory.
    tests_dir = str(__import__("pathlib").Path(__file__).parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_trainer import make_small_model  # type: ignore

    from ddssm.training.train import DDSSMTrainer

    model = make_small_model()
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    # Save a real (live-scaler-disabled) v2 ckpt, then inject a non-None
    # scaler_state into the payload and re-save.
    ckpt_path = str(tmp_path / "ckpt.pth")
    trainer.save_checkpoint(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    payload["scaler_state"] = {"scale": 2048.0, "growth_tracker": 0}
    torch.save(payload, ckpt_path)

    assert not trainer.scaler.is_enabled(), "precondition: live scaler disabled"
    with pytest.raises(RuntimeError, match="GradScaler"):
        trainer.restore_from_checkpoint(ckpt_path)


def test_rng_state_roundtrip_through_restore(tmp_path):
    """Restoring a checkpoint rewinds torch/numpy/python RNG to save time.

    Regression guard: checkpoints carried no RNG state, so a preempt-resume
    replayed a different noise/dropout/shuffle stream than the uninterrupted
    run would have seen.
    """
    import sys
    import random

    import numpy as np

    tests_dir = str(__import__("pathlib").Path(__file__).parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_trainer import make_small_model  # type: ignore

    from ddssm.training.train import DDSSMTrainer

    model = make_small_model()
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    torch.manual_seed(123)
    np.random.seed(456)
    random.seed(789)
    ckpt_path = str(tmp_path / "ckpt.pth")
    trainer.save_checkpoint(ckpt_path)

    # The draws an uninterrupted run would make right after the save.
    expected_torch = torch.rand(4)
    expected_np = np.random.random(4)
    expected_py = random.random()

    # Scramble all three streams (simulates dying and restarting elsewhere).
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    trainer.restore_from_checkpoint(ckpt_path)
    assert torch.equal(torch.rand(4), expected_torch)
    assert np.array_equal(np.random.random(4), expected_np)
    assert random.random() == expected_py


def test_legacy_payload_without_rng_state_restores(tmp_path):
    """A payload lacking ``rng_state`` (pre-RNG schema) restores cleanly."""
    import sys

    tests_dir = str(__import__("pathlib").Path(__file__).parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_trainer import make_small_model  # type: ignore

    from ddssm.training.train import DDSSMTrainer

    model = make_small_model()
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    ckpt_path = str(tmp_path / "ckpt.pth")
    trainer.save_checkpoint(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    payload.pop("rng_state", None)
    torch.save(payload, ckpt_path)

    trainer.restore_from_checkpoint(ckpt_path)  # must not raise


def test_restore_raises_on_grad_accum_steps_mismatch(tmp_path):
    """A ckpt with a different grad_accum_steps than the live trainer must raise.

    Loss is divided by ``self.grad_accum_steps`` in the backward path, so
    silently rescaling it across resume shifts the effective LR mid-run.
    Symmetric with the scaler/scheduler guards.
    """
    import sys

    tests_dir = str(__import__("pathlib").Path(__file__).parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_trainer import make_small_model  # type: ignore

    from ddssm.training.train import DDSSMTrainer

    model = make_small_model()
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    ckpt_path = str(tmp_path / "ckpt.pth")
    trainer.save_checkpoint(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    payload["grad_accum_steps"] = trainer.grad_accum_steps + 1
    torch.save(payload, ckpt_path)

    with pytest.raises(RuntimeError, match="grad_accum_steps"):
        trainer.restore_from_checkpoint(ckpt_path)


# ---------------------------------------------------------------------------
# M7 — v3 schema: split-mode fields, grad_skip_count, format-mismatch guards
# ---------------------------------------------------------------------------


class _DS(Dataset):
    """Tiny deterministic dataset shaped for the DiffusionTransition model."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(idx)
        return {
            "observed_data": torch.randn(1, 5),
            "observation_mask": torch.ones(1, 5),
            "timepoints": torch.arange(5, dtype=torch.long),
        }


def _make_trainer(model, *, split: bool, tmp_path, hparams=None) -> DDSSMTrainer:
    """Build a DDSSMTrainer with ``_active_loss`` installed (no fit() needed)."""
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        hparams=hparams,
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    trainer._active_loss = FullELBO(rate_lambda=lambda s: 1.0, use_split_loss=split)
    return trainer


def _drive_n_steps(trainer, n: int, *, resume_from: str | None = None):
    """Drive trainer.fit for n steps using the tiny diffusion-compatible loader."""
    loader = DataLoader(_DS(), batch_size=2)
    trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=n,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
        resume_from=resume_from,
    )
    return trainer


def _make_v2_payload(model, *, global_step: int) -> dict:
    """A verbatim v2 payload dict — deliberately hand-rolled.

    The consumers of this helper are legacy-compat tests that pin the LEGACY
    on-disk schema: building the payload through ``Checkpoint`` would silently
    track any future schema change and defeat the back-compat guarantee, so
    the v2 keys are spelled out by hand here.
    """
    return {
        "_format": "ddssm_ckpt_v2",
        "model_config_yaml": None,
        "model_state": model.state_dict(),
        "optimizer_state": None,
        "ema_decay": 0.999,
        "ema_state": None,
        "global_step": global_step,
        "grad_accum_steps": 1,
        "stage_prefix": None,
        "stage_start_step": 0,
        "rng_state": None,
        "scaler_state": None,
        "scheduler_state": None,
        # No split_loss / optimizer_state_psi / scheduler_state_psi /
        # grad_skip_count — v2 predates them.
    }


@pytest.mark.slow
def test_ckpt_v3_round_trip_split_mode(tmp_path):
    """Save at step 2, resume into a fresh split-mode trainer, run one more step.

    Checks: (a) the checkpoint carries the v3 split-mode fields, and
    (b) after restore the resumed trainer has the right global_step and
    non-empty φθ/ψ optimizer state (i.e. the restore really loaded it).
    Full param-equality vs a straight 3-step run is not asserted because
    Adam's β₂ EMA is slightly path-dependent under split mode; we check
    continuity and correct metadata instead.
    """
    # --- 2 steps → save ---
    torch.manual_seed(42)
    trainer_a = _make_trainer(make_vhp_model(), split=True, tmp_path=tmp_path / "a")
    _drive_n_steps(trainer_a, 2)
    assert trainer_a.opt_psi is not None
    assert len(trainer_a.opt_psi.state_dict()["state"]) > 0, (
        "precondition: ψ optimizer accumulated state before save"
    )
    ckpt_path = str(tmp_path / "split_v3.pth")
    trainer_a.save_checkpoint(ckpt_path)

    # Verify v3 fields in the checkpoint
    ckpt = Checkpoint.load(ckpt_path, device=torch.device("cpu"))
    assert ckpt.split_loss is True, "saved ckpt must be flagged split_loss=True"
    assert ckpt.optimizer_state_psi is not None, "must capture psi optimizer state"
    assert ckpt.global_step == 2
    assert ckpt.grad_skip_count == 0

    # --- Fresh split trainer: resume from the ckpt, run 1 more step ---
    torch.manual_seed(42)
    trainer_resume = _make_trainer(
        make_vhp_model(), split=True, tmp_path=tmp_path / "resume"
    )
    _drive_n_steps(trainer_resume, 3, resume_from=ckpt_path)
    # Post-resume: global_step advanced, params finite, no errors
    assert trainer_resume.global_step == 3, (
        f"Expected global_step=3 after resume+1step, got {trainer_resume.global_step}"
    )
    # The restore actually loaded optimizer state on both sides of the split.
    assert trainer_resume.opt_psi is not None
    assert len(trainer_resume.opt_psi.state_dict()["state"]) > 0, (
        "ψ optimizer state empty after resume — restore did not load it"
    )
    assert len(trainer_resume.optimizer.state_dict()["state"]) > 0, (
        "φθ optimizer state empty after resume — restore did not load it"
    )
    assert trainer_resume.grad_skip_count == 0
    for k, v in trainer_resume.model.state_dict().items():
        assert torch.isfinite(v).all(), f"param {k} has non-finite values after resume"


def test_legacy_v2_ckpt_loads_into_single_mode(tmp_path):
    """A hand-crafted v2 payload loads into a use_split_loss=False trainer.

    After restore, trainer.grad_skip_count == 0 (legacy default).
    """
    model = make_small_model()
    # Use grad_accum_steps=1 to match the legacy payload (trainer default is 4)
    trainer = _make_trainer(
        model,
        split=False,
        tmp_path=tmp_path,
        hparams=DDSSMHyperParamsConf(grad_accum_steps=1),
    )

    ckpt_path = str(tmp_path / "v2.pth")
    torch.save(_make_v2_payload(model, global_step=5), ckpt_path)

    trainer.restore_from_checkpoint(ckpt_path)  # must not raise
    assert trainer.global_step == 5
    assert trainer.grad_skip_count == 0, (
        "Legacy v2 payload must restore grad_skip_count=0 (default)"
    )


@pytest.mark.parametrize(
    ("ckpt_split", "trainer_split"),
    [(True, False), (False, True)],
    ids=["split-ckpt-into-single", "single-ckpt-into-split"],
)
def test_split_mode_mismatch_raises(ckpt_split, trainer_split, tmp_path):
    """A v3 ckpt whose split_loss flag disagrees with the live trainer raises."""
    model = make_small_model()

    ckpt_path = str(tmp_path / "ckpt.pth")
    Checkpoint(
        model_state=model.state_dict(),
        split_loss=ckpt_split,
        grad_accum_steps=1,
        global_step=1,
    ).save(ckpt_path)

    trainer = _make_trainer(
        model,
        split=trainer_split,
        tmp_path=tmp_path,
        hparams=DDSSMHyperParamsConf(grad_accum_steps=1),
    )
    with pytest.raises(ValueError, match="split"):
        trainer.restore_from_checkpoint(ckpt_path)


def test_split_loss_lr_schedulers_survive_round_trip(tmp_path):
    """After save/load in split mode, both schedulers' last_epoch are preserved.

    Uses the vhp diffusion model with the two-optimizer split topology and
    schedulers installed directly (``_install_split_topology`` +
    ``_install_scheduler``), so no fit() / training steps are needed.
    """
    trainer = _make_trainer(make_vhp_model(), split=True, tmp_path=tmp_path / "sched")
    # Install split topology directly (bypasses need to run fit())
    trainer._install_split_topology()

    # Install a real scheduler via _install_scheduler so both phith+psi are set
    sched_phith = make_warmup_cosine(
        trainer._optimizers[0], total_steps=100, warmup_steps=5, final_scale=0.1
    )
    trainer._install_scheduler(sched_phith)

    # Advance schedulers a few steps
    for _ in range(4):
        for sched in trainer._schedulers:
            sched.step()

    pre_save_epochs = [sched.last_epoch for sched in trainer._schedulers]
    assert len(pre_save_epochs) == 2, "split mode must have 2 schedulers"

    ckpt_path = str(tmp_path / "sched_ckpt.pth")
    trainer.save_checkpoint(ckpt_path)

    ckpt = Checkpoint.load(ckpt_path, device=torch.device("cpu"))
    assert ckpt.scheduler_state is not None
    assert ckpt.scheduler_state_psi is not None

    # Restore into a fresh split trainer with the same scheduler shape
    trainer2 = _make_trainer(make_vhp_model(), split=True, tmp_path=tmp_path / "sched2")
    trainer2._install_split_topology()
    sched_phith2 = make_warmup_cosine(
        trainer2._optimizers[0], total_steps=100, warmup_steps=5, final_scale=0.1
    )
    trainer2._install_scheduler(sched_phith2)

    trainer2.restore_from_checkpoint(ckpt_path)

    post_load_epochs = [sched.last_epoch for sched in trainer2._schedulers]
    assert post_load_epochs == pre_save_epochs, (
        f"Scheduler last_epoch mismatch after round-trip: "
        f"{pre_save_epochs} -> {post_load_epochs}"
    )


def test_grad_skip_count_survives_round_trip(tmp_path):
    """grad_skip_count=7 before save is restored after load.

    Also: legacy v2 payload → restored trainer has grad_skip_count=0.
    """
    # --- Part A: v3 round-trip ---
    trainer = DDSSMTrainer(
        model=make_small_model(),
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        quiet=True,
    )
    trainer.grad_skip_count = 7
    ckpt_path = str(tmp_path / "skip_ckpt.pth")
    trainer.save_checkpoint(ckpt_path)

    trainer2 = DDSSMTrainer(
        model=make_small_model(),
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb2"),
        quiet=True,
    )
    trainer2.restore_from_checkpoint(ckpt_path)
    assert trainer2.grad_skip_count == 7, (
        f"Expected grad_skip_count=7 after round-trip, got {trainer2.grad_skip_count}"
    )

    # --- Part B: legacy v2 payload → grad_skip_count defaults to 0 ---
    model3 = make_small_model()
    trainer3 = DDSSMTrainer(
        model=model3,
        device=torch.device("cpu"),
        hparams=DDSSMHyperParamsConf(grad_accum_steps=1),
        tensorboard_dir=str(tmp_path / "tb3"),
        quiet=True,
    )
    v2_path = str(tmp_path / "v2legacy.pth")
    torch.save(_make_v2_payload(model3, global_step=3), v2_path)

    trainer3.grad_skip_count = 99  # pre-set to something non-zero
    trainer3.restore_from_checkpoint(v2_path)
    assert trainer3.grad_skip_count == 0, (
        "Legacy v2 payload must restore grad_skip_count=0, "
        f"got {trainer3.grad_skip_count}"
    )
