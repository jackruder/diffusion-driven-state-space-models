"""Tests for the checkpoint module (ADR-0005): payload schema, cross-check, EMA."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch
import torch.nn as nn
import pytest

from ddssm.checkpoint import Checkpoint, load_into_model, prepare_model


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

    with caplog.at_level(logging.WARNING, logger="ddssm.checkpoint"):
        load_into_model(
            _Toy(), path, device=torch.device("cpu"),
            expected_model_config_yaml="hidden_dim: 80",
        )
    assert any("config drift" in r.message for r in caplog.records)


def test_cross_check_silent_when_match(tmp_path, caplog):
    model = _Toy()
    path = str(tmp_path / "ckpt.pth")
    Checkpoint.from_trainer(_fake_trainer(model, yaml="hidden_dim: 64")).save(path)

    with caplog.at_level(logging.WARNING, logger="ddssm.checkpoint"):
        load_into_model(
            _Toy(), path, device=torch.device("cpu"),
            expected_model_config_yaml="hidden_dim: 64",
        )
    assert not any("config drift" in r.message for r in caplog.records)


def test_load_ema_swaps_transition(tmp_path):
    model = _Toy()
    # Live transition weights = 0; EMA shadow = 1. The payload records both.
    with torch.no_grad():
        for p in model.transition.parameters():
            p.zero_()
    ema_shadow = {k: torch.ones_like(v) for k, v in model.transition.state_dict().items()}
    trainer = _fake_trainer(model)
    trainer.ema = SimpleNamespace(shadow=ema_shadow)
    path = str(tmp_path / "ckpt.pth")
    Checkpoint.from_trainer(trainer).save(path)

    # load_ema=False → live (zero) transition weights.
    live = _Toy()
    load_into_model(live, path, device=torch.device("cpu"), load_ema=False)
    assert torch.allclose(live.transition.weight, torch.zeros_like(live.transition.weight))

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
    ema_shadow = {k: torch.ones_like(v) for k, v in model.transition.state_dict().items()}
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
        exp_live, checkpoint_path=path, device=torch.device("cpu"), load_ema=False,
    )
    assert torch.allclose(m_live.transition.weight, torch.zeros_like(m_live.transition.weight))


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
    from ddssm.train import DDSSMTrainer

    model = make_small_model()
    trainer = DDSSMTrainer(
        model=model, device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"), quiet=True,
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
