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
        self.config = SimpleNamespace(checkpoint_dir="ckpts")


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
