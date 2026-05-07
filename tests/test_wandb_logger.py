"""Behavioral tests for the W&B logger and the experiment-level wandb wiring.

The ``wandb`` package is mocked so we can run these tests offline. We
verify two things:

* Step-axis fix: ``on_step`` and ``on_epoch`` no longer fight over W&B's
  single per-run step counter. ``on_step`` logs put the trainer's
  ``global_step`` into ``train_step``; ``on_epoch`` logs put the epoch
  index into ``epoch``. Neither call passes ``step=`` to ``wandb.log``.

* Disabled wiring: ``Experiment.run`` resolves ``wandb_config`` with
  ``enabled=False`` to ``None`` so the trainer doesn't try to import
  or call ``wandb`` at all.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ddssm.loggers import WandbLogger


@pytest.fixture
def fake_wandb(monkeypatch):
    """Install a stub ``wandb`` module so ``WandbLogger`` runs offline."""
    fake = SimpleNamespace(
        init=MagicMock(),
        log=MagicMock(),
        finish=MagicMock(),
        define_metric=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_disabled_short_circuits(fake_wandb):
    logger = WandbLogger(enabled=False)
    logger.on_step("train", 5, {"loss/total": 1.0})
    logger.on_epoch("val", 1, {"loss/total": 0.5})
    logger.close()
    fake_wandb.init.assert_not_called()
    fake_wandb.log.assert_not_called()


def test_step_axis_separation(fake_wandb):
    logger = WandbLogger(project="p", enabled=True)
    fake_wandb.init.assert_called_once()
    # define_metric must be called once per declared axis + per namespace.
    assert fake_wandb.define_metric.call_count >= 5

    logger.on_step("train", 100, {"loss/total": 0.1})
    logger.on_epoch("val", 3, {"loss/total": 0.2})

    # No ``step=`` kwarg in either log call -- W&B uses the embedded
    # ``train_step`` / ``epoch`` keys for monotonic ordering.
    for call in fake_wandb.log.call_args_list:
        assert "step" not in call.kwargs

    train_payload = fake_wandb.log.call_args_list[0].args[0]
    assert "train/loss/total" in train_payload
    assert train_payload["train_step"] == 100

    epoch_payload = fake_wandb.log.call_args_list[1].args[0]
    assert "epoch/val/loss/total" in epoch_payload
    assert epoch_payload["epoch"] == 3


def test_run_dir_forwarded_to_wandb_init(fake_wandb):
    WandbLogger(project="p", run_dir="/tmp/some_run", enabled=True)
    init_kwargs = fake_wandb.init.call_args.kwargs
    assert init_kwargs.get("dir") == "/tmp/some_run"


def test_experiment_disabled_wandb_returns_none(tmp_path):
    """Experiment._wandb_kwargs returns None when wandb_config is disabled."""
    from ddssm.experiment import Experiment, TrainingScalars

    expt = Experiment.__new__(Experiment)  # bypass dataclass __init__
    expt.wandb_config = {"enabled": False, "project": "p"}
    assert expt._wandb_kwargs(str(tmp_path)) is None

    expt.wandb_config = None
    assert expt._wandb_kwargs(str(tmp_path)) is None


def test_experiment_enabled_wandb_passes_through(tmp_path):
    from ddssm.experiment import Experiment

    expt = Experiment.__new__(Experiment)
    expt.wandb_config = {"enabled": True, "project": "myproj", "tags": ["a"]}
    out = expt._wandb_kwargs(str(tmp_path))
    assert out is not None
    assert out["project"] == "myproj"
    assert out["tags"] == ["a"]
    assert out["run_dir"] == str(tmp_path)
