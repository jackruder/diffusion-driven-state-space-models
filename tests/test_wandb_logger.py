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

from ddssm.training.loggers import WandbLogger


@pytest.fixture
def fake_wandb(monkeypatch):
    """Install a stub ``wandb`` module so ``WandbLogger`` runs offline.

    Includes the surface area the post-2026-05 logger uses:
    ``watch`` for grad/param histograms, ``Artifact`` + ``log_artifact``
    for the close()-time checkpoint upload, and ``run.id`` for the
    cross-stage reconnect path.
    """
    fake_artifact = MagicMock(name="Artifact_instance")
    fake = SimpleNamespace(
        init=MagicMock(),
        log=MagicMock(),
        finish=MagicMock(),
        define_metric=MagicMock(),
        watch=MagicMock(),
        Artifact=MagicMock(return_value=fake_artifact),
        log_artifact=MagicMock(),
        Image=MagicMock(side_effect=lambda p: f"<Image:{p}>"),
        run=SimpleNamespace(id="testrunid123"),
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
    from ddssm.experiment import Experiment

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


# ---------------------------------------------------------------------------
# Cross-stage reconnect: the train logger persists its run-id; standalone
# eval/viz/variance stages re-init the same W&B run via that id file.
# ---------------------------------------------------------------------------


def test_persist_run_id_writes_dotfile(fake_wandb, tmp_path):
    """Constructor must drop ``.wandb_run_id`` under ``run_dir``."""
    WandbLogger(project="p", run_dir=str(tmp_path), enabled=True)
    id_path = tmp_path / WandbLogger._RUN_ID_FILENAME
    assert id_path.is_file()
    assert id_path.read_text().strip() == "testrunid123"


def test_persist_run_id_skipped_without_run_dir(fake_wandb):
    """No run_dir → no dotfile attempt (no exception either)."""
    WandbLogger(project="p", enabled=True)
    # If _persist_run_id were called, the mock would record run access;
    # we can't easily assert non-call, but the test passes iff init works.


def test_resume_run_from_dir_happy_path(fake_wandb, tmp_path):
    """``resume_run_from_dir`` reinit-with-id when the dotfile is present."""
    from ddssm.training.loggers import resume_run_from_dir

    (tmp_path / WandbLogger._RUN_ID_FILENAME).write_text("savedid42")

    mod = resume_run_from_dir(
        str(tmp_path), {"enabled": True, "project": "p", "entity": "e"},
    )
    assert mod is fake_wandb
    init_kwargs = fake_wandb.init.call_args.kwargs
    assert init_kwargs["id"] == "savedid42"
    assert init_kwargs["resume"] == "allow"
    assert init_kwargs["dir"] == str(tmp_path)
    assert init_kwargs["project"] == "p"


def test_resume_run_from_dir_returns_none_when_disabled(fake_wandb, tmp_path):
    from ddssm.training.loggers import resume_run_from_dir

    (tmp_path / WandbLogger._RUN_ID_FILENAME).write_text("x")
    assert resume_run_from_dir(str(tmp_path), None) is None
    assert resume_run_from_dir(str(tmp_path), {"enabled": False}) is None
    fake_wandb.init.assert_not_called()


def test_resume_run_from_dir_returns_none_without_dotfile(fake_wandb, tmp_path):
    """No persisted run-id → no resume (returns None)."""
    from ddssm.training.loggers import resume_run_from_dir

    assert resume_run_from_dir(
        str(tmp_path), {"enabled": True, "project": "p"},
    ) is None
    fake_wandb.init.assert_not_called()


# ---------------------------------------------------------------------------
# wandb.watch: opt-in via watch_log kwarg; the logger never breaks training.
# ---------------------------------------------------------------------------


def test_watch_model_calls_wandb_watch(fake_wandb):
    import torch.nn as nn

    logger = WandbLogger(
        project="p",
        watch_log="gradients",
        watch_log_freq=50,
        enabled=True,
    )
    model = nn.Linear(3, 1)
    logger.watch_model(model)
    fake_wandb.watch.assert_called_once_with(
        model, log="gradients", log_freq=50,
    )


def test_watch_model_noop_without_watch_log(fake_wandb):
    """Default ``watch_log=None`` → no wandb.watch call."""
    import torch.nn as nn

    logger = WandbLogger(project="p", enabled=True)
    logger.watch_model(nn.Linear(3, 1))
    fake_wandb.watch.assert_not_called()


def test_watch_model_swallows_errors(fake_wandb):
    """``wandb.watch`` failure must not propagate (best-effort)."""
    import torch.nn as nn

    fake_wandb.watch.side_effect = RuntimeError("simulated wandb failure")
    logger = WandbLogger(project="p", watch_log="all", enabled=True)
    # Must not raise.
    logger.watch_model(nn.Linear(3, 1))


# ---------------------------------------------------------------------------
# close()-time artifact upload: ckpt_latest.pth + resolved_config.yaml.
# ---------------------------------------------------------------------------


def test_close_uploads_ckpt_and_config_artifacts(fake_wandb, tmp_path):
    """close() must upload ckpt + config as wandb.Artifact when files exist."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "ckpt_latest.pth").write_bytes(b"fake-ckpt-bytes")
    (tmp_path / "resolved_config.yaml").write_text("foo: bar\n")

    logger = WandbLogger(project="p", run_dir=str(tmp_path), enabled=True)
    logger.close()

    # Two artifacts: one for the model ckpt, one for the resolved config.
    assert fake_wandb.Artifact.call_count == 2
    artifact_kinds = {
        call.kwargs["type"] for call in fake_wandb.Artifact.call_args_list
    }
    assert artifact_kinds == {"model", "config"}
    # Each artifact gets log_artifact'd.
    assert fake_wandb.log_artifact.call_count == 2
    # And finish() runs after artifacts.
    fake_wandb.finish.assert_called_once()


def test_close_skips_missing_artifact_files(fake_wandb, tmp_path):
    """close() must NOT attempt to upload an artifact that doesn't exist."""
    # Only the config is present; ckpt is absent.
    (tmp_path / "resolved_config.yaml").write_text("foo: 1\n")

    logger = WandbLogger(project="p", run_dir=str(tmp_path), enabled=True)
    logger.close()

    # Only the config artifact got built.
    assert fake_wandb.Artifact.call_count == 1
    assert fake_wandb.Artifact.call_args.kwargs["type"] == "config"


def test_close_without_run_dir_skips_artifacts(fake_wandb):
    """No run_dir → no artifact-upload attempt at close()."""
    logger = WandbLogger(project="p", enabled=True)
    logger.close()
    fake_wandb.Artifact.assert_not_called()
    fake_wandb.log_artifact.assert_not_called()
    fake_wandb.finish.assert_called_once()


def test_close_artifact_failure_does_not_break_finish(fake_wandb, tmp_path):
    """A failed log_artifact must not prevent wandb.finish() from running.

    The trainer's ``finally:`` clean-up relies on close() always
    completing — masking that path with a transient W&B error would
    leak processes and hide the underlying training failure.
    """
    (tmp_path / "resolved_config.yaml").write_text("x: 1\n")
    fake_wandb.log_artifact.side_effect = RuntimeError("simulated upload fail")

    logger = WandbLogger(project="p", run_dir=str(tmp_path), enabled=True)
    logger.close()

    fake_wandb.finish.assert_called_once()
