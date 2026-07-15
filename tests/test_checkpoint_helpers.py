"""Tests for the reusable checkpoint helpers (module 5 of the adapter refactor).

Two mechanics are shared across every :class:`ModelAdapter` family and thus
extracted from ``checkpoint.py`` as public, importable helpers:

* :func:`atomic_save` — write-to-temp-then-``os.replace`` durable save.
* :func:`check_model_config_drift` — WARN-on-drift model-config cross-check,
  returning a bool so callers pick warn-vs-raise. Understands the
  adapter-wrapper unwrap (a pre-refactor bare checkpoint loaded under a
  post-refactor adapter-wrapped experiment diffs against the ``module:``
  subtree, not the whole wrapper).
"""

from __future__ import annotations

import os
import logging

import torch
from omegaconf import OmegaConf

from ddssm.training.checkpoint import atomic_save, check_model_config_drift

# ---------------------------------------------------------------------------
# atomic_save
# ---------------------------------------------------------------------------


def test_atomic_save_writes_loadable_payload(tmp_path):
    path = str(tmp_path / "obj.pth")
    payload = {"a": torch.arange(3), "b": 7}
    atomic_save(payload, path)

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["b"] == 7
    assert torch.equal(loaded["a"], torch.arange(3))


def test_atomic_save_leaves_no_temp_files(tmp_path):
    path = str(tmp_path / "obj.pth")
    atomic_save({"x": 1}, path)

    # No ``tmp_save_*`` scratch files should survive a successful save.
    leftovers = [n for n in os.listdir(str(tmp_path)) if n.startswith("tmp_save_")]
    assert leftovers == [], f"stray temp files: {leftovers}"


def test_atomic_save_overwrites_existing_path(tmp_path):
    path = str(tmp_path / "obj.pth")
    atomic_save({"v": 1}, path)
    atomic_save({"v": 2}, path)  # atomic replace over the existing file

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["v"] == 2
    leftovers = [n for n in os.listdir(str(tmp_path)) if n.startswith("tmp_save_")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# check_model_config_drift — basic warn/silent/None semantics
# ---------------------------------------------------------------------------


def test_drift_false_on_identical(caplog):
    with caplog.at_level(logging.WARNING, logger="ddssm.training.checkpoint"):
        drift = check_model_config_drift("hidden_dim: 64", "hidden_dim: 64")
    assert drift is False
    assert not any("config drift" in r.message for r in caplog.records)


def test_drift_false_when_saved_none():
    assert check_model_config_drift(None, "hidden_dim: 64") is False


def test_drift_false_when_expected_none():
    assert check_model_config_drift("hidden_dim: 64", None) is False


def test_drift_true_and_warns_on_difference(caplog):
    with caplog.at_level(logging.WARNING, logger="ddssm.training.checkpoint"):
        drift = check_model_config_drift("hidden_dim: 64", "hidden_dim: 80")
    assert drift is True
    assert any("config drift" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# check_model_config_drift — adapter-wrapper unwrap
# ---------------------------------------------------------------------------


def _bare_yaml(hidden_dim: int) -> str:
    """A bare model conf as ``OmegaConf.to_yaml`` would emit it."""
    return OmegaConf.to_yaml(
        OmegaConf.create({
            "_target_": "ddssm.model.dssd.DDSSM",
            "hidden_dim": hidden_dim,
        })
    )


def _wrapper_yaml(hidden_dim: int) -> str:
    """An adapter-wrapper conf nesting the bare model under ``module:``."""
    return OmegaConf.to_yaml(
        OmegaConf.create({
            "_target_": "ddssm.adapters.DDSSMAdapter",
            "module": {
                "_target_": "ddssm.model.dssd.DDSSM",
                "hidden_dim": hidden_dim,
            },
            "config": {"family": "ddssm"},
        })
    )


def test_unwrap_no_drift_when_module_matches_bare(caplog):
    saved = _bare_yaml(64)
    expected = _wrapper_yaml(64)  # module subtree == the saved bare conf
    with caplog.at_level(logging.WARNING, logger="ddssm.training.checkpoint"):
        drift = check_model_config_drift(saved, expected)
    assert drift is False, "unwrap should make bare-vs-module compare equal"
    assert not any("config drift" in r.message for r in caplog.records)


def test_unwrap_diff_is_against_module_subtree(caplog):
    saved = _bare_yaml(64)
    expected = _wrapper_yaml(80)  # module subtree differs (hidden_dim)
    with caplog.at_level(logging.WARNING, logger="ddssm.training.checkpoint"):
        drift = check_model_config_drift(saved, expected)
    assert drift is True
    # The warning must have fired, and its diff must be against the ``module``
    # subtree — not the whole wrapper — so wrapper-only keys never appear.
    msgs = "\n".join(r.getMessage() for r in caplog.records)
    assert "config drift" in msgs
    # Wrapper-only content (the ``config:`` subtree with ``family: ddssm`` and
    # the adapter ``_target_``) must NOT appear in the diff — only the
    # ``module`` subtree was diffed.
    assert "family: ddssm" not in msgs, "wrapper-only 'config' subtree leaked"
    assert "DDSSMAdapter" not in msgs, "wrapper-only _target_ leaked into the diff"
