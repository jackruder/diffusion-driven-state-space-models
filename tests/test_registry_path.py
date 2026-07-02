"""Tests for cwd-shadowing fix in _ensure_experiments_on_path.

Verifies that the repo's own experiments/ package takes sys.path priority
even when a decoy experiments/ directory exists earlier on sys.path.
"""

from __future__ import annotations

import sys
import os
import importlib
import types

import pytest


@pytest.fixture()
def clean_sys_path(tmp_path):
    """Yield with a clean sys.path (excluding any experiments entries) and restore after."""
    original_path = sys.path[:]
    original_modules = dict(sys.modules)
    # Remove any existing experiments from path / modules so each test is isolated.
    sys.path = [p for p in sys.path if not p.endswith(os.sep + "experiments")]
    sys.modules.pop("experiments", None)
    yield tmp_path
    sys.path[:] = original_path
    # Restore modules: remove any new ones, re-add removed ones.
    to_remove = [k for k in sys.modules if k not in original_modules]
    for k in to_remove:
        del sys.modules[k]
    sys.modules.update(original_modules)


def test_repo_root_inserted_at_front(clean_sys_path) -> None:
    """_ensure_experiments_on_path inserts repo root at index 0."""
    from ddssm.experiment.registry import _ensure_experiments_on_path, _find_repo_root

    repo_root = _find_repo_root()
    assert repo_root is not None, "Repo root should be findable from package location"

    _ensure_experiments_on_path()

    assert sys.path[0] == repo_root


def test_decoy_experiments_does_not_shadow(clean_sys_path, tmp_path) -> None:
    """A decoy experiments/ on sys.path is shadowed by the real repo root."""
    from ddssm.experiment.registry import _ensure_experiments_on_path, _find_repo_root

    repo_root = _find_repo_root()
    assert repo_root is not None

    # Create a decoy experiments package in tmp_path.
    decoy_pkg = tmp_path / "experiments"
    decoy_pkg.mkdir()
    (decoy_pkg / "__init__.py").write_text("DECOY = True\n")

    # Insert decoy BEFORE calling _ensure_experiments_on_path.
    sys.path.insert(0, str(tmp_path))

    _ensure_experiments_on_path()

    # Repo root must be at the front (before the decoy).
    assert sys.path[0] == repo_root
    # Repo root must appear before tmp_path.
    repo_idx = sys.path.index(repo_root)
    decoy_idx = sys.path.index(str(tmp_path))
    assert repo_idx < decoy_idx, (
        f"repo_root at index {repo_idx} should precede decoy at {decoy_idx}"
    )


def test_find_repo_root_locates_experiments_dir() -> None:
    """_find_repo_root() returns a directory that actually contains experiments/."""
    from ddssm.experiment.registry import _find_repo_root

    root = _find_repo_root()
    assert root is not None
    assert os.path.isdir(os.path.join(root, "experiments"))
