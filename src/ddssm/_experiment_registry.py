"""Discover and register notebook-style experiment configs for Hydra.

Each ``experiments/<name>.py`` file at the top of the repository
exposes a module-level ``exp`` variable holding an already-composed
:class:`Experiment` builds-dataclass. This module walks that package
on import, registers every ``exp`` it finds with Hydra's ConfigStore
under ``group="experiment", name="<module name>"``, and then the Hydra
CLI (``python -m ddssm.app experiment=NAME``) resolves it like any
ordinary preset.

There is no separate ``store()`` call needed inside an experiment
file — adding ``experiments/foo.py`` with an ``exp`` variable is
enough.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import sys

from hydra.core.config_store import ConfigStore

log = logging.getLogger(__name__)


def _ensure_experiments_on_path() -> None:
    """Put the repo root on ``sys.path`` so ``import experiments`` works.

    The ``experiments/`` package lives at the repository root, not
    inside ``src/ddssm``. ``python -m ddssm.app`` from the repo root
    needs that root on the path; we add it eagerly so users do not
    have to set ``PYTHONPATH``.
    """
    candidates = [os.getcwd()]
    # When installed editable, walk up from this file to find the repo root
    # by looking for an ``experiments`` sibling.
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "experiments")):
            candidates.append(here)
            break
    for path in candidates:
        if path and path not in sys.path:
            sys.path.insert(0, path)


def register_experiments() -> None:
    """Import every ``experiments/<name>.py`` and register ``exp``.

    Idempotent: re-running is a no-op for entries already in the
    ConfigStore.
    """
    _ensure_experiments_on_path()
    try:
        import experiments  # noqa: F401  -- triggers package init
    except ModuleNotFoundError:
        log.warning(
            "No experiments/ package found on sys.path. "
            "Run from a directory containing experiments/ or set PYTHONPATH."
        )
        return

    cs = ConfigStore.instance()
    for _, name, ispkg in pkgutil.iter_modules(experiments.__path__):
        if ispkg or name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"experiments.{name}")
        except Exception as e:  # pragma: no cover -- defensive
            log.warning("Skipping experiments/%s.py: %s", name, e)
            continue
        if not hasattr(mod, "exp"):
            continue
        cs.store(group="experiment", name=name, node=mod.exp)
        log.debug("Registered experiment=%s", name)


__all__ = ["register_experiments"]
