"""Bridge the hydra-zen ``experiment_store`` to Hydra's ConfigStore.

Each ``experiments/<name>.py`` module ends with a visible

    experiment_store(exp, name="<name>")

call against the pre-grouped store defined in
:mod:`experiments._registry`. Importing every experiment module
therefore populates that store; we then ask hydra-zen to push the
accumulated entries into Hydra's ConfigStore. The Hydra CLI
(``python -m ddssm.app experiment=NAME``) resolves them like any
other preset.

No per-module ``cs.store(...)`` boilerplate, no attribute-sniffing
walk: registration is *visible in source*.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import sys

log = logging.getLogger(__name__)


def _ensure_experiments_on_path() -> None:
    """Put the repo root on ``sys.path`` so ``import experiments`` works.

    The ``experiments/`` package lives at the repository root, not
    inside ``src/ddssm``. ``python -m ddssm.app`` from the repo root
    needs that root on the path; we add it eagerly so users do not
    have to set ``PYTHONPATH``.
    """
    candidates = [os.getcwd()]
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
    """Import every experiment module, then push the zen-store to Hydra."""
    _ensure_experiments_on_path()
    try:
        import experiments
    except ModuleNotFoundError:
        log.warning(
            "No experiments/ package found on sys.path. "
            "Run from a directory containing experiments/ or set PYTHONPATH."
        )
        return

    for _, name, ispkg in pkgutil.iter_modules(experiments.__path__):
        if ispkg or name.startswith("_"):
            continue
        try:
            importlib.import_module(f"experiments.{name}")
        except Exception as e:  # pragma: no cover -- defensive
            log.warning("Skipping experiments/%s.py: %s", name, e)

    from experiments._registry import experiment_store
    experiment_store.add_to_hydra_store(overwrite_ok=True)


__all__ = ["register_experiments"]
