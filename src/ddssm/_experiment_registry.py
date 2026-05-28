"""Bridge the hydra-zen ``store`` singleton to Hydra's ConfigStore.

Importing :mod:`experiments` runs every ``<store>(thing, name=...)``
call in :mod:`experiments.datasets` (library dataset presets) and the
:mod:`experiments.init_centering` family. The default
:obj:`hydra_zen.store` singleton then holds the full registry across
every group; one ``store.add_to_hydra_store()`` call publishes the
whole thing into
Hydra's :class:`ConfigStore`. The Hydra CLI
(``python -m ddssm.app experiment=NAME model=NAME data=NAME ...``)
resolves them like any preset.
"""

from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger(__name__)


def _ensure_experiments_on_path() -> None:
    """Put the repo root on ``sys.path`` so ``import experiments`` works."""
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
    """Import the experiments package and push every store entry to Hydra."""
    _ensure_experiments_on_path()
    try:
        import experiments  # noqa: F401 — its __init__ imports every subpackage
    except ModuleNotFoundError:
        log.warning(
            "No experiments/ package found on sys.path. "
            "Run from a directory containing experiments/ or set PYTHONPATH."
        )
        return

    from hydra_zen import store
    store.add_to_hydra_store(overwrite_ok=True)


__all__ = ["register_experiments"]
