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

import os
import sys
import logging

log = logging.getLogger(__name__)


def _find_repo_root() -> str | None:
    """Walk up from this file to find the repo root containing ``experiments/``."""
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "experiments")):
            return here
    return None


def _ensure_experiments_on_path() -> None:
    """Put the repo root first on ``sys.path`` for ``import experiments``.

    Derived from this file's location (never the cwd), so a same-named
    ``experiments/`` directory in the working directory cannot shadow the
    repo's package.
    """
    repo_root = _find_repo_root()
    if repo_root is None:
        log.debug(
            "Could not locate repo root (experiments/ dir) relative to %s; "
            "sys.path unchanged.",
            __file__,
        )
        return
    # Insert at front so our experiments/ shadows any cwd-based shadow.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    else:
        # Already present — move to front to guarantee priority.
        sys.path.remove(repo_root)
        sys.path.insert(0, repo_root)

    # Sanity-check: if ``experiments`` is already imported, verify it came
    # from the expected location so we surface shadowing bugs eagerly.
    if "experiments" in sys.modules:
        mod = sys.modules["experiments"]
        mod_file = getattr(mod, "__file__", None) or getattr(
            mod, "__path__", [None]
        )[0]
        if mod_file and not os.path.abspath(mod_file).startswith(
            repo_root + os.sep
        ):
            log.warning(
                "experiments module already imported from %s, expected under %s. "
                "A shadowing package may have been imported first.",
                mod_file,
                repo_root,
            )


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
