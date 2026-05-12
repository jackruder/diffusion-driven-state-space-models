"""Hydra-zen store for the named experiment presets.

Each ``experiments/<name>.py`` module ends with::

    experiment_store(exp, name="<name>")

That call adds the composed :class:`Experiment` config to a
pre-grouped hydra-zen store. At CLI time
(:mod:`ddssm._experiment_registry`) we import every experiment
module — triggering its ``experiment_store(...)`` call — and then
push the accumulated entries into Hydra's ConfigStore with a single
``store.add_to_hydra_store()`` call.

The result: every preset is reachable as
``python -m ddssm.app experiment=<name>``, and registration is
*visible in source* rather than hidden in an auto-discovery walk.
"""

from __future__ import annotations

from hydra_zen import store

experiment_store = store(group="experiment")

__all__ = ["experiment_store"]
