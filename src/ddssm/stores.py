"""Hydra-zen ``store(group=...)`` handles — one per axis of variation.

Every store below is a pre-grouped partial of the default
:obj:`hydra_zen.store` singleton, so calls like
``model_store(SmallGauss, name="small_gauss")`` accumulate into the
same underlying registry. One :func:`store.add_to_hydra_store` call
(in :func:`ddssm._experiment_registry.register_experiments`)
publishes the whole registry into Hydra's ``ConfigStore``.

Only the axes that actually carry named presets today are kept. The
model is composed in Python (see ``experiments/init_centering/model.py``),
not selected via per-sub-component CLI groups, so finer-grained stores
(``encoder``/``decoder``/``transition``/``unet``/…) were dead scaffolding —
defined and exported but never populated, which made e.g. ``encoder=…``
look CLI-selectable when it selected nothing. Add a store back here if and
when a real preset populates that axis.
"""

from __future__ import annotations

from hydra_zen import store

# Composed top-level configs
model_store = store(group="model")
# ``package="experiment.data"`` so a ``+data=NAME`` selection overrides the
# dataset baked into the chosen experiment preset, instead of writing an
# unread top-level ``data:`` key.
data_store = store(group="data", package="experiment.data")
experiment_store = store(group="experiment")

# Multirun / Optuna sweep presets. Entries are merged at root via
# ``package="_global_"`` so each preset can set top-level
# ``hydra.sweeper.*`` keys (matching the legacy ``# @package _global_``
# YAML semantic). Activate with ``+sweep=<name>``.
sweep_store = store(group="sweep", package="_global_")


__all__ = [
    "model_store", "data_store", "experiment_store",
    "sweep_store",
]
