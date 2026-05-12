"""Hydra-zen ``store(group=...)`` handles — one per axis of variation.

Every store below is a pre-grouped partial of the default
:obj:`hydra_zen.store` singleton, so calls like
``model_store(SmallGauss, name="small_gauss")`` accumulate into the
same underlying registry. One :func:`store.add_to_hydra_store` call
(in :func:`ddssm._experiment_registry.register_experiments`)
publishes the whole registry into Hydra's ``ConfigStore``.

The full granularity is here so any axis can carry named CLI handles
(e.g. ``python -m ddssm.app experiment=harmonic_gauss
encoder=small_csdi``). Prune the stores you don't need.
"""

from __future__ import annotations

from hydra_zen import store

# Composed top-level configs
model_store = store(group="model")
data_store = store(group="data")
experiment_store = store(group="experiment")

# Sub-components of a DDSSM model
encoder_store = store(group="encoder")
decoder_store = store(group="decoder")
z_init_store = store(group="z_init")
transition_store = store(group="transition")

# Sub-components of a transition (CSDI U-Net + noise schedule)
unet_store = store(group="unet")
schedule_store = store(group="schedule")

# Training-time configs
hparams_store = store(group="hparams")
training_store = store(group="training")

# Evaluation / visualization specs
eval_store = store(group="eval")
viz_store = store(group="viz")

# Optimization-driver / variance-probe specs
objective_store = store(group="objective")
variance_store = store(group="variance")

# Multirun / Optuna sweep presets. Entries are merged at root via
# ``package="_global_"`` so each preset can set top-level
# ``hydra.sweeper.*`` keys (matching the legacy ``# @package _global_``
# YAML semantic). Activate with ``+sweep=<name>``.
sweep_store = store(group="sweep", package="_global_")


__all__ = [
    "model_store", "data_store", "experiment_store",
    "encoder_store", "decoder_store", "z_init_store", "transition_store",
    "unet_store", "schedule_store",
    "hparams_store", "training_store",
    "eval_store", "viz_store",
    "objective_store", "variance_store",
    "sweep_store",
]
