"""Hydra-zen config-group stores for DDSSM.

Three groups, three stores. Every named preset registers itself with
its store at import time; ``ddssm._experiment_registry`` pushes the
union into Hydra's ConfigStore at CLI startup.

* ``model_store``      — full :class:`DDSSM` compositions (shape +
                          encoder/decoder/z_init/transition baked in).
* ``data_store``       — :class:`Synthetic` / :class:`KDD` data modules.
* ``experiment_store`` — :class:`ExperimentC` instances that tie a
                          model + dataset + training/eval/viz together.

Defined in :file:`verifications.org`; tangled here.
"""

from __future__ import annotations

from hydra_zen import store

model_store = store(group="model")
data_store = store(group="data")
experiment_store = store(group="experiment")

__all__ = ["model_store", "data_store", "experiment_store"]
