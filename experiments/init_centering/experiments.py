"""Named init-centering experiments: two role-specific smokes + the ablation study.

The two smoke presets are the canonical entry points:

- :data:`init_smoke_simple` — ``(zero, fixed)`` on the 1D ablation
  dataset. Minimum surface + numerical V2 anchor. Use to validate the
  pipeline is wired correctly without changing any cell-machinery
  defaults.
- :data:`init_smoke_high_surface` — ``(persistence, per_t)`` on the MV
  ablation dataset. Exercises the parameter-free persistence baseline,
  per-t σ_data EMA, and the multivariate observation lift.

The ablation grid itself is a first-class :class:`~ddssm.cluster.study.Study`
(``experiments.init_centering.study.INIT_CENTERING_STUDY``): 4 cells × 2
datasets = 8 registered presets named ``init_<cell>__<dataset>`` (e.g.
``init_persistence_per_t__1d``). Registration, launching (via
``python -m ddssm.launch init_centering`` — ADR-0007/0008), and reporting all
flow through that Study.
"""

from __future__ import annotations

from experiments._make import experiment
from ddssm.experiment.stores import experiment_store
from experiments.init_centering.data import (
    NonlinBimodalLift1D,
    NonlinBimodalLiftMV,
)
from experiments.init_centering.evals import (
    PilotEval,
    PilotObjective,
)
from experiments.init_centering.model import SmokeModel
from experiments.init_centering.hparams import Training800, SmokeHparams

# ---------------------------------------------------------------------------
# Simple-smoke cell: (zero, fixed) on the 1D ablation dataset.
# No eval wired; ``train()`` returns the trainer for inspection.
#
# Run: ``python -m ddssm.app experiment=init_smoke_simple``.
# ---------------------------------------------------------------------------

init_smoke_simple = experiment(
    data=NonlinBimodalLift1D,
    model=SmokeModel(
        baseline_form="zero",
        tracking_mode="fixed",
        latent_dim=1,
        data_dim=1,
    ),
    hparams=SmokeHparams,
    training=Training800,
)
experiment_store(init_smoke_simple, name="init_smoke_simple")


# ---------------------------------------------------------------------------
# High-surface-smoke cell: (persistence, per_t) on the MV ablation dataset.
# Exercises the persistence baseline + per-t σ_data EMA + the MV
# observation lift. Eval + objective wired so it can also act as a
# single-trial inspection target before launching the full sweep.
#
# Run a single trial: ``python -m ddssm.app experiment=init_smoke_high_surface``
# ---------------------------------------------------------------------------

init_smoke_high_surface = experiment(
    data=NonlinBimodalLiftMV,
    model=SmokeModel(
        baseline_form="persistence",
        tracking_mode="per_t",
        latent_dim=4,
        data_dim=8,
    ),
    hparams=SmokeHparams,
    training=Training800,
    eval=PilotEval,
    objective=PilotObjective,
)
experiment_store(init_smoke_high_surface, name="init_smoke_high_surface")


# The study's cell points are published to ``experiment_store`` by
# ``register_study(..., into=experiment_store)`` in ``study.py``. That runs at
# import time when the package ``__init__`` imports the ``study`` submodule, so
# this module needs no separate call.
