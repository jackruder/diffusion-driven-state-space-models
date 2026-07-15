"""Named CSDI experiment presets: a windowed smoke + paper-scale solar.

The CSDI baseline runs through the SAME ``Experiment`` workflow as native DDSSM
presets. ``experiments._make.experiment`` detects that ``CSDIModel`` targets a
``ModelAdapter`` subclass and curries ``hparams`` onto it via
``dataclasses.replace(model, config=hparams)`` (no double-wrap).

* :data:`csdi_smoke` — an in-memory WINDOWED smoke (``SmokeData``) with a tiny
  arch (layers=2, channels=32, num_steps=10). NOT the synthetic
  ``init_smoke_simple`` preset: ``SyntheticDataModule.metadata.forecast_split``
  is ``None`` and the CSDI mapping requires it. Run::

      python -m ddssm.app experiment=csdi_smoke experiment.training.steps=50

* :data:`csdi_solar` — GluonTS Solar (L1=168 / L2=24, target_dim=137). Reuses
  the shipped windowed ``Solar`` data preset (``GluonTSDataModule`` sets
  ``forecast_split = L1``). Paper-scale; not exercised in the gate run.

Both csv val-loss objectives REQUIRE ``validate_every > 0``.
"""

from __future__ import annotations

from experiments._make import experiment
from ddssm.data.presets import Solar
from experiments.csdi.data import SmokeData
from experiments.csdi.evals import (
    CSDIEval,
    CSDISmokeEval,
    CSDIValObjective,
)
from experiments.csdi.model import CSDIModel
from ddssm.experiment.stores import experiment_store
from experiments.csdi.hparams import CSDISmokeHparams, CSDISolarHparams
from ddssm.experiment.builders import Training

# ---------------------------------------------------------------------------
# csdi_smoke: tiny windowed in-memory smoke. validate_every > 0 (objective).
# ---------------------------------------------------------------------------
csdi_smoke = experiment(
    data=SmokeData,
    # config is a harmless placeholder; the factory curries hparams onto it.
    model=CSDIModel(config=CSDISmokeHparams),
    hparams=CSDISmokeHparams,
    training=Training(
        steps=200,
        log_every=10,
        validate_every=50,
        checkpoint_every=100,
        amp=False,
    ),
    eval=CSDISmokeEval,
    objective=CSDIValObjective,
)
experiment_store(csdi_smoke, name="csdi_smoke")


# ---------------------------------------------------------------------------
# csdi_solar: paper-scale GluonTS solar (windowed, target_dim=137).
# ---------------------------------------------------------------------------
csdi_solar = experiment(
    data=Solar,
    model=CSDIModel(config=CSDISolarHparams),
    hparams=CSDISolarHparams,
    training=Training(
        steps=20000,
        log_every=100,
        validate_every=500,
        checkpoint_every=2000,
        amp=False,
    ),
    eval=CSDIEval,
    objective=CSDIValObjective,
)
experiment_store(csdi_solar, name="csdi_solar")


__all__ = ["csdi_smoke", "csdi_solar"]
