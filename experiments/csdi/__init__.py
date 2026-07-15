"""CSDI experiment family (re-vendored CSDI forecaster behind the adapter seam).

Registers the CSDI baseline as first-class ``Experiment`` presets so it trains /
evaluates / sweeps through the same workflow as the native DDSSM presets. One
in-memory WINDOWED smoke (``csdi_smoke``) for pipeline validation plus a
paper-scale GluonTS solar preset (``csdi_solar``). The lean Optuna sweep
``+sweep=csdi_lean`` tunes lr + channels + layers + batch on the validation loss.

    python -m ddssm.app experiment=csdi_smoke experiment.training.steps=50
"""

from . import (
    data,
    evals,
    model,
    sweeps,
    hparams,
    experiments,
)

__all__ = [
    "data",
    "evals",
    "experiments",
    "hparams",
    "model",
    "sweeps",
]
