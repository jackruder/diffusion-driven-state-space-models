"""Gluonts-forecast benchmark family (CSDI/TimeGrad NIPS datasets).

Benchmarks the settled DDSSM (additive ``GaussianEncoder`` +
``TransformerFutureSummary`` + latent CSDI-style ``DiffusionTransition``) on the
five GP-copula NIPS datasets. One axis (dataset) → five presets
``gluonts_forecast__<dataset>`` plus a ``gluonts_smoke``. The lean Optuna sweep
``+sweep=gluonts_lean`` tunes latent_dim + LRs + batch on the validation ELBO;
the ``GluonEval`` (CRPS-sum/NLL/MAE/RMSE @ 100 samples) runs on finalists via
``ddssm.evaluate``. Launch the whole study with ``python -m ddssm.launch
gluonts_forecast``.
"""

from . import (
    evals,
    model,
    study,
    report,
    sweeps,
    hparams,
    datasets,
    experiments,
)

__all__ = [
    "datasets",
    "evals",
    "experiments",
    "hparams",
    "model",
    "report",
    "study",
    "sweeps",
]
