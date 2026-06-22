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
    datasets,  # noqa: F401  -- dataset axis
    model,  # noqa: F401  -- registers GluonModel
    hparams,  # noqa: F401  -- stages + hparams
    evals,  # noqa: F401  -- eval + objective specs
    sweeps,  # noqa: F401  -- registers the lean sweep
    study,  # noqa: F401  -- registers the per-dataset presets
    experiments,  # noqa: F401  -- registers gluonts_smoke
    report,  # noqa: F401  -- comparison table
)

__all__ = [
    "datasets",
    "model",
    "hparams",
    "evals",
    "sweeps",
    "study",
    "experiments",
    "report",
]
