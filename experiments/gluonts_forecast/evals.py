"""Objective + eval specs for the gluonts_forecast family.

The Optuna SWEEP objective is the **validation ELBO**, read straight from
``metrics.csv`` (logged every ``validate_every`` step) — ``source="csv"`` so
``objective_value`` never triggers the eval pipeline. The sweep is therefore
training-bound: no per-trial forecast sampling.

``GluonEval`` is the publication metric pipeline (CRPS-sum / NLL / MAE / RMSE at
100 samples on the held-out test split). It is run by ``python -m ddssm.evaluate``
on the per-dataset FINALISTS only — never every trial.
"""

from __future__ import annotations

from ddssm.experiment.builders import Eval, Objective

# Sweep objective: minimise validation -ELBO. ``split="val"`` filters metrics.csv
# rows; ``loss/total`` is the per-step (λ-weighted) objective the trainer logs.
# tail_frac=0.1 averages the final 10% of validation points (de-noises the pick).
ValElboObjective = Objective(
    metric="loss/total",
    split="val",
    source="csv",
    tail_frac=0.1,
)

# Finalist eval — run via ``ddssm.evaluate`` on the best checkpoint(s) per dataset.
# num_samples=100 is the reported convention. (De-normalization for CSDI-scale
# comparability is handled separately — see the de-normalize-metrics task.)
GluonEval = Eval(
    metrics=["crps_sum", "nll", "mae", "rmse"],
    split="test",
    num_samples=100,
    output_filename="metrics.json",
)

__all__ = ["GluonEval", "ValElboObjective"]
