"""Objective + eval specs for the CSDI experiment family.

The Optuna SWEEP objective is the **validation loss** read straight from
``metrics.csv`` (``source="csv"``, logged every ``validate_every`` step) — so
``objective_value`` never triggers the eval pipeline. This REQUIRES
``validate_every > 0`` in every preset/sweep variant that uses it.

``CSDIEval`` is the forecast-metric pipeline (CRPS-sum / MAE / RMSE / energy
score). ``CSDISmokeEval`` uses fewer samples (20) to keep the smoke fast; the
solar preset reuses the paper convention (100 samples).
"""

from __future__ import annotations

from ddssm.experiment.builders import Eval, Objective

# Sweep objective: minimise validation ``loss/total`` (the CSDI training loss).
# ``source="csv"`` reads metrics.csv (val rows require validate_every > 0);
# tail_frac=0.1 averages the final 10% of validation points.
CSDIValObjective = Objective(
    metric="loss/total",
    split="val",
    source="csv",
    tail_frac=0.1,
)

# Forecast-metric pipeline. Full sample count for the paper-scale solar preset.
CSDIEval = Eval(
    metrics=["crps_sum", "mae", "rmse", "energy_score"],
    split="test",
    num_samples=100,
    output_filename="metrics.json",
)

# Smoke variant: fewer samples so the CPU forecast sampling stays fast.
CSDISmokeEval = Eval(
    metrics=["crps_sum", "mae", "rmse", "energy_score"],
    split="test",
    num_samples=20,
    output_filename="metrics.json",
)


__all__ = ["CSDIEval", "CSDISmokeEval", "CSDIValObjective"]
