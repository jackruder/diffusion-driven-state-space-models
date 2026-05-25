"""Objective + eval specs for the init-centering pilot Optuna sweep.

The Phase-C pilot uses the five headline metric primitives from
``init-experiment.org`` § Headline metrics (the ones Phase A landed)
as the post-training eval surface.  The Optuna objective is
``stage2_elbo_surrogate`` — the doc's "metric (2) is the comparison
objective" — read from ``metrics.json`` after evaluation.
"""

from __future__ import annotations

from ddssm.builders import Eval, Objective


# Read from ``metrics.json`` (not ``metrics.csv``) — the Phase-A
# eval metric is computed post-training and surfaces as a JSON scalar.
PilotObjective = Objective(
    metric="stage2_elbo_surrogate",
    source="json",
)


# The five Phase-A headline metrics, computed on the val split.
PilotEval = Eval(
    metrics=[
        "stage2_elbo_surrogate",
        "sigma_data_drift",
        "wallclock_to_target",
        "crps_sum_latent",
        "gt_latent_jsd",
    ],
    split="val",
    num_samples=16,
    output_filename="metrics.json",
)


__all__ = ["PilotEval", "PilotObjective"]
