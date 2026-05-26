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


# The five Phase-A headline metrics + two diagnostic secondary metrics
# (per ``init-experiment.org`` § Secondary metrics, the trivial subset
# from the grilling decision: #5 q_aux_kl_trajectory and #6
# log_sigma_p2_collapse).
PilotEval = Eval(
    metrics=[
        "stage2_elbo_surrogate",
        "sigma_data_drift",
        "wallclock_to_target",
        "crps_sum_latent",
        "gt_latent_jsd",
        "q_aux_kl_trajectory",
        "log_sigma_p2_collapse",
    ],
    split="val",
    num_samples=16,
    output_filename="metrics.json",
)


__all__ = ["PilotEval", "PilotObjective"]
