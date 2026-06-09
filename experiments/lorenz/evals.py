"""Eval and objective specs for the Lorenz sweep family.

Single-objective ``LorenzObjective`` minimises ``stage2_elbo_surrogate``.
Multi-objective ``LorenzMOObjective`` adds ``wallclock_to_target_step``
(steps to first reaching ``loss/total <= LORENZ_WALLCLOCK_TARGET``) so the
Pareto front separates fast-but-shallow configs from slow-but-deep ones.

``LORENZ_WALLCLOCK_TARGET = 120.0`` sits between the observed convergent
bands: lorenz_low_lr landed at 124.0 and lorenz_gaussian_full at 115.4,
so good diffusion configs should cross 120 at genuinely different step
counts while weak configs take the csv_tail_step penalty.
"""

from __future__ import annotations

from ddssm.experiment.builders import Eval, Objective, Objectives

LORENZ_WALLCLOCK_TARGET: float = 120.0

LorenzObjective = Objective(
    metric="stage2_elbo_surrogate",
    source="json",
)

LorenzMOObjective = Objectives(specs=[
    Objective(
        metric="wallclock_to_target_step",
        source="json",
        penalty="csv_tail_step",
    ),
    Objective(
        metric="stage2_elbo_surrogate",
        source="json",
    ),
])

LorenzEval = Eval(
    metrics=[
        "stage2_elbo_surrogate",
        "sigma_data_drift",
        "wallclock_to_target",
        "wallclock_to_relative_target",
    ],
    split="val",
    num_samples=16,
    output_filename="metrics.json",
    kwargs={
        "wallclock_to_target": {"target_value": LORENZ_WALLCLOCK_TARGET},
    },
)

__all__ = [
    "LORENZ_WALLCLOCK_TARGET",
    "LorenzEval",
    "LorenzMOObjective",
    "LorenzObjective",
]
