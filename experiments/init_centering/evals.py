"""Objective + eval specs for the init-centering pilot Optuna sweep.

The Phase-C pilot uses the five headline metric primitives from
``init-experiment.org`` § Headline metrics (the ones Phase A landed)
as the post-training eval surface.

The default ``PilotObjective`` (legacy single-objective) optimises
``stage2_elbo_surrogate`` alone. ``PilotMOObjective`` is the
two-axis multi-objective list used by Round-1 sweeps:

  * ``wallclock_to_target_seconds`` (minimise) — seconds from stage-2
    start to first hitting ``loss/total <= target``. ``penalty=
    "csv_tail_time"`` substitutes the trial's full training wall-clock
    when the target is never reached, keeping misses on the same units
    as hits. The target itself is set via the eval's
    ``kwargs.wallclock_to_target.target_value`` Hydra override (the
    overnight script exposes it as ``WALLCLOCK_TARGET``).
  * ``stage2_elbo_surrogate`` (minimise) — depth of final fit.

The Pareto front separates fast-to-target-but-shallow hparams from
slow-but-deep ones. ``wallclock_to_relative_target_seconds`` (time to
90% of the trial's own descent) is computed as a diagnostic but NOT
an Optuna axis — see report.py.
"""

from __future__ import annotations

from ddssm.builders import Eval, Objective

# Legacy single-objective spec (still used by smoke tests and the
# variance probe family). Read from ``metrics.json`` since the metric
# is computed post-training.
PilotObjective = Objective(
    metric="stage2_elbo_surrogate",
    source="json",
)

# Two-axis multi-objective spec. ``Experiment.objective`` accepts a
# list of ObjectiveSpec and returns a list[float] when one is wired
# in; the sweeper's ``direction: [minimize, minimize]`` handles it.
PilotMOObjective = [
    Objective(
        metric="wallclock_to_target_seconds",
        source="json",
        penalty="csv_tail_time",
    ),
    Objective(
        metric="stage2_elbo_surrogate",
        source="json",
    ),
]


# Default ELBO target for the wallclock objective. Round-1 sweeps use
# -30 (identity-class hparams reach easily, zero/mlp are competitive).
# Override via Hydra: ``experiment.eval.kwargs.wallclock_to_target.target_value=...``.
PILOT_WALLCLOCK_TARGET: float = -30.0


# The five Phase-A headline metrics + two diagnostic secondary metrics
# (per ``init-experiment.org`` § Secondary metrics, the trivial subset
# from the grilling decision: #5 q_aux_kl_trajectory and #6
# log_sigma_p2_collapse) + relative-target wallclock as a MOO
# diagnostic (always-defined regardless of fixed-target hit/miss).
PilotEval = Eval(
    metrics=[
        "stage2_elbo_surrogate",
        "sigma_data_drift",
        "wallclock_to_target",
        "wallclock_to_relative_target",
        "crps_sum_latent",
        "gt_latent_jsd",
        "q_aux_kl_trajectory",
        "log_sigma_p2_collapse",
    ],
    split="val",
    num_samples=16,
    output_filename="metrics.json",
    kwargs={
        "wallclock_to_target": {"target_value": PILOT_WALLCLOCK_TARGET},
    },
)


__all__ = [
    "PILOT_WALLCLOCK_TARGET",
    "PilotEval",
    "PilotMOObjective",
    "PilotObjective",
]
