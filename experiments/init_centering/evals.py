"""Objective + eval specs for the init-centering pilot Optuna sweep.

The eval surface (:data:`PilotEval`) is the five headline metric
primitives from ``init-experiment.org`` § Headline metrics (the ones
Phase A landed) plus two diagnostic secondary metrics and the
relative-target wallclock diagnostic — see :data:`PilotEval` for the
full breakdown.

The default ``PilotObjective`` (legacy single-objective) optimises
``stage2_elbo_surrogate`` alone. ``PilotMOObjective`` is the
two-axis multi-objective list used by Round-2 sweeps:

  * ``wallclock_to_target_step`` (minimise) — training *steps* to first
    hitting ``loss/total <= target``. ``penalty="csv_tail_step"``
    substitutes the trial's full step budget when the target is never
    reached, keeping misses on the same (step) units as hits. Steps are
    contention-invariant — unlike the round-1 ``wallclock_to_target_seconds``
    axis they survive GPU packing / per-cell hardware differences. The
    target is set via the eval's ``kwargs.wallclock_to_target.target_value``
    Hydra override.
  * ``stage2_elbo_surrogate`` (minimise) — depth of final fit.

The Pareto front separates fast-to-target-but-shallow hparams from
slow-but-deep ones. ``wallclock_to_relative_target_*`` (to 90% of the
trial's own descent) is computed as a diagnostic but NOT an Optuna
axis — see report.py.
"""

from __future__ import annotations

from ddssm.experiment.builders import Eval, Objective, Objectives

# Legacy single-objective spec (still used by smoke tests and the
# variance probe family). Read from ``metrics.json`` since the metric
# is computed post-training.
PilotObjective = Objective(
    metric="stage2_elbo_surrogate",
    source="json",
)

# Two-axis multi-objective spec. ``Objectives`` wraps the ordered list
# so Hydra-zen instantiates each ``ObjectiveSpec`` properly; the
# returned list[float] is matched against the sweeper's
# ``direction: [minimize, minimize]``.
PilotMOObjective = Objectives(
    specs=[
        Objective(
            metric="wallclock_to_target_step",
            source="json",
            penalty="csv_tail_step",
        ),
        Objective(
            metric="stage2_elbo_surrogate",
            source="json",
        ),
    ]
)


# Default ELBO target for the wallclock objective: steps to ``loss/total <= target``.
# WAS -100.0 (round2), which assumed the trans-KL fix would push convergent ELBO
# into a NEGATIVE -100..-600 band. That premise was wrong: the observed convergent
# ``loss/total`` stays POSITIVE (~50-330; the obj1 tail-mean band is ~174-257 across
# cells), so ``<= -100`` was NEVER reached -> every trial took the csv_tail_step
# penalty (~5000) -> obj0 collapsed to a flat ~5005-5500 and the MOO degenerated to
# single-objective on obj1 (round2 +trials run, 2026-06-01). 250 sits mid-band,
# so good configs cross it at genuinely different steps (a real speed axis) while
# weak cells that never reach it keep the penalty. Override via Hydra:
# ``experiment.eval.kwargs.wallclock_to_target.target_value=...``.
PILOT_WALLCLOCK_TARGET: float = 250.0


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
