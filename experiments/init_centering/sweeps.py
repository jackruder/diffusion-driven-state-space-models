"""Optuna sweep presets for the init-centering family.

The init-centering ablation sweeps two handoff-protocol knobs, two
regulariser strengths, and a base-LR plus two per-group multipliers
(7 dims for Learnable cells, 6 dims for Pinned cells — ``anchor_lambda``
is a no-op when μ_p is frozen so Optuna's TPE sees a flat response
along that axis for Pinned).

Per-axis ranges:

* ``n_pretrain``     — log-uniform integer ``range(5, 500)``. Tightened
                        from the original (50, 2000) per empirical
                        observation that nonlinear-bimodal-lift converges
                        around 250-300 stage-1 steps (so 2000 was wasteful)
                        and that "almost no pretrain" (lower bound 5) is
                        a useful regime to probe.
* ``sigma_pert``     — log-uniform float ``interval(1e-3, 5e-2)`` —
                        tightened from the original ``(1e-4, 1e-1)``;
                        the lower bound is operationally indistinguishable
                        from 0 (encoder weights scale ~1e-2 so 1e-3 is a
                        sub-percent relative perturbation) but Optuna's
                        log-uniform still cannot reach 0 — and per
                        ADR-0002 the protocol forbids σ_pert=0.
* ``anchor_lambda``  — log-uniform float ``interval(1e-4, 1e-1)``.
                        Strength of R_μp; active only under Learnable.
                        Sampled for every cell but a no-op for Pinned
                        (the regulariser term zeros out when μ_p is
                        frozen).
* ``lambda_sigma_p`` — log-uniform float ``interval(1e-3, 1e-1)``.
                        Stage-1 log-variance anchor strength
                        (``model-v2.org`` § State-conditional prior
                        variance calls 1e-2 a "starting suggestion"
                        to be tuned empirically).
* ``base_lr``        — log-uniform float ``interval(1e-5, 1e-3)``.
                        Encoder LR baseline; decoder + transition LRs
                        are derived via the multipliers below.
* ``dec_mult`` /
  ``trans_mult``     — log-uniform float ``interval(0.1, 10.0)``.
                        Per-group LR ratios relative to ``base_lr``.

Run::

    python -m ddssm.app --multirun \\
        experiment=init_<cell> +sweep=init_ablation \\
        hydra.sweeper.n_trials=40 \\
        hydra.sweeper.study_name=ablation_$(date +%s)
"""

from __future__ import annotations

from hydra_zen import make_config

from conf.registry import sweep_store

_INIT_ABLATION_PARAMS = {
    # Centering-handoff knobs.
    "experiment.training.stages.n_pretrain":
        "tag(log, int(interval(5, 500)))",
    "experiment.training.stages.sigma_pert":
        "tag(log, interval(1e-3, 5e-2))",
    # Regulariser strengths. Post-ADR-0004 these live on the per-stage
    # loss objects, so the sweep targets the stage builder
    # (``experiment.training.stages.*``), not ``model``/``hparams``.
    "experiment.training.stages.anchor_lambda":
        "tag(log, interval(1e-4, 1e-1))",
    "experiment.training.stages.lambda_sigma_p":
        "tag(log, interval(1e-3, 1e-1))",
    # Base LR + per-group multipliers (replaces 3 independent LRs).
    "experiment.training.stages.base_lr":
        "tag(log, interval(1e-5, 1e-3))",
    "experiment.training.stages.dec_mult":
        "tag(log, interval(0.1, 10.0))",
    "experiment.training.stages.trans_mult":
        "tag(log, interval(0.1, 10.0))",
    # Per-stage λ-warmup fractions (CONTEXT.md § "lambda_warmup redesign").
    # Cover 5-50% of stage 1 and 2-25% of stage 2 — the lower bound is
    # "barely-any warmup", the upper bound is "warmup covers most of
    # the stage".
    "experiment.training.stages.stage_1_warmup_frac":
        "tag(log, interval(0.05, 0.5))",
    "experiment.training.stages.stage_2_warmup_frac":
        "tag(log, interval(0.02, 0.25))",
}


InitAblation = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(
        sweeper=dict(
            direction="minimize",
            params=_INIT_ABLATION_PARAMS,
        ),
    ),
)
sweep_store(InitAblation, name="init_ablation")


# Multi-objective variant. Same search space as ``InitAblation``, but
# routes through the ``ddssm_optuna_moo`` sweeper preset (NSGA-II
# sampler, ``direction: [minimize, minimize]``). Pair with a cell
# experiment whose ``objective`` is a ``list[ObjectiveSpec]`` matching
# the direction length — :data:`experiments.init_centering.evals.PilotMOObjective`.
InitAblationMOO = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna_moo"}],
    hydra=dict(
        sweeper=dict(
            # ListConfig of strings — NSGA-II uses both directions.
            direction=["minimize", "minimize"],
            params=_INIT_ABLATION_PARAMS,
        ),
    ),
)
sweep_store(InitAblationMOO, name="init_ablation_moo")


# Round-2 narrowed search space — used by EVERY round-2 cell (pinned + learnable).
# Surgically tightened off the CLEAN round-1 axes (ELBO + steps), per the round-1
# param-importance + top-20%-concentration analysis:
#   * base_lr dominates the ELBO response (importance ~0.42) and its bottom
#     decade [1e-5, 5e-5] is consistently dead -> floor raised to 1e-4.
#   * dec_mult bottom [0.1, 0.3] is never in any cell's top-20%; the high
#     ceiling is kept (mlp_per_t prefers ~3-7) -> [0.3, 10].
#   * lambda_sigma_p / sigma_pert / stage_*_warmup upper tails are rarely
#     good -> mild upper trims.
#   * n_pretrain is ~irrelevant (importance ~0.03); drop the dead upper tail.
#   * anchor_lambda is kept WIDE (``interval(1e-4, 1e-1)``). There's no round-1
#     data to narrow it (it was never swept), and it's a no-op flat axis for the
#     pinned cells (μ_p frozen → R_μp zeros out), but the learnable cells tune
#     their R_μp regulariser through it.
_INIT_ABLATION_R2_PARAMS = {
    "experiment.training.stages.n_pretrain":
        "tag(log, int(interval(5, 300)))",
    "experiment.training.stages.sigma_pert":
        "tag(log, interval(1e-3, 3e-2))",
    "experiment.training.stages.anchor_lambda":
        "tag(log, interval(1e-4, 1e-1))",
    "experiment.training.stages.lambda_sigma_p":
        "tag(log, interval(1e-3, 5e-2))",
    "experiment.training.stages.base_lr":
        "tag(log, interval(1e-4, 1e-3))",
    "experiment.training.stages.dec_mult":
        "tag(log, interval(0.3, 10.0))",
    "experiment.training.stages.trans_mult":
        "tag(log, interval(0.1, 10.0))",
    "experiment.training.stages.stage_1_warmup_frac":
        "tag(log, interval(0.05, 0.35))",
    "experiment.training.stages.stage_2_warmup_frac":
        "tag(log, interval(0.02, 0.18))",
}


InitAblationMOO_R2 = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna_moo"}],
    hydra=dict(
        sweeper=dict(
            direction=["minimize", "minimize"],
            params=_INIT_ABLATION_R2_PARAMS,
        ),
    ),
)
sweep_store(InitAblationMOO_R2, name="init_ablation_moo_r2")


# Back-compat alias. The launcher and CLI examples still reference
# ``+sweep=init_pilot`` from the Phase-C era; keep the name working so
# old commands don't break. New code should prefer ``init_ablation``.
sweep_store(InitAblation, name="init_pilot")


__all__ = ["InitAblation", "InitAblationMOO", "InitAblationMOO_R2"]
