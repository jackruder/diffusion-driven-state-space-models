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

from ddssm.stores import sweep_store
from experiments._sweep import SweepSpace
from experiments.init_centering.hparams import StagesB
from experiments.init_centering.evals import PilotMOObjective

# Field names below are validated against ``StagesB`` (the stage-builder
# config) at import time — a typo or renamed factory arg raises here rather
# than crashing every trial after launch. Post-ADR-0004 the regulariser
# strengths + LRs live on the per-stage loss objects, so the sweep targets
# the stage builder (``experiment.training.stages.*``), not ``model``/``hparams``.
_ablation = SweepSpace(target=StagesB, prefix="experiment.training.stages")
_ablation.log_int("n_pretrain", 5, 500)
_ablation.log("sigma_pert", 1e-3, 5e-2)
_ablation.log("anchor_lambda", 1e-4, 1e-1)
_ablation.log("lambda_sigma_p", 1e-3, 1e-1)
_ablation.log("base_lr", 1e-5, 1e-3)        # encoder LR; dec/trans via multipliers
_ablation.log("dec_mult", 0.1, 10.0)
_ablation.log("trans_mult", 0.1, 10.0)
# Per-stage λ-warmup fractions (CONTEXT.md § "lambda_warmup redesign"):
# 5-50% of stage 1, 2-25% of stage 2.
_ablation.log("stage_1_warmup_frac", 0.05, 0.5)
_ablation.log("stage_2_warmup_frac", 0.02, 0.25)


InitAblation = _ablation.build(sweeper="ddssm_optuna", direction="minimize")
sweep_store(InitAblation, name="init_ablation")


# Multi-objective variant — same search space, NSGA-II sweeper, two minimize
# directions. The direction length is checked against PilotMOObjective.specs.
InitAblationMOO = _ablation.build(
    sweeper="ddssm_optuna_moo",
    direction=["minimize", "minimize"],
    objectives=PilotMOObjective,
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
_ablation_r2 = SweepSpace(target=StagesB, prefix="experiment.training.stages")
_ablation_r2.log_int("n_pretrain", 5, 300)
_ablation_r2.log("sigma_pert", 1e-3, 3e-2)
_ablation_r2.log("anchor_lambda", 1e-4, 1e-1)
_ablation_r2.log("lambda_sigma_p", 1e-3, 5e-2)
_ablation_r2.log("base_lr", 1e-4, 1e-3)
_ablation_r2.log("dec_mult", 0.3, 10.0)
_ablation_r2.log("trans_mult", 0.1, 10.0)
_ablation_r2.log("stage_1_warmup_frac", 0.05, 0.35)
_ablation_r2.log("stage_2_warmup_frac", 0.02, 0.18)


InitAblationMOO_R2 = _ablation_r2.build(
    sweeper="ddssm_optuna_moo",
    direction=["minimize", "minimize"],
    objectives=PilotMOObjective,
)
sweep_store(InitAblationMOO_R2, name="init_ablation_moo_r2")


# Back-compat alias. The launcher and CLI examples still reference
# ``+sweep=init_pilot`` from the Phase-C era; keep the name working so
# old commands don't break. New code should prefer ``init_ablation``.
sweep_store(InitAblation, name="init_pilot")


__all__ = ["InitAblation", "InitAblationMOO", "InitAblationMOO_R2"]
