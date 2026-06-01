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


# Round-2 search space — used by EVERY round-2 cell (pinned + learnable).
# Derived from the FIRST VALID round-1 data (round1v2_20260531, the seed-fixed
# rerun — earlier rounds had the duplicate-seed bug, see
# ddssm.launch._SAMPLER_SEED_OVERRIDE). Univariate signal analysis over 189
# pooled COMPLETE trials (Spearman |rho| of log-param vs obj1, tertile-mean obj1,
# best-25% concentration) found only THREE axes carry signal, all monotone
# "higher is better":
#   * base_lr     |rho|=0.76 (dominant): bottom 1.5 decades [1e-5, 1e-4] dead,
#                 best-25% in [1.5e-4, 8e-4] -> floor raised to 1e-4.
#   * dec_mult    |rho|=0.32: bottom [0.1, 0.3] never in best-25% -> [0.3, 10].
#   * trans_mult  |rho|=0.31: bottom [0.1, 0.5] weak               -> [0.5, 10].
# The other six axes (n_pretrain, sigma_pert, anchor_lambda, lambda_sigma_p,
# stage_1/2_warmup_frac) were flat noise (|rho| < 0.13, best-25% spans the full
# range). Per the round-2 decision we keep them at the FULL round-1 ranges rather
# than fixing them — the analysis is univariate (no fANOVA interaction terms;
# sklearn absent on the cluster), so we only commit to narrowing the three clear
# signals and let the others vary.
_ablation_r2 = SweepSpace(target=StagesB, prefix="experiment.training.stages")
_ablation_r2.log_int("n_pretrain", 5, 500)
_ablation_r2.log("sigma_pert", 1e-3, 5e-2)
_ablation_r2.log("anchor_lambda", 1e-4, 1e-1)
_ablation_r2.log("lambda_sigma_p", 1e-3, 1e-1)
_ablation_r2.log("base_lr", 1e-4, 1e-3)     # narrowed: dead bottom dropped
_ablation_r2.log("dec_mult", 0.3, 10.0)     # narrowed: dead bottom dropped
_ablation_r2.log("trans_mult", 0.5, 10.0)   # narrowed: weak bottom dropped
_ablation_r2.log("stage_1_warmup_frac", 0.05, 0.5)
_ablation_r2.log("stage_2_warmup_frac", 0.02, 0.25)


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
