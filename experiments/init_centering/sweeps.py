"""Optuna sweep presets for the init-centering family.

Single-phase training leaves only the LR knobs to sweep:

* ``enc_lr`` — log-uniform float ``interval(1e-4, 1e-3)``.
* ``dec_lr`` — log-uniform float ``interval(5e-5, 5e-3)``.
* ``trans_lr`` — log-uniform float ``interval(5e-5, 5e-3)``.

Run::

    python -m ddssm.app --multirun \\
        experiment=init_smoke_high_surface +sweep=init_ablation \\
        hydra.sweeper.n_trials=40 \\
        hydra.sweeper.study_name=ablation_$(date +%s)
"""

from __future__ import annotations

from experiments._sweep import SweepSpace
from ddssm.experiment.stores import sweep_store
from ddssm.experiment.builders import Hparams
from experiments.init_centering.evals import PilotMOObjective


_ablation = SweepSpace(target=Hparams, prefix="experiment.hparams")
_ablation.log("enc_lr", 1e-4, 1e-3)
_ablation.log("dec_lr", 5e-5, 5e-3)
_ablation.log("trans_lr", 5e-5, 5e-3)


InitAblation = _ablation.build(sweeper="ddssm_optuna", direction="minimize")
sweep_store(InitAblation, name="init_ablation")


InitAblationMOO = _ablation.build(
    sweeper="ddssm_optuna_moo",
    direction=["minimize", "minimize"],
    objectives=PilotMOObjective,
)
sweep_store(InitAblationMOO, name="init_ablation_moo")

# Back-compat alias for study.py referencing ``init_ablation_moo_r2``.
sweep_store(InitAblationMOO, name="init_ablation_moo_r2")

# Back-compat alias. The launcher and CLI examples still reference
# ``+sweep=init_pilot`` from the Phase-C era; keep the name working so
# old commands don't break.
sweep_store(InitAblation, name="init_pilot")


__all__ = ["InitAblation", "InitAblationMOO"]
