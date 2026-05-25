"""Optuna sweep presets for the init-centering family.

Phase-C pilot: 20 trials on the canonical cell (MLP / Pinned / per-t
EMA), sampling the two handoff knobs from
``init-experiment.org`` § Hyperparameters:

* ``N_pretrain`` — log-uniform integer ``range(50, 2000)``.
* ``σ_pert``    — log-uniform float ``interval(1e-4, 1e-1)``.

Plus the four per-component learning rates as a log-uniform
``interval(1e-5, 1e-3)`` (matching :data:`SyntheticLR`'s ranges).

Known gap: the doc recommends including ``0`` as a control for both
``N_pretrain`` and ``σ_pert``.  Optuna's log-uniform cannot sample
zero, so the pilot studies only the log-uniform region.  The two
control runs become explicit cells of the Phase-D 18-cell grid.

Run::

    python -m ddssm.app --multirun \\
        experiment=init_centering_pilot +sweep=init_pilot \\
        hydra.sweeper.n_trials=20 \\
        hydra.sweeper.study_name=pilot_$(date +%s)
"""

from __future__ import annotations

from hydra_zen import make_config

from conf.registry import sweep_store


InitPilot = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(
        sweeper=dict(
            direction="minimize",
            params={
                # Centering-handoff knobs (the doc's two sweep axes).
                "experiment.model.stages.n_pretrain":
                    "tag(log, int(interval(50, 2000)))",
                "experiment.model.stages.sigma_pert":
                    "tag(log, interval(1e-4, 1e-1))",
                # Per-module learning rates (informational sweep; the
                # canonical cell uses 5e-4 across the board by default).
                "experiment.model.stages.enc_lr":
                    "tag(log, interval(1e-5, 1e-3))",
                "experiment.model.stages.dec_lr":
                    "tag(log, interval(1e-5, 1e-3))",
                "experiment.model.stages.trans_lr":
                    "tag(log, interval(1e-5, 1e-3))",
            },
        ),
    ),
)
sweep_store(InitPilot, name="init_pilot")


__all__ = ["InitPilot"]
