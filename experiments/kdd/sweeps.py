"""Optuna sweep presets for the KDD experiments.

Phase-1 mirrors ``scripts/experiments/kdd/optuna-kdd-p1.py`` so the
Hydra-native path produces equivalent search behaviour::

    python -m ddssm.app --multirun \\
        experiment=kdd_gauss \\
        +sweep=kdd_phase1 \\
        hydra.sweeper.n_trials=50 \\
        hydra.sweeper.study_name=ddssm_kdd_recon
"""

from __future__ import annotations

from hydra_zen import make_config

from conf.registry import sweep_store


KDDPhase1 = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(
        sweeper=dict(
            direction="minimize",
            params={
                "experiment.hyperparams.enc_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.dec_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.zinit_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.trans_lr": "interval(1e-5, 3e-4)",
                "experiment.hyperparams.lambda_warmup_steps":
                    "range(50, 4000, step=50)",
                "experiment.hyperparams.lambda_end": "interval(0.7, 2.0)",
                "experiment.hyperparams.weight_decay": "interval(1e-5, 1e-2)",
                "experiment.hyperparams.batch_size": "choice(32, 64, 128)",
            },
        ),
    ),
)
sweep_store(KDDPhase1, name="kdd_phase1")
