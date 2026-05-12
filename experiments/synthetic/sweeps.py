"""Optuna sweep presets for the synthetic-data experiments.

Each preset registers under ``group="sweep"`` and is activated via
``+sweep=<name>`` on the CLI. ``package="_global_"`` (set on
``sweep_store`` in :mod:`conf.registry`) merges the preset at root,
matching the legacy ``# @package _global_`` YAML semantic.

Example::

    python -m ddssm.app --multirun \\
        experiment=synthetic_gauss \\
        +sweep=synthetic_lr \\
        hydra.sweeper.n_trials=20 \\
        hydra.sweeper.study_name=ddssm_synth_lr
"""

from __future__ import annotations

from hydra_zen import make_config

from conf.registry import sweep_store


SyntheticLR = make_config(
    hydra_defaults=["_self_", {"override /hydra/sweeper": "ddssm_optuna"}],
    hydra=dict(
        sweeper=dict(
            direction="minimize",
            params={
                "experiment.hyperparams.enc_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.dec_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.trans_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.zinit_lr": "interval(1e-5, 1e-3)",
                "experiment.hyperparams.lambda_warmup_steps":
                    "range(50, 400, step=50)",
                "experiment.hyperparams.lambda_end": "interval(0.5, 2.0)",
                "experiment.hyperparams.batch_size": "choice(16, 32, 64)",
            },
        ),
    ),
)
sweep_store(SyntheticLR, name="synthetic_lr")
