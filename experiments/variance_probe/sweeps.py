"""Sweep / preset configs for the variance-probe experiments.

The ``variance_probe`` preset is a *config preset*, not an Optuna
search space — it enables W&B logging and tweaks the probe runner's
``R``/``n_batches``/``seeds`` so a probe run is reproducible across
the synthetic modes::

    # train
    python -m ddssm.app \\
        experiment=variance_probe_lgssm +sweep=variance_probe

    # run probe from the checkpoint
    python -m ddssm.variance \\
        experiment=variance_probe_lgssm \\
        checkpoint='${experiment.checkpoint_dir}/ckpt_latest.pth' \\
        +sweep=variance_probe
"""

from __future__ import annotations

from hydra_zen import make_config

from conf.registry import sweep_store


VarianceProbe = make_config(
    hydra_defaults=["_self_", {"override /wandb": "enabled"}],
    experiment=dict(
        wandb_config=dict(
            project="ddssm-variance-probe",
            tags=["variance-probe"],
        ),
        variance=dict(
            R=32,
            n_batches=1,
            seeds=[0],
        ),
    ),
)
sweep_store(VarianceProbe, name="variance_probe")
