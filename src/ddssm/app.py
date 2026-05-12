r"""Hydra entry point for DDSSM training.

Usage::

    # Smoke run with the default Synthetic + Gaussian experiment.
    python -m ddssm.app

    # Pick a different registered experiment.
    python -m ddssm.app experiment=synthetic_diffusion
    python -m ddssm.app experiment=kdd_gauss

    # Override anything inside the experiment subtree.
    python -m ddssm.app experiment=synthetic_gauss \\
        experiment.training.steps=200 \\
        experiment.hyperparams.batch_size=64

    # Optuna sweep using a pre-defined search space.
    python -m ddssm.app --multirun \\
        experiment=synthetic_gauss \\
        +sweep=synthetic_lr \\
        hydra.sweeper.n_trials=20

The instantiated :class:`Experiment` object owns the run; this module
just resolves the run directory, picks the device, persists the
resolved config for reproducibility, and forwards the result.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

# Importing conf populates Hydra's ConfigStore via ``store.add_to_hydra_store``.
# Must precede @hydra.main so config groups resolve.
from . import conf  # noqa: F401
from .workflow import RunMetadata, train_config

log = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> object:
    """Run training from a Hydra-composed experiment config."""
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    choices = hydra_cfg.runtime.choices
    log.info("run_dir=%s", run_dir)
    return train_config(
        cfg,
        run_dir=run_dir,
        metadata=RunMetadata(
            config_identity=f"hydra:experiment={choices.get('experiment', 'unknown')}",
            overrides=tuple(hydra_cfg.overrides.task),
        ),
    )


if __name__ == "__main__":
    main()
