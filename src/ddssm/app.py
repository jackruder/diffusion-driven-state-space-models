"""Hydra entry point for DDSSM training.

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
import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf

# Importing conf populates Hydra's ConfigStore via ``store.add_to_hydra_store``.
# Must precede @hydra.main so config groups resolve.
from . import conf  # noqa: F401

log = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = HydraConfig.get().runtime.output_dir
    log.info("Device=%s run_dir=%s", device, run_dir)

    try:
        with open(f"{run_dir}/resolved_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
    except OSError as e:
        log.warning("Could not persist resolved_config.yaml: %s", e)

    experiment = instantiate(cfg.experiment)
    return experiment.train(device=device, run_dir=run_dir)


if __name__ == "__main__":
    main()
