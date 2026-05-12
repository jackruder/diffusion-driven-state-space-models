"""Hydra entry point for DDSSM training.

Usage::

    # Default experiment (harmonic + Gaussian transition).
    python -m ddssm.app

    # Pick a different experiment.
    python -m ddssm.app experiment=harmonic_diffusion
    python -m ddssm.app experiment=kdd_gauss

    # Override any field at any depth via dot-notation.
    python -m ddssm.app experiment=harmonic_gauss \\
        experiment.training.steps=200 \\
        experiment.model.transition.hidden_dim=128

    # Optuna sweep using a pre-defined search space.
    python -m ddssm.app --multirun \\
        experiment=synthetic_gauss \\
        +sweep=synthetic_lr \\
        hydra.sweeper.n_trials=20

Experiments are discovered from ``experiments/*.py`` in the repo root;
see :mod:`ddssm._experiment_registry`.
"""

from __future__ import annotations

import logging

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf

from ._experiment_registry import register_experiments

register_experiments()

log = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
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
