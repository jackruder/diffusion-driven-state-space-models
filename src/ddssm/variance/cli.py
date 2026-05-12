"""Hydra entry point for the variance probe stage."""

from __future__ import annotations

import logging
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf

from .._experiment_registry import register_experiments

register_experiments()

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = HydraConfig.get().runtime.output_dir
    orig_cwd = hydra.utils.get_original_cwd()
    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is not None and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(orig_cwd, checkpoint_path)
    experiment = instantiate(cfg.experiment)
    return experiment.variance_probe(
        device=device,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
