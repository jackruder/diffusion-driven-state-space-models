"""Hydra entry point for variance-probe runs."""

from __future__ import annotations

import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ddssm import conf  # noqa: F401
from ddssm.workflow import RunMetadata, variance_config

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    log.info("run_dir=%s", run_dir)

    orig_cwd = hydra.utils.get_original_cwd()
    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is not None and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(orig_cwd, checkpoint_path)

    return variance_config(
        cfg,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        metadata=RunMetadata(
            config_identity=(
                f"hydra:experiment={hydra_cfg.runtime.choices.get('experiment', 'unknown')}"
            ),
            overrides=tuple(hydra_cfg.overrides.task),
        ),
    )


if __name__ == "__main__":
    main()
