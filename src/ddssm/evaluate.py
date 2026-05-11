"""Hydra entry point for the evaluation stage.

Loads a trained checkpoint into the experiment's model, walks the
metrics declared on ``cfg.experiment.eval``, and writes a single
``metrics.json`` to the Hydra run dir.

Usage::

    # Evaluate the most recent KDD run
    python -m ddssm.evaluate experiment=kdd_gauss \\
        +checkpoint=outputs/.../ckpt_latest.pth

    # Override which metrics to compute and on which split
    python -m ddssm.evaluate experiment=kdd_gauss \\
        +checkpoint=path/to/ckpt.pth \\
        experiment.eval.metrics='[mae, crps_sum, recon_mse]' \\
        experiment.eval.split=test

    # CSV-only metric (no checkpoint needed)
    python -m ddssm.evaluate experiment=synthetic_gauss \\
        +csv_path=outputs/.../metrics.csv \\
        experiment.eval.metrics='[loss_tail]'
"""

from __future__ import annotations

import os
import logging

import hydra
import torch
from hydra_zen import instantiate
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig

from . import conf  # noqa: F401  -- registers the ConfigStore

log = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = HydraConfig.get().runtime.output_dir
    log.info("Device=%s run_dir=%s", device, run_dir)

    orig_cwd = hydra.utils.get_original_cwd()
    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is not None and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(orig_cwd, checkpoint_path)
    csv_path = cfg.get("csv_path", None)
    if csv_path is not None and not os.path.isabs(csv_path):
        csv_path = os.path.join(orig_cwd, csv_path)

    experiment = instantiate(cfg.experiment)
    return experiment.evaluate(
        device=device,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()
