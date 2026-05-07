"""Hydra entry point for the visualization stage.

Loads a trained checkpoint into the experiment's model, walks the
plots declared on ``cfg.experiment.viz``, and saves PNGs to the Hydra
run dir.

Usage::

    # Default plots for the experiment, rendered on the test split
    python -m ddssm.visualize experiment=kdd_gauss \\
        +checkpoint=outputs/.../ckpt_latest.pth

    # Pass a CSV path so the metrics_csv plot has data to draw
    python -m ddssm.visualize experiment=synthetic_gauss \\
        +checkpoint=path/to/ckpt.pth \\
        +csv_path=outputs/.../metrics.csv

    # Override the plot list at the CLI
    python -m ddssm.visualize experiment=kdd_gauss \\
        +checkpoint=path/to/ckpt.pth \\
        'experiment.viz.plots=[{name: forecast_1d, save_filename: f.png}]'
"""

from __future__ import annotations

import logging

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf

from . import conf  # noqa: F401  -- registers the ConfigStore

log = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = HydraConfig.get().runtime.output_dir
    log.info("Device=%s run_dir=%s", device, run_dir)

    checkpoint_path = cfg.get("checkpoint", None)
    csv_path = cfg.get("csv_path", None)

    experiment = instantiate(cfg.experiment)
    return experiment.visualize(
        device=device,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()
