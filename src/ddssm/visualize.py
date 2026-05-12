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
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from . import conf  # noqa: F401  -- registers the ConfigStore
from .workflow import RunMetadata, visualize_config

log = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    log.info("run_dir=%s", run_dir)

    orig_cwd = hydra.utils.get_original_cwd()
    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is not None and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(orig_cwd, checkpoint_path)
    csv_path = cfg.get("csv_path", None)
    if csv_path is not None and not os.path.isabs(csv_path):
        csv_path = os.path.join(orig_cwd, csv_path)

    return visualize_config(
        cfg,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
        metadata=RunMetadata(
            config_identity=(
                f"hydra:experiment={hydra_cfg.runtime.choices.get('experiment', 'unknown')}"
            ),
            overrides=tuple(hydra_cfg.overrides.task),
        ),
    )


if __name__ == "__main__":
    main()
