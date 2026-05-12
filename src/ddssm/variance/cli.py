"""Hydra entry point for the variance probe stage.

If the requested checkpoint does not exist yet, ``main`` runs the
training stage in-process and then proceeds to the probe — so

    python -m ddssm.variance experiment=variance_probe_lgssm \\
        hydra.run.dir=runs/variance_probe/lgssm

is a one-shot ``train then probe`` command. Pass ``+checkpoint=path``
to point at an existing checkpoint and skip training.
"""

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

    # Default checkpoint location is ``<run_dir>/checkpoints/ckpt_latest.pth``
    # — matches where ``Experiment.train`` writes. Override with
    # ``+checkpoint=path/to/other.pth`` for a one-off file.
    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(run_dir, "checkpoints", "ckpt_latest.pth")
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(orig_cwd, checkpoint_path)

    os.makedirs(run_dir, exist_ok=True)

    experiment = instantiate(cfg.experiment)

    if not os.path.exists(checkpoint_path):
        log.info(
            "Checkpoint not found at %s — running training stage first.",
            checkpoint_path,
        )
        experiment.train(device=device, run_dir=run_dir)
        # ``Experiment.train`` writes ckpt_latest.pth into the run_dir.
        checkpoint_path = os.path.join(run_dir, "checkpoints", "ckpt_latest.pth")
        log.info("Training finished; probing from %s", checkpoint_path)

    return experiment.variance_probe(
        device=device,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
