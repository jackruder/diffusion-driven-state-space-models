"""Hydra-based CLI entry point for training DDSSM models.

Usage::

    # Train with default Gaussian transition
    python -m ddssm.app

    # Train with Diffusion transition
    python -m ddssm.app transition=diffusion

    # Override individual params from the CLI
    python -m ddssm.app data_dim=2 latent_dim=8 hyperparams.batch_size=32

    # Use an experiment file that overrides the base config
    python -m ddssm.app +experiment=kdd_gauss_beijing
"""

import logging

import hydra
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf

import torch

log = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Build the DDSSM model from config and start training."""
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device: %s", device)

    # Build each sub-module individually, then build the model
    encoder = instantiate(cfg.encoder)
    decoder = instantiate(cfg.decoder)
    z_init = instantiate(cfg.z_init)
    transition = instantiate(cfg.transition)

    from .dssd import DDSSM_base
    from .conf import DDSSMHyperParamsConf
    from types import SimpleNamespace

    # Hyperparams from config
    hp = cfg.hyperparams
    hyperparams = SimpleNamespace(**OmegaConf.to_container(hp, resolve=True))

    model = DDSSM_base(
        encoder=encoder,
        decoder=decoder,
        z_init=z_init,
        transition=transition,
        j=cfg.j,
        data_dim=cfg.data_dim,
        latent_dim=cfg.latent_dim,
        emb_time_dim=cfg.emb_time_dim,
        covariate_dim=cfg.get("covariate_dim", 0),
        use_observation_mask=cfg.get("use_observation_mask", True),
        hyperparams=hyperparams,
        checkpoint_dir=cfg.get("checkpoint_dir", "./checkpoints"),
    ).to(device)

    log.info("Model built: %d parameters", sum(p.numel() for p in model.parameters()))

    from .train import DDSSMTrainer

    trainer = DDSSMTrainer(model, device)
    log.info("Trainer ready. Run trainer.train(...) to start training.")
    return trainer


if __name__ == "__main__":
    main()
