"""Central hydra-zen configuration: ZenStore + config-only dataclasses.

Top-level ``*Conf`` classes live next to the classes they describe:
  - ``DDSSMHyperParamsConf`` and ``DDSSMConf`` in ``ddssm.dssd``
  - ``DDSSMTrainerConf`` in ``ddssm.train``
  - per-module ``*Conf`` in their respective modules

This module:
  - Re-exports those configs for ``from ddssm.conf import ...`` access.
  - Defines store-registered ``transition`` Confs with ``${...}`` interpolations
    on shape kwargs (``latent_dim``, ``j``, ``emb_time_dim``, ``covariate_dim``)
    so a defaults-list selection like ``- transition: gaussian`` produces a
    fully-wired structured config without needing per-field YAML.
  - Owns the ``ZenStore`` and registers the ``transition``, ``model``,
    ``trainer`` config groups, then materialises them into Hydra's ConfigStore
    so ``@hydra.main`` can resolve them.
  - Defines the slim Stages dataclasses (``StageSpecConf`` / ``StagesConf``)
    that are config-only (full stage logic lives in ``ddssm.stages``).
  - Provides ``load_yaml_config(yaml_path)`` for back-compat YAML loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from hydra_zen import builds, ZenStore
from omegaconf import MISSING

from .dssd import DDSSMConf, DDSSMHyperParamsConf, REWOConf
from .train import DDSSMTrainerConf
from .diffnets import ContextProducerConf, CSDIUnetConf
from .gaussians import GaussianHeadConf
from .transitions.transitions import GaussianTransition
from .transitions.diffusion import DiffusionTransition, DiffusionScheduleConfig


# ---------------------------------------------------------------------------
# Top-level transition Confs for the ``transition`` config group.
#
# Shape kwargs interpolate from root cfg keys; sub-module Confs stay nested
# (their own zen_partial defaults handle inner shape wiring at construction).
# ---------------------------------------------------------------------------

TransitionGaussianConf = builds(
    GaussianTransition,
    populate_full_signature=True,
    latent_dim="${latent_dim}",
    j="${j}",
    emb_time_dim="${emb_time_dim}",
    covariate_dim="${covariate_dim}",
    context=ContextProducerConf(),
    gaussian_head=GaussianHeadConf(),
)

TransitionDiffusionConf = builds(
    DiffusionTransition,
    populate_full_signature=True,
    latent_dim="${latent_dim}",
    j="${j}",
    emb_time_dim="${emb_time_dim}",
    covariate_dim="${covariate_dim}",
    unet=CSDIUnetConf(),
    schedule=DiffusionScheduleConfig(),
)


# ---------------------------------------------------------------------------
# ZenStore with config groups
# ---------------------------------------------------------------------------

store = ZenStore(name="ddssm")

store(TransitionGaussianConf, group="transition", name="gaussian")
store(TransitionDiffusionConf, group="transition", name="diffusion")
store(DDSSMConf, group="model", name="default")
store(DDSSMTrainerConf, group="trainer", name="default")

# Materialise the store into Hydra's ConfigStore so @hydra.main can resolve it.
# Importing this module is sufficient to activate the registrations.
store.add_to_hydra_store(overwrite_ok=True)


# ---------------------------------------------------------------------------
# Stages dataclasses (slim versions; full logic lives in stages.py)
# ---------------------------------------------------------------------------

@dataclass
class StageLrsConf:
    dec_lr: float = 5e-4
    zinit_lr: float = 5e-4
    trans_lr: float = 0.0


@dataclass
class StageTrainableConf:
    decoder: bool = True
    z_init: bool = True
    transition: bool = False


@dataclass
class StageSchedulerConf:
    warmup_steps: int = 0
    final_lr_scale: float = 1.0


@dataclass
class LambdaRampConf:
    end: float | None = 1.0
    delay: int = 0
    steps: int | None = None


@dataclass
class StageSpecConf:
    steps: int = MISSING
    trainable: StageTrainableConf = field(default_factory=StageTrainableConf)
    lrs: StageLrsConf = field(default_factory=StageLrsConf)
    scheduler: StageSchedulerConf = field(default_factory=StageSchedulerConf)
    carry_diff_moments: bool = False
    lambda_ramp: LambdaRampConf = field(default_factory=LambdaRampConf)
    log_every: int = 10
    val_every: int = 100
    checkpoint_every: int = 1000


@dataclass
class StagesConf:
    stage_2: StageSpecConf | None = None
    stage_3: StageSpecConf | None = None
    run: List[str] = field(default_factory=lambda: ["stage_1", "stage_2", "stage_3"])


# ---------------------------------------------------------------------------
# Convenience helper: load a Hydra YAML and return an instantiable config.
# ---------------------------------------------------------------------------

def load_yaml_config(yaml_path: str) -> Any:
    """Load a Hydra-compatible YAML and return an OmegaConf DictConfig.

    The returned object can be passed to ``hydra_zen.instantiate(cfg.model)``
    (or any sub-key) to construct the corresponding object.
    """
    from omegaconf import OmegaConf

    with open(yaml_path, "r") as f:
        cfg = OmegaConf.load(f)
    return cfg


__all__ = [
    "DDSSMConf",
    "DDSSMHyperParamsConf",
    "DDSSMTrainerConf",
    "REWOConf",
    "TransitionGaussianConf",
    "TransitionDiffusionConf",
    "StageLrsConf",
    "StageTrainableConf",
    "StageSchedulerConf",
    "LambdaRampConf",
    "StageSpecConf",
    "StagesConf",
    "load_yaml_config",
    "store",
]
