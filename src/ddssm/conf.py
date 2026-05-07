"""Central hydra-zen configuration: ZenStore + config-only dataclasses.

Top-level ``*Conf`` classes live next to the classes they describe:
  - ``DDSSMHyperParamsConf`` and ``DDSSMConf`` in ``ddssm.dssd``
  - ``DDSSMTrainerConf`` in ``ddssm.train``
  - per-module ``*Conf`` in their respective modules

This file:
  - Re-exports those configs for convenient ``from ddssm.conf import ...`` access.
  - Owns the ``ZenStore`` and registers ``transition``, ``model``, and
    ``trainer`` config groups.
  - Defines the slim Stages dataclasses (``StageSpecConf`` / ``StagesConf``)
    that are config-only (full stage logic lives in ``ddssm.stages``).
  - Provides ``build_model(yaml_path)`` for back-compat YAML loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from hydra_zen import ZenStore
from omegaconf import MISSING

from .dssd import DDSSMConf, DDSSMHyperParamsConf, REWOConf
from .train import DDSSMTrainerConf
from .transitions.transitions import GaussianTransitionConf
from .transitions.diffusion import DiffusionTransitionConf


# ---------------------------------------------------------------------------
# ZenStore with config groups
# ---------------------------------------------------------------------------

store = ZenStore(name="ddssm")

# transition group: gaussian | diffusion
store(GaussianTransitionConf, group="transition", name="gaussian")
store(DiffusionTransitionConf, group="transition", name="diffusion")
store(DDSSMConf, group="model", name="default")
store(DDSSMTrainerConf, group="trainer", name="default")


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
# (Phase 5 will rename this to ``load_yaml_config``.)
# ---------------------------------------------------------------------------

def build_model(yaml_path: str) -> Any:
    """Load a Hydra-compatible YAML and return a DDSSMConf-style config.

    The returned object can be passed to ``hydra_zen.instantiate(cfg)`` to
    construct the ``DDSSM_base`` model.
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
    "StageLrsConf",
    "StageTrainableConf",
    "StageSchedulerConf",
    "LambdaRampConf",
    "StageSpecConf",
    "StagesConf",
    "build_model",
    "store",
]
