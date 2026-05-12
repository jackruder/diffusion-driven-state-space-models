"""Run DDSSM experiments from Python-authored Hydra-Zen configs."""

from __future__ import annotations

import os
import json
from typing import Mapping, Sequence
import logging
from pathlib import Path
from dataclasses import field, dataclass

import torch
from hydra_zen import instantiate
from omegaconf import OmegaConf, DictConfig

from .conf import (
    CSDIUnetGroupConf,
    TimeMixerConvConf,
    DecoderGaussianConf,
    EncoderGaussianConf,
    InitPriorGaussianConf,
    TransitionGaussianConf,
    ContextProducerCSDIConf,
    FeatureMixerTransformerConf,
)

log = logging.getLogger(__name__)

type ConfigInput = object
type StageResult = object


@dataclass(frozen=True)
class ConfigGroups:
    """Default config-group choices for Python source-block experiment runs."""

    transition: ConfigInput = TransitionGaussianConf
    encoder: ConfigInput = EncoderGaussianConf
    decoder: ConfigInput = DecoderGaussianConf
    z_init: ConfigInput = InitPriorGaussianConf
    context: ConfigInput = ContextProducerCSDIConf
    unet: ConfigInput = CSDIUnetGroupConf
    time_mixer: ConfigInput = TimeMixerConvConf
    feature_mixer: ConfigInput = FeatureMixerTransformerConf


DEFAULT_CONFIG_GROUPS = ConfigGroups()


@dataclass(frozen=True)
class RunMetadata:
    """Identity and override metadata persisted with each stage run."""

    config_identity: str = "python-config"
    overrides: Sequence[str] = field(default_factory=tuple)


def compose_experiment_config(
    experiment_config: ConfigInput,
    *,
    groups: ConfigGroups | None = None,
    updates: Mapping[str, ConfigInput] | None = None,
) -> DictConfig:
    """Build a root config from a Python-defined experiment config object.

    The returned config mirrors the Hydra CLI root shape, including the config
    groups needed to resolve presets that contain ``${transition}``,
    ``${encoder}``, and similar interpolations.
    """
    group_choices = groups or DEFAULT_CONFIG_GROUPS
    cfg = OmegaConf.create({
        "experiment": OmegaConf.structured(experiment_config),
        "transition": OmegaConf.structured(group_choices.transition),
        "encoder": OmegaConf.structured(group_choices.encoder),
        "decoder": OmegaConf.structured(group_choices.decoder),
        "z_init": OmegaConf.structured(group_choices.z_init),
        "context": OmegaConf.structured(group_choices.context),
        "unet": OmegaConf.structured(group_choices.unet),
        "time_mixer": OmegaConf.structured(group_choices.time_mixer),
        "feature_mixer": OmegaConf.structured(group_choices.feature_mixer),
    })
    if updates:
        cfg = OmegaConf.merge(cfg, updates)
    return cfg


def train_config(
    config: ConfigInput,
    *,
    run_dir: str | os.PathLike[str],
    device: torch.device | str | None = None,
    groups: ConfigGroups | None = None,
    updates: Mapping[str, ConfigInput] | None = None,
    metadata: RunMetadata | None = None,
) -> float | StageResult:
    """Train an experiment from a Python config or a composed root config."""
    cfg = _as_root_config(config, groups=groups, updates=updates)
    run_path = Path(run_dir)
    torch_device = _resolve_device(device)
    resolved_path = persist_resolved_config(cfg, run_path)
    experiment = instantiate(cfg.experiment)
    result = experiment.train(device=torch_device, run_dir=str(run_path))
    write_experiment_log(
        stage="train",
        cfg=cfg,
        run_dir=run_path,
        resolved_config_path=resolved_path,
        result=result,
        metadata=metadata,
    )
    return result


def evaluate_config(
    config: ConfigInput,
    *,
    run_dir: str | os.PathLike[str],
    checkpoint_path: str | None = None,
    csv_path: str | None = None,
    device: torch.device | str | None = None,
    groups: ConfigGroups | None = None,
    updates: Mapping[str, ConfigInput] | None = None,
    metadata: RunMetadata | None = None,
) -> dict:
    """Evaluate an experiment from a Python config or a composed root config."""
    cfg = _as_root_config(config, groups=groups, updates=updates)
    run_path = Path(run_dir)
    torch_device = _resolve_device(device)
    resolved_path = persist_resolved_config(cfg, run_path)
    experiment = instantiate(cfg.experiment)
    result = experiment.evaluate(
        device=torch_device,
        run_dir=str(run_path),
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
    )
    write_experiment_log(
        stage="evaluate",
        cfg=cfg,
        run_dir=run_path,
        resolved_config_path=resolved_path,
        result=result,
        metadata=metadata,
    )
    return result


def visualize_config(
    config: ConfigInput,
    *,
    run_dir: str | os.PathLike[str],
    checkpoint_path: str | None = None,
    csv_path: str | None = None,
    device: torch.device | str | None = None,
    groups: ConfigGroups | None = None,
    updates: Mapping[str, ConfigInput] | None = None,
    metadata: RunMetadata | None = None,
) -> list[str]:
    """Visualize an experiment from a Python config or a composed root config."""
    cfg = _as_root_config(config, groups=groups, updates=updates)
    run_path = Path(run_dir)
    torch_device = _resolve_device(device)
    resolved_path = persist_resolved_config(cfg, run_path)
    experiment = instantiate(cfg.experiment)
    result = experiment.visualize(
        device=torch_device,
        run_dir=str(run_path),
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
    )
    write_experiment_log(
        stage="visualize",
        cfg=cfg,
        run_dir=run_path,
        resolved_config_path=resolved_path,
        result=result,
        metadata=metadata,
    )
    return result


def variance_config(
    config: ConfigInput,
    *,
    run_dir: str | os.PathLike[str],
    checkpoint_path: str | None = None,
    device: torch.device | str | None = None,
    groups: ConfigGroups | None = None,
    updates: Mapping[str, ConfigInput] | None = None,
    metadata: RunMetadata | None = None,
) -> dict[str, StageResult]:
    """Run a variance probe from a Python config or a composed root config."""
    cfg = _as_root_config(config, groups=groups, updates=updates)
    run_path = Path(run_dir)
    torch_device = _resolve_device(device)
    resolved_path = persist_resolved_config(cfg, run_path)
    experiment = instantiate(cfg.experiment)
    result = experiment.variance_probe(
        device=torch_device,
        run_dir=str(run_path),
        checkpoint_path=checkpoint_path,
    )
    write_experiment_log(
        stage="variance",
        cfg=cfg,
        run_dir=run_path,
        resolved_config_path=resolved_path,
        result=result,
        metadata=metadata,
    )
    return result


def persist_resolved_config(cfg: DictConfig, run_dir: Path) -> Path:
    """Write the resolved config artifact used by every workflow stage."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "resolved_config.yaml"
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return path


def write_experiment_log(
    *,
    stage: str,
    cfg: DictConfig,
    run_dir: Path,
    resolved_config_path: Path,
    result: StageResult,
    metadata: RunMetadata | None = None,
) -> Path:
    """Persist a lightweight per-stage experiment log."""
    info = metadata or RunMetadata()
    path = run_dir / "experiment_log.json"
    payload = {
        "stage": stage,
        "config_identity": info.config_identity,
        "overrides": list(info.overrides),
        "resolved_config": str(resolved_config_path),
        "run_dir": str(run_dir),
        "key_metrics": _key_metrics(result),
        "experiment_target": OmegaConf.select(cfg, "experiment._target_"),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("Wrote experiment log to %s", path)
    return path


def _as_root_config(
    config: ConfigInput,
    *,
    groups: ConfigGroups | None,
    updates: Mapping[str, ConfigInput] | None,
) -> DictConfig:
    if isinstance(config, DictConfig) and "experiment" in config:
        cfg = OmegaConf.create(config)
        return OmegaConf.merge(cfg, updates) if updates else cfg
    return compose_experiment_config(config, groups=groups, updates=updates)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def _key_metrics(result: StageResult) -> dict[str, StageResult]:
    if isinstance(result, float):
        return {"objective": result}
    if isinstance(result, Mapping):
        return {str(key): value for key, value in result.items()}
    if isinstance(result, list):
        return {"outputs": result, "count": len(result)}
    return {}


__all__ = [
    "DEFAULT_CONFIG_GROUPS",
    "ConfigGroups",
    "RunMetadata",
    "compose_experiment_config",
    "evaluate_config",
    "persist_resolved_config",
    "train_config",
    "variance_config",
    "visualize_config",
    "write_experiment_log",
]
