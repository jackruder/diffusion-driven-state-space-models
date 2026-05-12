"""Tests for Python-authored experiment config workflow helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ddssm.conf import TransitionDiffusionConf
from ddssm.workflow import (
    RunMetadata,
    ConfigGroups,
    write_experiment_log,
    compose_experiment_config,
)
from ddssm.conf.experiments.synthetic import HarmonicExperimentConf


def test_python_source_block_config_composes() -> None:
    """Python config objects compose with explicit group defaults."""
    cfg = compose_experiment_config(
        HarmonicExperimentConf,
        groups=ConfigGroups(transition=TransitionDiffusionConf),
        updates={"experiment": {"training": {"steps": 7}}},
    )

    assert cfg.experiment.training.steps == 7
    assert cfg.experiment.transition._target_.endswith("DiffusionTransition")
    assert cfg.experiment.model.transition._target_.endswith("DiffusionTransition")


def test_experiment_log_records_workflow_metadata(tmp_path: Path) -> None:
    """Experiment logs capture source-block workflow metadata and metrics."""
    cfg = compose_experiment_config(HarmonicExperimentConf)
    resolved_path = tmp_path / "resolved_config.yaml"
    resolved_path.write_text("experiment: harmonic\n")

    log_path = write_experiment_log(
        stage="train",
        cfg=cfg,
        run_dir=tmp_path,
        resolved_config_path=resolved_path,
        result=1.25,
        metadata=RunMetadata(
            config_identity="python:ddssm.conf.experiments.synthetic.HarmonicExperimentConf",
            overrides=("experiment.training.steps=7",),
        ),
    )

    payload = OmegaConf.create(log_path.read_text())
    assert payload.stage == "train"
    assert payload.config_identity.endswith("HarmonicExperimentConf")
    assert payload.overrides == ["experiment.training.steps=7"]
    assert payload.key_metrics.objective == pytest.approx(1.25)
