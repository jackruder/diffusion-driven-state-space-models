"""Compose-and-instantiate tests for the Hydra ``experiment`` config group.

Each registered experiment must:

* Compose at config time (``hydra.compose``) without raising.
* Yield an :class:`Experiment` instance whose ``data``, ``model``,
  ``training``, and ``objective`` fields are populated.
* Produce a non-empty model parameter count.

These tests don't run training -- they only check config plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate

import ddssm.conf  # noqa: F401  -- registers ConfigStore entries
from ddssm.data.datamodule import DDSSMDataModule
from ddssm.experiment import Experiment, ObjectiveSpec, TrainingScalars

CONF_DIR = (Path(__file__).resolve().parent.parent / "conf").as_posix()

EXPERIMENTS = [
    "synthetic_gauss",
    "synthetic_diffusion",
    "kdd_gauss",
    "kdd_diffusion",
]

SWEEPS = ["synthetic_lr", "kdd_phase1"]


@pytest.fixture(autouse=True)
def _clear_global_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_preset_composes(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.experiment.training.steps > 0
    assert cfg.experiment.model._target_.endswith("DDSSM_base")
    assert cfg.experiment.data._target_  # data module target is set


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_instantiates(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    expt = instantiate(cfg.experiment)
    assert isinstance(expt, Experiment)
    assert isinstance(expt.data, DDSSMDataModule)
    assert isinstance(expt.training, TrainingScalars)
    assert isinstance(expt.objective, ObjectiveSpec)
    n_params = sum(p.numel() for p in expt.model.parameters())
    assert n_params > 0


def test_default_experiment_is_synthetic_gauss() -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config")
    assert cfg.experiment.data._target_.endswith("SyntheticDataModule")
    assert cfg.experiment.model.transition._target_.endswith("GaussianTransition")


def test_objective_returns_inf_on_missing_csv(tmp_path) -> None:
    obj = ObjectiveSpec(metric="loss/total", split="train", tail_frac=0.1)
    assert obj.read(str(tmp_path / "missing.csv")) == float("inf")


@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_preset_composes(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"+sweep={name}"],
            return_hydra_config=True,
        )
    sweeper = cfg.hydra.sweeper
    assert "optuna" in sweeper._target_.lower()
    assert sweeper.direction == "minimize"
    assert len(sweeper.params) > 0


def test_experiment_and_sweep_combine() -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["experiment=synthetic_gauss", "+sweep=synthetic_lr"],
            return_hydra_config=True,
        )
    assert cfg.experiment.data._target_.endswith("SyntheticDataModule")
    assert "optuna" in cfg.hydra.sweeper._target_.lower()
