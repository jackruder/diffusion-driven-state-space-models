"""Compose-and-instantiate tests for Hydra experiment + sweep config groups.

These tests exercise the new ``conf/experiment/`` and ``conf/sweep/`` config
groups added alongside the Hydra-native training / Optuna flow. They verify
that every preset:

* composes with the rest of the config tree, and
* (for experiments) yields a model that ``hydra_zen.instantiate`` can build
  without raising.

The tests do not run training or invoke Optuna; they only check config
plumbing, which keeps them fast and CPU-friendly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate

import ddssm.conf  # noqa: F401  -- registers ConfigStore entries

# Resolve the repo's ``conf/`` directory once.
CONF_DIR = (Path(__file__).resolve().parent.parent / "conf").as_posix()

EXPERIMENTS = [
    "synthetic_gauss",
    "synthetic_diffusion",
    "kdd_gauss",
    "kdd_diffusion",
]

SWEEPS = [
    "synthetic_lr",
    "kdd_phase1",
]


@pytest.fixture(autouse=True)
def _clear_global_hydra():
    """Reset Hydra's singleton state between tests so initialise_config_dir is reusable."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_preset_composes(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"+experiment={name}"])
    # Every experiment must select a transition module.
    assert cfg.transition._target_, f"experiment {name} missing transition._target_"
    # Each experiment sets either a real dataset target or the explicit ``none`` preset.
    assert "dataset" in cfg, f"experiment {name} missing dataset entry"
    # Training scalars must be present so ``ddssm.app`` can drive ``trainer.fit``.
    assert int(cfg.training.steps) > 0


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_model_instantiates(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"+experiment={name}"])
    model = instantiate(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, f"experiment {name} produced an empty model"


@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_preset_composes(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"+sweep={name}"],
            return_hydra_config=True,
        )
    sweeper = cfg.hydra.sweeper
    # Sweep presets must activate the Optuna sweeper so ``--multirun`` works.
    assert "optuna" in sweeper._target_.lower(), (
        f"sweep {name} did not select the optuna sweeper, got {sweeper._target_}"
    )
    assert sweeper.direction == "minimize", (
        f"sweep {name} should minimise the objective (loss)"
    )
    assert len(sweeper.params) > 0, f"sweep {name} declared no search-space params"


def test_experiment_and_sweep_combine() -> None:
    """The two overlay groups must be composable in a single multirun command."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=synthetic_gauss", "+sweep=synthetic_lr"],
            return_hydra_config=True,
        )
    assert cfg.dataset._target_.endswith("SyntheticDataset")
    assert "optuna" in cfg.hydra.sweeper._target_.lower()
    assert "hyperparams.enc_lr" in cfg.hydra.sweeper.params


def test_dataset_none_preset_disables_autotrain_signal() -> None:
    """The ``none`` dataset preset is the signal ``ddssm.app`` uses to skip ``fit``."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["dataset=none"])
    # ``none`` must surface a null ``_target_`` so ``_is_dataset_disabled`` returns True.
    assert cfg.dataset.get("_target_") in (None, "null", "")


def test_training_defaults_present_on_base_config() -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config")
    assert cfg.training.steps > 0
    assert cfg.training.log_every > 0
    assert "return_objective" in cfg.training
