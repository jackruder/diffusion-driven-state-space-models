"""Compose-and-instantiate tests for the named experiments.

Each preset registered to ``experiment_store`` (group=``experiment``)
must:

* Compose to an :class:`Experiment` instance via ``instantiate(node)``,
  with populated ``data``, ``model``, ``training``, and a non-empty
  model parameter count.
* Resolve through the Hydra CLI bridge
  (``ddssm._experiment_registry.register_experiments`` →
  ``compose(config_name='config', overrides=[experiment=NAME])``).
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate, store
import pytest

from ddssm._experiment_registry import register_experiments
from ddssm.data.datamodule import DDSSMDataModule
from ddssm.experiment import Experiment, ObjectiveSpec, TrainingScalars

CONF_DIR = (Path(__file__).resolve().parent.parent / "src" / "ddssm" / "conf").as_posix()


def _registered_experiment_names() -> list[str]:
    """Pull the live experiment-group entries out of the zen-store singleton."""
    register_experiments()
    group = store["experiment"]   # tuple-keyed dict ((group, name) -> node)
    return sorted(name for _, name in group)


def _exp(name: str):
    """Look up the registered experiment Conf node by name."""
    return store["experiment"][("experiment", name)]


# Populated once at collection time so ``parametrize`` sees the same list
# the runtime registry exposes — no hardcoded names.
EXPERIMENTS = _registered_experiment_names()

SWEEPS = ["synthetic_lr", "kdd_phase1"]


@pytest.fixture(autouse=True)
def _clear_global_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    register_experiments()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def test_experiments_registered() -> None:
    """All 14 named presets are reachable through the experiment store."""
    assert len(EXPERIMENTS) == 14, EXPERIMENTS


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_instantiates(name: str) -> None:
    expt = instantiate(_exp(name))
    assert isinstance(expt, Experiment)
    assert isinstance(expt.data, DDSSMDataModule)
    assert isinstance(expt.training, TrainingScalars)
    n_params = sum(p.numel() for p in expt.model.parameters())
    assert n_params > 0


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_cli_compose(name: str) -> None:
    """``python -m ddssm.app experiment=NAME`` resolves to the same exp."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.experiment.training.steps > 0
    assert cfg.experiment.model._target_.endswith("DDSSM_base")
    assert cfg.experiment.data._target_


def test_default_experiment_is_harmonic_gauss() -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config")
    assert cfg.experiment.data._target_.endswith("SyntheticDataModule")
    assert cfg.experiment.model.transition._target_.endswith("GaussianTransition")
    assert cfg.experiment.model.encoder._target_.endswith("GaussianEncoder")
    assert cfg.experiment.model.decoder._target_.endswith("GaussianDecoder")
    assert cfg.experiment.model.z_init._target_.endswith("GaussianInitPrior")


@pytest.mark.parametrize("name,expected_dim,expected_j", [
    ("harmonic_gauss", 1, 1),
    ("bimodal_gauss", 1, 1),
    ("robot_2d_gauss", 2, 2),
    ("kdd_gauss", 6, 1),
])
def test_experiment_shape_baked_in(name: str, expected_dim: int, expected_j: int) -> None:
    """Shapes are resolved to concrete ints, not interpolation strings."""
    exp = _exp(name)
    assert exp.model.data_dim == expected_dim
    assert exp.model.j == expected_j
    assert exp.model.encoder.data_dim == expected_dim
    assert exp.model.encoder.j == expected_j
    assert exp.model.transition.j == expected_j


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


@pytest.mark.parametrize("name,expected_metrics", [
    ("harmonic_gauss", ["mae", "crps_sum"]),
    ("bimodal_gauss", ["energy_score", "crps_sum"]),
    ("robot_2d_gauss", ["energy_score", "crps_sum"]),
])
def test_eval_metrics(name: str, expected_metrics: list) -> None:
    assert list(_exp(name).eval.metrics) == expected_metrics


@pytest.mark.parametrize("name,expected_first_plot", [
    ("harmonic_gauss", "forecast_1d"),
    ("bimodal_gauss", "forecast_1d"),
    ("robot_2d_gauss", "forecast_2d_spatial"),
])
def test_viz_first_plot(name: str, expected_first_plot: str) -> None:
    assert _exp(name).viz.plots[0].name == expected_first_plot
