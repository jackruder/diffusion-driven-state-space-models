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


# ---------------------------------------------------------------------------
# New synthetic verification presets (harmonic / bimodal / robot).
# ---------------------------------------------------------------------------

NEW_SYNTH_EXPERIMENTS = [
    "harmonic_gauss",
    "harmonic_diff",
    "harmonic_gauss_j2",
    "harmonic_diff_j2",
    "harmonic_noisy_gauss",
    "harmonic_noisy_diff",
    "bimodal_gauss",
    "bimodal_diff",
    "robot_gauss_2d",
    "robot_diff_2d",
]


@pytest.mark.parametrize("name", NEW_SYNTH_EXPERIMENTS)
def test_new_experiment_preset_composes(name: str) -> None:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.experiment.training.steps > 0
    assert cfg.experiment.model._target_.endswith("DDSSM_base")
    assert cfg.experiment.data._target_.endswith("SyntheticDataModule")


@pytest.mark.parametrize("name", NEW_SYNTH_EXPERIMENTS)
def test_new_experiment_instantiates(name: str) -> None:
    """Model builds, shapes are wired correctly, eval/viz specs are present."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    expt = instantiate(cfg.experiment)
    assert isinstance(expt, Experiment)
    assert isinstance(expt.data, DDSSMDataModule)
    assert isinstance(expt.training, TrainingScalars)
    assert isinstance(expt.objective, ObjectiveSpec)
    assert expt.eval is not None, f"{name}: eval spec is None"
    assert expt.viz is not None, f"{name}: viz spec is None"
    n_params = sum(p.numel() for p in expt.model.parameters())
    assert n_params > 0


@pytest.mark.parametrize("name,expected_data_dim,expected_j", [
    ("harmonic_gauss",    1, 1),
    ("harmonic_gauss_j2", 1, 2),
    ("harmonic_diff_j2",  1, 2),
    ("bimodal_diff",      1, 1),
    ("robot_gauss_2d",    2, 2),
    ("robot_diff_2d",     2, 2),
])
def test_new_experiment_shape_fields(name: str, expected_data_dim: int, expected_j: int) -> None:
    """Shape fields in the composed config must match what the model will be built with."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.experiment.data_dim == expected_data_dim, (
        f"{name}: data_dim={cfg.experiment.data_dim} != {expected_data_dim}"
    )
    assert cfg.experiment.j == expected_j, (
        f"{name}: j={cfg.experiment.j} != {expected_j}"
    )


@pytest.mark.parametrize("name,expected_metrics", [
    ("harmonic_gauss",      ["mae", "crps_sum"]),
    ("harmonic_noisy_diff", ["mae", "crps_sum"]),
    ("bimodal_gauss",       ["energy_score", "crps_sum"]),
    ("bimodal_diff",        ["energy_score", "crps_sum"]),
    ("robot_gauss_2d",      ["energy_score", "crps_sum"]),
    ("robot_diff_2d",       ["energy_score", "crps_sum"]),
])
def test_new_experiment_eval_metrics(name: str, expected_metrics: list) -> None:
    """Eval metric list must match the family spec (harmonic→mae, bimodal/robot→energy_score)."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert list(cfg.experiment.eval.metrics) == expected_metrics


@pytest.mark.parametrize("name,expected_first_plot", [
    ("harmonic_gauss",  "forecast_1d"),
    ("bimodal_diff",    "forecast_1d"),
    ("robot_gauss_2d",  "forecast_2d_spatial"),
    ("robot_diff_2d",   "forecast_2d_spatial"),
])
def test_new_experiment_viz_first_plot(name: str, expected_first_plot: str) -> None:
    """Robot presets must use the 2D spatial plot; all others use forecast_1d."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.experiment.viz.plots[0].name == expected_first_plot


@pytest.mark.parametrize("name", ["harmonic_gauss", "bimodal_diff", "robot_gauss_2d"])
def test_synth_eval_conf_has_explicit_t_split(name: str) -> None:
    """Synthetic eval confs with forecasting metrics MUST carry an explicit T_split.

    SyntheticDataModule.metadata.forecast_split is always None, so without an
    explicit T_split the eval runner would silently pass None to _iter_forecast_batches
    and raise at metric-compute time.
    """
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    t_split = cfg.experiment.eval.T_split
    assert t_split is not None and int(t_split) > 0, (
        f"{name}: eval.T_split={t_split!r} — forecast metrics (mae/energy_score/crps_sum) "
        "require a non-None T_split on synthetic datasets"
    )
