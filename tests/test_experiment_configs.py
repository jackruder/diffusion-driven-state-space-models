"""Compose-and-instantiate tests for the named experiments.

Each preset registered to ``experiment_store`` (group=``experiment``)
must:

* Compose to an :class:`Experiment` instance via ``instantiate(node)``,
  with populated ``data``, ``model``, ``training``, and a non-empty
  model parameter count.
* Resolve through the Hydra CLI bridge
  (``ddssm._experiment_registry.register_experiments`` →
  ``compose(config_name='config', overrides=[experiment=NAME])``).

Post legacy-purge the only family is init-centering (diffusion / VHP path);
the synthetic / kdd / variance_probe families were deleted (their
datasets survive as library code in ``ddssm.data.presets``).
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
from hydra_zen import store, instantiate
from hydra.core.global_hydra import GlobalHydra

from ddssm.experiment import Experiment, ObjectiveSpec, TrainingScalars
from ddssm.data.datamodule import DDSSMDataModule
from ddssm.experiment.registry import register_experiments

CONF_DIR = (Path(__file__).resolve().parent.parent / "src" / "ddssm" / "conf").as_posix()


def _registered_names(group: str) -> list[str]:
    """Pull the live entries of a zen-store group as a sorted name list."""
    register_experiments()
    if group not in store:
        return []
    return sorted(name for _, name in store[group])


def _exp(name: str):
    """Look up the registered experiment Conf node by name."""
    return store["experiment"]["experiment", name]


# Populated once at collection time so ``parametrize`` sees the same list
# the runtime registry exposes — no hardcoded names.
EXPERIMENTS = _registered_names("experiment")
SWEEPS = _registered_names("sweep")


@pytest.fixture(autouse=True)
def _clear_global_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    register_experiments()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def test_experiments_registered() -> None:
    """All 26 init-centering presets are reachable through the store.

    Composition: 24 ablation-study points (12 cells × 2 datasets, named
    ``init_<cell>__<dataset>``) + 2 role-specific smokes
    (``init_smoke_simple`` and ``init_smoke_high_surface``).
    """
    assert len(EXPERIMENTS) == 26, EXPERIMENTS
    assert all(name.startswith("init_") for name in EXPERIMENTS), EXPERIMENTS


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
    # The init-centering preset builds DDSSM_base through a factory
    # wrapper that wires shared baseline / aux instances.
    target = cfg.experiment.model._target_
    assert target.endswith("_build_init_centering_model"), target
    assert cfg.experiment.data._target_


def test_data_group_override_targets_experiment_data() -> None:
    """``+data=NAME`` overrides the preset's baked dataset.

    The data store is packaged at ``experiment.data`` so a ``+data=``
    selection replaces ``experiment.data`` rather than writing an unread
    top-level ``data:`` key (the old silent no-op).
    """
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["experiment=init_smoke_simple", "+data=harmonic"],
        )
    assert cfg.experiment.data.mode == "harmonic"
    # No stray top-level data: key — the override landed inside experiment.
    assert "data" not in cfg


def test_default_experiment_is_init_smoke_simple() -> None:
    """The default composes to the canonical (zero, pinned, fixed) anchor cell."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config")
    assert cfg.experiment.data._target_.endswith("SyntheticDataModule")
    assert cfg.experiment.model._target_.endswith("_build_init_centering_model")
    assert cfg.experiment.model.baseline_form == "zero"
    assert cfg.experiment.model.baseline_mode == "pinned"
    assert cfg.experiment.model.tracking_mode == "fixed"


@pytest.mark.parametrize("name,expected_data_dim,expected_latent", [
    ("init_smoke_simple", 1, 1),
    ("init_smoke_high_surface", 8, 4),
])
def test_experiment_shape_baked_in(name: str, expected_data_dim: int, expected_latent: int) -> None:
    """Factory shape kwargs resolve to concrete ints, not interpolation strings."""
    exp = _exp(name)
    assert exp.model.data_dim == expected_data_dim
    assert exp.model.latent_dim == expected_latent


def test_objective_returns_inf_on_missing_csv(tmp_path) -> None:
    obj = ObjectiveSpec(metric="loss/total", split="train", tail_frac=0.1)
    assert obj.read(str(tmp_path / "missing.csv")) == float("inf")


@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_preset_composes(name: str) -> None:
    """Every registered sweep preset composes via ``+sweep=NAME``.

    The init-centering Optuna presets (``init_ablation``, ``init_pilot``)
    swap in the ``ddssm_optuna`` sweeper and populate a non-empty search
    space; ``init_ablation_moo`` uses a list of minimize directions.
    """
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"+sweep={name}"],
            return_hydra_config=True,
        )
    sweeper = cfg.hydra.sweeper
    if sweeper.params:
        assert "optuna" in sweeper._target_.lower()
        direction = sweeper.direction
        if isinstance(direction, str):
            assert direction == "minimize"
        else:
            assert all(d == "minimize" for d in direction)
            assert len(direction) >= 2
        assert len(sweeper.params) > 0


def test_high_surface_smoke_eval_metrics() -> None:
    """The high-surface smoke wires the Phase-A headline eval metrics."""
    metrics = list(_exp("init_smoke_high_surface").eval.metrics)
    assert "stage2_elbo_surrogate" in metrics
