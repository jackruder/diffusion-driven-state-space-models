"""Phase-C tests for the ``init_centering_pilot`` experiment + ``init_pilot`` sweep.

Verifies:

* Both presets register into the appropriate ``conf.registry`` stores.
* The pilot experiment's ``objective`` is configured with
  ``source='json'`` and ``metric='stage2_elbo_surrogate'``.
* The pilot sweep declares the two doc-mandated search axes
  (``n_pretrain``, ``sigma_pert``) under
  ``hydra.sweeper.params``.
* The pilot eval spec lists the five Phase-A headline metrics.

The actual 20-trial Optuna run is a manual user-driven smoke; the
fast suite only asserts the wiring is right.
"""

from __future__ import annotations

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate
import pytest

from conf.registry import store
from ddssm._experiment_registry import register_experiments
from ddssm.experiment import Experiment, ObjectiveSpec
from pathlib import Path


CONF_DIR = (Path(__file__).resolve().parent.parent / "src" / "ddssm" / "conf").as_posix()


@pytest.fixture(autouse=True)
def _clear_global_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    register_experiments()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def test_pilot_experiment_registered() -> None:
    """``init_centering_pilot`` shows up in the experiment store."""
    register_experiments()
    names = [name for _, name in store["experiment"]]
    assert "init_centering_pilot" in names


def test_pilot_sweep_registered() -> None:
    """``init_pilot`` shows up in the sweep store."""
    register_experiments()
    names = [name for _, name in store["sweep"]]
    assert "init_pilot" in names


def test_pilot_experiment_instantiates() -> None:
    """Pilot composes through Hydra and exposes the eval + objective specs."""
    cfg = store["experiment"]["experiment", "init_centering_pilot"]
    exp = instantiate(cfg)
    assert isinstance(exp, Experiment)
    # Objective must be the JSON-source ``stage2_elbo_surrogate``.
    assert isinstance(exp.objective, ObjectiveSpec)
    assert exp.objective.metric == "stage2_elbo_surrogate"
    assert exp.objective.source == "json"
    # Eval must list the five Phase-A headline metrics.
    assert exp.eval is not None
    assert set(exp.eval.metrics) == {
        "stage2_elbo_surrogate",
        "sigma_data_drift",
        "wallclock_to_target",
        "crps_sum_latent",
        "gt_latent_jsd",
    }
    assert exp.eval.split == "val"


def test_pilot_sweep_composes_via_cli() -> None:
    """``+sweep=init_pilot`` switches the sweeper and populates params."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=init_centering_pilot",
                "+sweep=init_pilot",
            ],
            return_hydra_config=True,
        )
    sweeper = cfg.hydra.sweeper
    assert "optuna" in sweeper._target_.lower()
    assert sweeper.direction == "minimize"
    params = dict(sweeper.params)
    # Doc-mandated sweep axes.
    assert "experiment.model.stages.n_pretrain" in params
    assert "experiment.model.stages.sigma_pert" in params
    # Sanity: the n_pretrain axis is a log-uniform integer range.
    assert "tag(log" in params["experiment.model.stages.n_pretrain"]
    assert "tag(log" in params["experiment.model.stages.sigma_pert"]


def test_pilot_model_target_resolves_to_init_centering_factory() -> None:
    """Pilot reuses the same model factory as the smoke preset."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["experiment=init_centering_pilot"],
        )
    target = cfg.experiment.model._target_
    assert target.endswith("_build_init_centering_model"), target


@pytest.mark.slow
def test_pilot_end_to_end_writes_metrics_json_and_returns_objective_value(
    tmp_path: Path,
) -> None:
    """The pilot's train() chains eval, writes ``metrics.json``, returns a float.

    5 + 5 steps with the canonical cell; the
    ``stage2_elbo_surrogate`` objective should land as a finite scalar
    and ``metrics.json`` should appear in the run directory.
    """
    import json
    import torch

    cfg = store["experiment"]["experiment", "init_centering_pilot"]
    exp = instantiate(cfg)
    # Shrink stages for a fast run.
    exp.model.config.stages.stage_1.steps = 5
    exp.model.config.stages.stage_2.steps = 5
    exp.model.config.stages.stage_1.log_every = 1
    exp.model.config.stages.stage_2.log_every = 1
    exp.model.config.stages.stage_1.val_every = 0
    exp.model.config.stages.stage_2.val_every = 0
    exp.model.config.stages.stage_1.checkpoint_every = 100
    exp.model.config.stages.stage_2.checkpoint_every = 100

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    value = exp.train(device=torch.device("cpu"), run_dir=str(run_dir))

    # The objective wires through Experiment.train and surfaces a scalar.
    assert isinstance(value, float)
    import math as _math
    assert _math.isfinite(value), f"objective returned non-finite value: {value}"

    # metrics.json must exist with the eval-pipeline's keys.
    metrics_json = run_dir / "metrics.json"
    assert metrics_json.exists(), "Phase-A metrics.json missing from run_dir"
    payload = json.loads(metrics_json.read_text())
    assert "stage2_elbo_surrogate" in payload, payload.keys()

    # The final checkpoint was saved (needed for Phase-E reporting).
    assert (run_dir / "checkpoints" / "ckpt_final.pth").exists()
