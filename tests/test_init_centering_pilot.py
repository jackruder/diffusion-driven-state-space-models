"""Tests for the ``init_smoke_high_surface`` experiment + ``init_ablation`` sweep.

Verifies:

* The high-surface smoke preset and ``init_ablation`` sweep both
  register into the appropriate ``conf.registry`` stores. The
  back-compat alias ``init_pilot`` still resolves.
* The high-surface smoke's ``objective`` is ``stage2_elbo_surrogate``
  read from JSON.
* The ablation sweep declares the seven search axes from the grilling
  decision: ``n_pretrain``, ``sigma_pert``, ``anchor_lambda``,
  ``lambda_sigma_p``, ``base_lr``, ``dec_mult``, ``trans_mult``.
* The eval spec lists the five Phase-A headline metrics.

The legacy filename ``test_init_centering_pilot.py`` is retained for
git-history continuity; the contents now exercise the renamed
high-surface smoke per CONTEXT.md (the term "pilot" was overloaded
and dropped during grilling).
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


def test_high_surface_smoke_registered() -> None:
    """``init_smoke_high_surface`` shows up in the experiment store."""
    register_experiments()
    names = [name for _, name in store["experiment"]]
    assert "init_smoke_high_surface" in names


def test_ablation_sweep_registered_with_back_compat_alias() -> None:
    """``init_ablation`` is the canonical name; ``init_pilot`` aliases it."""
    register_experiments()
    names = [name for _, name in store["sweep"]]
    assert "init_ablation" in names
    assert "init_pilot" in names  # back-compat alias


def test_high_surface_smoke_instantiates() -> None:
    """The high-surface smoke composes through Hydra with the right specs."""
    cfg = store["experiment"]["experiment", "init_smoke_high_surface"]
    exp = instantiate(cfg)
    assert isinstance(exp, Experiment)
    # Objective: JSON-source stage2_elbo_surrogate.
    assert isinstance(exp.objective, ObjectiveSpec)
    assert exp.objective.metric == "stage2_elbo_surrogate"
    assert exp.objective.source == "json"
    # Eval: five Phase-A headline metrics.
    assert exp.eval is not None
    assert set(exp.eval.metrics) == {
        "stage2_elbo_surrogate",
        "sigma_data_drift",
        "wallclock_to_target",
        "crps_sum_latent",
        "gt_latent_jsd",
    }
    assert exp.eval.split == "val"
    # Cell axes: (mlp, learnable, per_t) on the MV dataset.
    assert exp.model.baseline_mode == "learnable"
    assert exp.model.sigma_data.tracking_mode == "per_t"
    assert exp.model.latent_dim == 4


def test_ablation_sweep_composes_via_cli() -> None:
    """``+sweep=init_ablation`` switches the sweeper and populates params."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=init_smoke_high_surface",
                "+sweep=init_ablation",
            ],
            return_hydra_config=True,
        )
    sweeper = cfg.hydra.sweeper
    assert "optuna" in sweeper._target_.lower()
    assert sweeper.direction == "minimize"
    params = dict(sweeper.params)
    # All 7 sweep axes from the grilling decision.
    for key in (
        "experiment.model.stages.n_pretrain",
        "experiment.model.stages.sigma_pert",
        "experiment.model.anchor_lambda",
        "experiment.hparams.lambda_sigma_p",
        "experiment.model.stages.base_lr",
        "experiment.model.stages.dec_mult",
        "experiment.model.stages.trans_mult",
    ):
        assert key in params, f"missing sweep axis: {key}"
    # Sanity: at least the two handoff knobs are log-uniform.
    assert "tag(log" in params["experiment.model.stages.n_pretrain"]
    assert "tag(log" in params["experiment.model.stages.sigma_pert"]


def test_high_surface_smoke_resolves_to_init_centering_factory() -> None:
    """The smoke uses the parametric init-centering factory."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["experiment=init_smoke_high_surface"],
        )
    target = cfg.experiment.model._target_
    assert target.endswith("_build_init_centering_model"), target


@pytest.mark.slow
def test_high_surface_smoke_end_to_end_writes_metrics_json(
    tmp_path: Path,
) -> None:
    """train() chains eval, writes ``metrics.json``, returns a finite float."""
    import json
    import math as _math
    import torch

    cfg = store["experiment"]["experiment", "init_smoke_high_surface"]
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

    assert isinstance(value, float)
    assert _math.isfinite(value), f"objective returned non-finite value: {value}"

    metrics_json = run_dir / "metrics.json"
    assert metrics_json.exists(), "Phase-A metrics.json missing from run_dir"
    payload = json.loads(metrics_json.read_text())
    assert "stage2_elbo_surrogate" in payload, payload.keys()

    assert (run_dir / "checkpoints" / "ckpt_final.pth").exists()
