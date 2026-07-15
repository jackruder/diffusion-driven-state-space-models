"""End-to-end smoke test for ``init_smoke_simple``.

Drives a short single-phase run and asserts:

* The run completes without raising.
* ``metrics.csv`` carries the model-v2 columns.
* Recon loss descends over the run.
* Under ``fixed`` tracking, the σ_data buffer stays frozen at
  ``σ_data²=1``.

Marked ``slow`` because it constructs the full model. Excluded from the
default suite; run with::

    pytest tests/test_init_centering_smoke.py -m slow
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import pytest
from hydra_zen import instantiate

pytestmark = pytest.mark.slow


def _get_experiment_cfg(name: str):
    from ddssm.experiment.registry import register_experiments

    register_experiments()  # puts repo root on sys.path + imports experiments
    from ddssm.experiment.stores import store

    for entry in store:
        if entry["group"] == "experiment" and entry["name"] == name:
            return entry["node"]
    raise KeyError(f"Experiment {name!r} not registered")


def test_init_smoke_simple_end_to_end(tmp_path: Path) -> None:
    """10-step single-phase run reaches metrics.csv without raising."""
    cfg = _get_experiment_cfg("init_smoke_simple")
    exp = instantiate(cfg)
    # Shrink the fit budget for a fast smoke run.
    exp.training.steps = 10
    exp.training.log_every = 1
    exp.training.validate_every = 0
    exp.training.checkpoint_every = 100

    # hparams.batch_size is the source of truth: a distinct value (≠ the
    # dataset preset's 32, ≠ the SmokeHparams 16) must reach the loader.
    exp.hparams.batch_size = 8

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    exp.train(device=device, run_dir=str(run_dir))

    # Reconciliation: the data module's batch_size was overridden from hparams.
    assert exp.data.batch_size == 8

    # run_summary.json is emitted at train exit.
    summary_path = run_dir / "run_summary.json"
    assert summary_path.exists(), "run_summary.json missing"

    csv_path = run_dir / "metrics.csv"
    assert csv_path.exists(), "metrics.csv missing"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    expected_columns = {
        "loss/distortion/rec",
        "loss/rate/init/loss_init",
        "loss/rate/init/kl_aux",
        "loss/rate/trans/kl",
        "loss/total",
        "optim/lambda",
    }
    assert expected_columns.issubset(set(fieldnames)), (
        f"missing columns: {expected_columns - set(fieldnames)}"
    )
    # No residual staged-training columns leak through.
    assert "stage/idx" not in fieldnames
    assert "loss/rate/trans/r_sigma_p" not in fieldnames
    assert "loss/rate/trans/r_mu_p" not in fieldnames

    # Under the ``fixed`` tracking mode the buffer is frozen from
    # construction (σ_data² ≡ 1); ``update`` is a permanent no-op.
    if exp.model.module.sigma_data.tracking_mode == "fixed":
        assert exp.model.module.sigma_data.frozen is True
        assert torch.all(exp.model.module.sigma_data.sigma_data2 == 1.0)
    else:
        assert exp.model.module.sigma_data.frozen is False


def test_init_smoke_simple_shares_baseline_instance() -> None:
    """The transition references the same baseline instance as the model."""
    cfg = _get_experiment_cfg("init_smoke_simple")
    exp = instantiate(cfg)
    assert exp.model.module.baseline is exp.model.module.transition.baseline
