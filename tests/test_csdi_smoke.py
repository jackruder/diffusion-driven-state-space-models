"""Smoke test for the CSDI experiment family (``csdi_smoke`` / ``csdi_solar``).

Asserts the CSDI baseline flows through the SAME ``Experiment`` workflow the
native DDSSM presets use:

* Both presets REGISTER (appear in the hydra-zen store) and ``instantiate`` of
  ``csdi_smoke`` yields an :class:`~ddssm.experiment.experiment.Experiment` whose
  ``.model`` is a :class:`~ddssm.adapters.csdi.CSDIAdapter`.
* A short end-to-end train run writes ``metrics.csv`` including val rows
  (``loss/total`` for train + val splits) — the objective needs val rows.
* The forecast-metric pipeline writes ``metrics.json`` with the four headline
  metrics present.
* The ``csdi_lean`` SweepSpace validates its fields at import (an unknown field
  would raise), so importing/constructing it is the assertion.

Kept tiny (CPU, layers=2/channels=32/num_steps=10, 20 steps) so it stays fast.

NOTE (deviation, see report): the shipped ``ddssm.eval.runner`` passes
``prepare_model(...).module`` (the raw ``CSDI_Forecasting``) to the metric
registry, but ``forecast`` lives on the *adapter*, not the raw module — so the
standalone ``ddssm.evaluate`` CLI cannot drive CSDI until that one line is
corrected to pass the adapter. That fix lives in ``src/ddssm/eval/runner.py``,
outside this module's allowlist. The eval assertion below therefore drives the
same ``EvalSpec``/metric-registry contract with the adapter as ``ctx.model`` —
exactly what the corrected runner will supply.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import torch
from hydra_zen import instantiate

from ddssm.adapters.csdi import CSDIAdapter

if TYPE_CHECKING:
    from pathlib import Path


def _get_experiment_cfg(name: str) -> object:
    """Register every family, then fetch the named experiment config node."""
    from ddssm.experiment.registry import register_experiments

    register_experiments()  # puts repo root on sys.path + imports experiments
    from ddssm.experiment.stores import store

    for entry in store:
        if entry["group"] == "experiment" and entry["name"] == name:
            return entry["node"]
    raise KeyError(f"Experiment {name!r} not registered")


def _registered_experiment_names() -> set[str]:
    from ddssm.experiment.registry import register_experiments

    register_experiments()
    from ddssm.experiment.stores import store

    return {e["name"] for e in store if e["group"] == "experiment"}


def test_csdi_presets_registered() -> None:
    """Both ``csdi_smoke`` and ``csdi_solar`` register as experiment presets."""
    names = _registered_experiment_names()
    assert "csdi_smoke" in names
    assert "csdi_solar" in names


def test_csdi_smoke_instantiates_to_adapter() -> None:
    """``instantiate(csdi_smoke)`` yields an Experiment whose model is a CSDIAdapter."""
    cfg = _get_experiment_cfg("csdi_smoke")
    exp = instantiate(cfg)
    assert isinstance(exp.model, CSDIAdapter)


def test_csdi_lean_sweep_imports() -> None:
    """The ``csdi_lean`` SweepSpace validates its fields at import (no raise)."""
    from experiments.csdi.sweeps import CSDILeanSweep

    assert CSDILeanSweep is not None


def test_csdi_smoke_end_to_end(tmp_path: Path) -> None:
    """Short run writes metrics.csv with train + val ``loss/total`` rows."""
    cfg = _get_experiment_cfg("csdi_smoke")
    exp = instantiate(cfg)
    exp.training.steps = 20
    exp.training.log_every = 5
    exp.training.validate_every = 10

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    exp.train(device=torch.device("cpu"), run_dir=str(run_dir))

    csv_path = run_dir / "metrics.csv"
    assert csv_path.exists(), "metrics.csv missing"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    assert "loss/total" in fieldnames, f"missing loss/total; got {fieldnames}"
    assert "split" in fieldnames, f"missing split column; got {fieldnames}"
    splits = {r["split"] for r in rows}
    assert "train" in splits, f"no train rows; splits={splits}"
    assert "val" in splits, f"no val rows; splits={splits}"


def test_csdi_smoke_eval_writes_metrics_json(tmp_path: Path) -> None:
    """The forecast-metric pipeline writes metrics.json with the headline metrics.

    Drives the experiment's own ``EvalSpec`` against the CSDI adapter via the
    metric registry (the contract the corrected ``run_eval`` supplies). See the
    module docstring for why we do not call ``Experiment.evaluate`` directly.
    """
    from ddssm.eval.metrics import METRIC_REGISTRY, EvalContext

    cfg = _get_experiment_cfg("csdi_smoke")
    exp = instantiate(cfg)
    exp.training.steps = 10
    exp.training.log_every = 5
    exp.training.validate_every = 10

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    exp.train(device=torch.device("cpu"), run_dir=str(run_dir))

    spec = exp.eval
    assert spec is not None, "csdi_smoke must wire an EvalSpec"

    data = exp.data
    metadata = data.metadata
    ctx = EvalContext(
        model=exp.model,  # the CSDIAdapter (has .forecast)
        loader=data.loader(spec.split),
        device=torch.device("cpu"),
        batch_transform=data.batch_transform,
        T_split=metadata.forecast_split_or(spec.T_split),
        num_samples=int(spec.num_samples),
        run_dir=str(run_dir),
        means=getattr(metadata, "means", None),
        stds=getattr(metadata, "stds", None),
    )
    results: dict = {}
    for name in spec.metrics:
        results.update(METRIC_REGISTRY[name](ctx))

    out_path = run_dir / spec.output_filename
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=float)

    assert out_path.exists(), "metrics.json missing"
    with out_path.open() as f:
        loaded = json.load(f)
    for metric in ("crps_sum", "mae", "rmse", "energy_score"):
        assert metric in loaded, f"metric {metric!r} absent from metrics.json"


def test_csdi_smoke_uses_windowed_data() -> None:
    """The smoke data module is windowed (forecast_split is an int, not None)."""
    cfg = _get_experiment_cfg("csdi_smoke")
    exp = instantiate(cfg)
    assert exp.data.metadata.forecast_split is not None
    assert isinstance(exp.data.metadata.forecast_split, int)
