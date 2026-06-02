"""Phase-C tests for :class:`ObjectiveSpec.source` (csv / json branch).

Verifies the two read paths in isolation — CSV (legacy) and JSON
(post-eval) — plus the failure modes (missing file, missing key,
non-finite value) that surface as ``+inf`` so failed Optuna trials
register cleanly under ``minimize``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ddssm.experiment import Experiment, ObjectiveSpec

# ---------------------------------------------------------------------------
# CSV source — preserves the pre-Phase-C behaviour.
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv as _csv

    with open(path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_csv_source_returns_tail_mean(tmp_path: Path) -> None:
    """Default CSV source: read tail of ``loss/total``."""
    csv_path = tmp_path / "metrics.csv"
    _write_csv(csv_path, [
        {"step": str(i), "split": "train", "loss/total": str(float(i))}
        for i in range(1, 11)
    ])
    spec = ObjectiveSpec(metric="loss/total", split="train", tail_frac=0.2)
    # tail_frac=0.2 over 10 rows → tail_n=2 (rows 9.0 and 10.0)
    assert spec.read(str(csv_path)) == pytest.approx(9.5)


def test_csv_source_filters_by_split(tmp_path: Path) -> None:
    """Rows whose ``split`` doesn't match are skipped."""
    csv_path = tmp_path / "metrics.csv"
    _write_csv(csv_path, [
        {"step": "1", "split": "train", "loss/total": "1.0"},
        {"step": "2", "split": "val", "loss/total": "100.0"},
        {"step": "3", "split": "train", "loss/total": "2.0"},
    ])
    train = ObjectiveSpec(metric="loss/total", split="train", tail_frac=1.0)
    assert train.read(str(csv_path)) == pytest.approx(1.5)


def test_csv_source_returns_inf_on_missing_file(tmp_path: Path) -> None:
    """No CSV ⇒ ``+inf`` so the trial fails cleanly."""
    spec = ObjectiveSpec(metric="loss/total")
    assert math.isinf(spec.read(str(tmp_path / "nonexistent.csv")))


def test_csv_metric_fallback_warns(tmp_path: Path, caplog) -> None:
    """Missing configured metric → falls back to a 'loss' column AND warns."""
    import logging

    csv_path = tmp_path / "metrics.csv"
    _write_csv(csv_path, [
        {"step": "1", "split": "train", "loss/total": "2.0"},
        {"step": "2", "split": "train", "loss/total": "4.0"},
    ])
    spec = ObjectiveSpec(metric="not_a_real_metric", split="train", tail_frac=1.0)
    with caplog.at_level(logging.WARNING, logger="ddssm.experiment"):
        val = spec.read(str(csv_path))
    # Behaviour unchanged: still falls back to loss/total (mean 3.0).
    assert val == pytest.approx(3.0)
    assert "not_a_real_metric" in caplog.text
    assert "falling back" in caplog.text


def test_csv_split_missing_column_warns(tmp_path: Path, caplog) -> None:
    """split set but no split column → no-op filter, surfaced as a warning."""
    import logging

    csv_path = tmp_path / "metrics.csv"
    _write_csv(csv_path, [
        {"step": "1", "loss/total": "1.0"},
        {"step": "2", "loss/total": "3.0"},
    ])
    spec = ObjectiveSpec(metric="loss/total", split="train", tail_frac=1.0)
    with caplog.at_level(logging.WARNING, logger="ddssm.experiment"):
        val = spec.read(str(csv_path))
    # No filtering happened — mean over all rows.
    assert val == pytest.approx(2.0)
    assert "no 'split' column" in caplog.text


def test_json_missing_key_warns(tmp_path: Path, caplog) -> None:
    """Missing JSON metric key → penalty applied AND warns."""
    import logging

    json_path = tmp_path / "metrics.json"
    json_path.write_text(json.dumps({"some_other_metric": 1.0}))
    spec = ObjectiveSpec(metric="mae", source="json", penalty="inf")
    with caplog.at_level(logging.WARNING, logger="ddssm.experiment"):
        val = spec.read(str(tmp_path))
    assert math.isinf(val)
    assert "mae" in caplog.text


def test_csv_source_accepts_run_dir(tmp_path: Path) -> None:
    """Passing a run_dir (not a file) reads ``metrics.csv`` inside it."""
    csv_path = tmp_path / "metrics.csv"
    _write_csv(csv_path, [
        {"step": "1", "split": "train", "loss/total": "0.7"},
    ])
    spec = ObjectiveSpec(metric="loss/total", split="train", tail_frac=1.0)
    assert spec.read(str(tmp_path)) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# JSON source — Phase-C eval-as-objective branch.
# ---------------------------------------------------------------------------


def test_json_source_reads_scalar(tmp_path: Path) -> None:
    """``source='json'`` indexes ``metrics.json`` by ``metric`` name."""
    (tmp_path / "metrics.json").write_text(json.dumps({
        "stage2_elbo_surrogate": 0.42,
        "sigma_data_drift_mean": 0.1,
    }))
    spec = ObjectiveSpec(metric="stage2_elbo_surrogate", source="json")
    assert spec.read(str(tmp_path)) == pytest.approx(0.42)


def test_json_source_returns_inf_on_missing_key(tmp_path: Path) -> None:
    """Missing key ⇒ ``+inf``."""
    (tmp_path / "metrics.json").write_text(json.dumps({"other_metric": 1.0}))
    spec = ObjectiveSpec(metric="stage2_elbo_surrogate", source="json")
    assert math.isinf(spec.read(str(tmp_path)))


def test_json_source_returns_inf_on_non_finite_value(tmp_path: Path) -> None:
    """NaN / inf values surface as ``+inf`` (not as the raw value)."""
    (tmp_path / "metrics.json").write_text(
        '{"stage2_elbo_surrogate": "NaN"}'
    )
    spec = ObjectiveSpec(metric="stage2_elbo_surrogate", source="json")
    result = spec.read(str(tmp_path))
    assert math.isinf(result) and result > 0


def test_json_source_returns_inf_on_missing_file(tmp_path: Path) -> None:
    """No ``metrics.json`` ⇒ ``+inf``."""
    spec = ObjectiveSpec(metric="anything", source="json")
    assert math.isinf(spec.read(str(tmp_path)))


def test_json_source_accepts_direct_file_path(tmp_path: Path) -> None:
    """Direct JSON path is supported (mirrors CSV's direct-file path)."""
    json_path = tmp_path / "custom.json"
    json_path.write_text(json.dumps({"k": 7.0}))
    spec = ObjectiveSpec(metric="k", source="json")
    assert spec.read(str(json_path)) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Multi-objective (``Objectives``) — regression for the MOO instantiation
# bug. When a MOO objective is built the way the presets build it
# (``ddssm.builders.Objectives(specs=[Objective(...), ...])``) and then
# run through ``hydra_zen.instantiate``, OmegaConf coerces each nested
# spec into an ``ObjectiveSpec``-typed ``DictConfig`` and strips its
# ``_target_`` — so the elements are NOT live ``ObjectiveSpec`` objects
# and have no ``.read`` method. ``Experiment.objective_value`` used to
# call ``o.read(...)`` straight on them and crashed with
# ``ConfigAttributeError: Key 'read' not in 'ObjectiveSpec'``, which
# broke every MOO Optuna trial. These tests pin the contract: a MOO
# objective built + instantiated the production way must yield a list of
# finite floats from ``objective_value``.
# ---------------------------------------------------------------------------


def _experiment_with_objective(objective: object) -> Experiment:
    """A bare ``Experiment`` carrying only ``objective``.

    ``objective_value``'s csv-source path touches nothing else (no eval,
    trainer, model, or data), so we skip ``__init__`` to avoid building
    the full composition just to exercise the objective-read contract.
    """
    exp = Experiment.__new__(Experiment)
    exp.objective = objective
    exp.eval = None
    return exp


def test_moo_objective_instantiates_and_reads(tmp_path: Path) -> None:
    """MOO objective via builders+instantiate yields a readable spec list.

    Reproduces the production wiring from ``init_centering/evals.py``.
    Pre-fix this raised ``ConfigAttributeError`` because the instantiated
    specs were ``DictConfig`` without ``.read``.
    """
    import torch
    from hydra_zen import instantiate

    from ddssm.experiment import Objectives
    from ddssm.experiment.builders import (
        Objective as ObjectiveCfg,
        Objectives as ObjectivesCfg,
    )

    cfg = ObjectivesCfg(specs=[
        ObjectiveCfg(metric="m1", split="train", tail_frac=1.0, source="csv"),
        ObjectiveCfg(metric="m2", split="train", tail_frac=1.0, source="csv"),
    ])
    obj = instantiate(cfg)
    assert isinstance(obj, Objectives)
    # Guard the bug's precondition: the nested specs come back as
    # OmegaConf nodes, not live ObjectiveSpec objects.
    assert not all(isinstance(s, ObjectiveSpec) for s in obj.specs)

    _write_csv(tmp_path / "metrics.csv", [
        {"step": "1", "split": "train", "m1": "1.0", "m2": "10.0"},
        {"step": "2", "split": "train", "m1": "3.0", "m2": "30.0"},
    ])

    exp = _experiment_with_objective(obj)
    values = exp.objective_value(
        device=torch.device("cpu"), run_dir=str(tmp_path),
    )
    assert isinstance(values, list)
    assert values == [pytest.approx(2.0), pytest.approx(20.0)]


def test_moo_objective_penalty_survives_instantiation(tmp_path: Path) -> None:
    """A per-spec ``penalty`` survives the instantiate→coerce round-trip.

    Set on a MOO spec, it must not be reset to the ``inf`` default.
    """
    import torch
    from hydra_zen import instantiate

    from ddssm.experiment.builders import (
        Objective as ObjectiveCfg,
        Objectives as ObjectivesCfg,
    )

    cfg = ObjectivesCfg(specs=[
        # json metric is absent from metrics.json → penalty kicks in.
        ObjectiveCfg(
            metric="missing", source="json", penalty="csv_tail_time",
        ),
    ])
    obj = instantiate(cfg)

    _write_csv(tmp_path / "metrics.csv", [
        {"step": "1", "split": "train", "time/elapsed_s": "12.5"},
    ])
    (tmp_path / "metrics.json").write_text(json.dumps({"other": 1.0}))

    exp = _experiment_with_objective(obj)
    # eval is None + a json-source spec ⇒ objective_value short-circuits
    # to the +inf penalty list, so this exercises the coercion before the
    # eval gate. Assert the penalty type survived (csv_tail_time, not the
    # inf default) by reading the spec directly post-coercion.
    from ddssm.experiment import _as_objective_spec

    spec = _as_objective_spec(obj.specs[0])
    assert spec.penalty == "csv_tail_time"
    assert spec.read(str(tmp_path)) == pytest.approx(12.5)
    # And the full path returns a finite penalty value (not a crash).
    values = exp.objective_value(
        device=torch.device("cpu"), run_dir=str(tmp_path),
    )
    assert isinstance(values, list) and len(values) == 1


# ---------------------------------------------------------------------------
# csv_tail_step penalty — step-denominated sibling of csv_tail_time, for
# steps-to-target objectives (e.g. ``wallclock_to_target_step``). A miss costs
# the full step budget, keeping misses on the same units as hits.
# ---------------------------------------------------------------------------


def test_csv_tail_step_penalty_on_miss(tmp_path: Path) -> None:
    """JSON metric absent + ``penalty='csv_tail_step'`` ⇒ last CSV ``step``."""
    _write_csv(tmp_path / "metrics.csv", [
        {"step": "100", "split": "train", "loss/total": "5.0"},
        {"step": "4900", "split": "train", "loss/total": "1.0"},
        {"step": "5000", "split": "train", "loss/total": "0.5"},
    ])
    (tmp_path / "metrics.json").write_text(json.dumps({"other": 1.0}))
    spec = ObjectiveSpec(
        metric="wallclock_to_target_step", source="json", penalty="csv_tail_step",
    )
    # Target never reached → cost the full step budget (last step = 5000).
    assert spec.read(str(tmp_path)) == pytest.approx(5000.0)


def test_csv_tail_step_hit_returns_recorded_value(tmp_path: Path) -> None:
    """When the metric IS present, the step penalty is never consulted."""
    (tmp_path / "metrics.json").write_text(
        json.dumps({"wallclock_to_target_step": 1234})
    )
    spec = ObjectiveSpec(
        metric="wallclock_to_target_step", source="json", penalty="csv_tail_step",
    )
    assert spec.read(str(tmp_path)) == pytest.approx(1234.0)


def test_csv_tail_step_returns_inf_without_csv(tmp_path: Path) -> None:
    """Miss + no ``metrics.csv`` ⇒ ``+inf`` (no step budget to fall back on)."""
    (tmp_path / "metrics.json").write_text(json.dumps({"other": 1.0}))
    spec = ObjectiveSpec(
        metric="wallclock_to_target_step", source="json", penalty="csv_tail_step",
    )
    result = spec.read(str(tmp_path))
    assert math.isinf(result) and result > 0


def test_csv_tail_step_survives_instantiation(tmp_path: Path) -> None:
    """``penalty='csv_tail_step'`` survives the hydra-zen instantiate round-trip."""
    from hydra_zen import instantiate

    from ddssm.experiment import _as_objective_spec
    from ddssm.experiment.builders import (
        Objective as ObjectiveCfg,
        Objectives as ObjectivesCfg,
    )

    cfg = ObjectivesCfg(specs=[
        ObjectiveCfg(
            metric="wallclock_to_target_step", source="json", penalty="csv_tail_step",
        ),
    ])
    obj = instantiate(cfg)
    spec = _as_objective_spec(obj.specs[0])
    assert spec.penalty == "csv_tail_step"
    _write_csv(tmp_path / "metrics.csv", [
        {"step": "5000", "split": "train", "loss/total": "0.5"},
    ])
    (tmp_path / "metrics.json").write_text(json.dumps({"other": 1.0}))
    assert spec.read(str(tmp_path)) == pytest.approx(5000.0)
