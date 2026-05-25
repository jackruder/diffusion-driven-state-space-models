"""Phase-E report-pipeline tests.

Build a synthetic Phase-D output layout in ``tmp_path`` (a handful of
fake ``metrics.json`` files + a tiny Optuna SQLite DB), then exercise:

* ``aggregate`` — produces the right per-trial records, scalars +
  trajectories populated from the JSON payloads.
* ``save_artifacts`` / ``load_records`` — round-trip through
  ``summary.csv`` + ``records.jsonl``.
* ``plot_*`` + ``write_headline_table`` — read the JSONL artifact and
  write non-empty output files (no model is ever instantiated).

The user's "serialize before plotting" directive is checked by the
``test_plot_only_reads_artifacts_no_aggregation`` test: the plot
pipeline must succeed against a record store that *no longer has the
original sweep dirs on disk*.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from experiments.init_centering.cells import cell_name
from experiments.init_centering.report import (
    RECORDS_FILENAME,
    SUMMARY_FILENAME,
    TrialRecord,
    aggregate,
    load_records,
    plot_sigma_data_drift,
    plot_wallclock_to_target,
    save_artifacts,
    write_headline_table,
)


# A small representative subset of cells (not all 18, to keep tests fast).
_TEST_CELLS = [
    cell_name("zero", "pinned", "fixed"),
    cell_name("mlp", "pinned", "per_t"),
]
_TEST_CONTROL = "init_canonical_ctrl_sigma0"


def _fake_metrics_payload(
    *, elbo: float, wallclock: float, sigma_t_max: int = 6,
) -> dict:
    """A complete fake ``metrics.json`` matching the Phase-A schema."""
    return {
        "stage2_elbo_surrogate": elbo,
        "stage2_elbo_surrogate_recon": elbo - 0.1,
        "wallclock_to_target_step": 50,
        "wallclock_to_target_seconds": wallclock,
        "wallclock_to_target_value": elbo,
        "wallclock_to_target_direction": "min",
        "wallclock_to_target_column": "loss/total",
        "sigma_data_drift_available": True,
        "sigma_data2_buffer": [1.0 + 0.1 * i for i in range(sigma_t_max)],
        "sigma_data2_t_indices": list(range(2, sigma_t_max + 1)),
        "sigma_data2_component1_per_t": [0.5] * (sigma_t_max - 1),
        "sigma_data2_component2_per_t": [0.3] * (sigma_t_max - 1),
        "sigma_data2_decomposition_sum_per_t": [0.8] * (sigma_t_max - 1),
        "crps_sum_latent_available": True,
        "crps_sum_latent_mean": 0.42,
        "crps_sum_latent_per_t": [0.4, 0.42, 0.44, 0.46],
        "gt_latent_jsd_available": True,
        "gt_latent_jsd_mean": 0.12,
        "gt_latent_jsd_per_t": [0.10, 0.12, 0.13, 0.14],
        "gt_latent_jsd_t_indices": [1, 2, 3, 4],
        "gt_latent_jsd_n_batches": 4,
    }


def _build_fake_sweep_layout(
    root: Path,
    *,
    study_prefix: str = "phase_d",
    n_trials_per_cell: int = 3,
) -> tuple[Path, Path]:
    """Mock the Phase-D launcher's on-disk output.

    Returns ``(sweeps_root, optuna_dir)``.
    """
    sweeps_root = root / "runs" / "sweeps"
    optuna_dir = root / "runs" / "optuna"
    sweeps_root.mkdir(parents=True)
    optuna_dir.mkdir(parents=True)

    for cell in _TEST_CELLS:
        sweep_dir = sweeps_root / f"{study_prefix}_{cell}"
        sweep_dir.mkdir(parents=True)
        for k in range(n_trials_per_cell):
            trial_dir = sweep_dir / str(k)
            trial_dir.mkdir()
            payload = _fake_metrics_payload(
                elbo=0.5 + 0.01 * k, wallclock=120.0 + 5 * k,
            )
            with open(trial_dir / "metrics.json", "w") as f:
                json.dump(payload, f)

    # Control: single trial, directly inside the sweep dir.
    control_dir = sweeps_root / f"{study_prefix}_{_TEST_CONTROL}"
    control_dir.mkdir()
    with open(control_dir / "metrics.json", "w") as f:
        json.dump(
            _fake_metrics_payload(elbo=0.45, wallclock=110.0),
            f,
        )
    return sweeps_root, optuna_dir


def _build_optuna_dbs(
    optuna_dir: Path,
    *,
    study_prefix: str = "phase_d",
    n_trials_per_cell: int = 3,
) -> None:
    """Create one Optuna SQLite DB per cell with N completed trials."""
    optuna = pytest.importorskip("optuna")

    for cell in _TEST_CELLS:
        study_name = f"{study_prefix}_{cell}"
        db_path = optuna_dir / f"{study_name}.db"
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            direction="minimize",
        )
        for k in range(n_trials_per_cell):
            trial = optuna.trial.create_trial(
                params={"n_pretrain": 100 + 50 * k, "sigma_pert": 1e-3 * (k + 1)},
                distributions={
                    "n_pretrain": optuna.distributions.IntDistribution(
                        50, 2000, log=True,
                    ),
                    "sigma_pert": optuna.distributions.FloatDistribution(
                        1e-4, 1e-1, log=True,
                    ),
                },
                value=0.5 + 0.01 * k,
            )
            study.add_trial(trial)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_returns_one_record_per_trial(tmp_path: Path) -> None:
    """aggregate yields one record per (cell, trial) seen on disk."""
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    _build_optuna_dbs(optuna_dir)

    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    # 2 cells × 3 trials + 1 control = 7 records
    assert len(records) == 2 * 3 + 1


def test_aggregate_lifts_headline_scalars_from_metrics_json(tmp_path: Path) -> None:
    """The scalar fields agree with the synthetic ``metrics.json`` payloads."""
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    _build_optuna_dbs(optuna_dir)
    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    for r in records:
        if not r.is_control:
            # trial number k ⇒ elbo = 0.5 + 0.01 * k, wallclock = 120 + 5k
            expected_elbo = 0.5 + 0.01 * r.trial_number
            expected_wc = 120.0 + 5.0 * r.trial_number
            assert r.stage2_elbo_surrogate == pytest.approx(expected_elbo)
            assert r.wallclock_to_target_seconds == pytest.approx(expected_wc)
            assert r.crps_sum_latent_mean == pytest.approx(0.42)
            assert r.gt_latent_jsd_mean == pytest.approx(0.12)
            # σ_data² buffer is length-6 in the synthetic payload.
            assert len(r.sigma_data2_buffer) == 6
            assert r.sigma_data2_buffer_mean is not None


def test_aggregate_joins_optuna_metadata(tmp_path: Path) -> None:
    """Trial value, state, and params come from the cell's Optuna DB."""
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    _build_optuna_dbs(optuna_dir)
    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    for r in records:
        if not r.is_control:
            assert r.optuna_value is not None
            assert r.optuna_state == "COMPLETE"
            assert "n_pretrain" in r.params
            assert "sigma_pert" in r.params


def test_aggregate_skips_missing_optuna_db_gracefully(tmp_path: Path) -> None:
    """A missing DB leaves optuna_* fields as None — aggregation still works."""
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    # Intentionally do not build the DBs.
    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    for r in records:
        if not r.is_control:
            assert r.optuna_value is None
            assert r.optuna_state is None
            assert r.params == {}
            # But headline metrics still came from metrics.json.
            assert r.stage2_elbo_surrogate is not None


def test_aggregate_marks_control_cells(tmp_path: Path) -> None:
    """Control cells get ``is_control=True``; their sweep dir IS the run dir."""
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    controls = [r for r in records if r.is_control]
    assert len(controls) == 1
    assert controls[0].cell_name == _TEST_CONTROL
    assert controls[0].trial_number == 0


# ---------------------------------------------------------------------------
# Artifact IO
# ---------------------------------------------------------------------------


def test_save_artifacts_writes_summary_and_jsonl(tmp_path: Path) -> None:
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    _build_optuna_dbs(optuna_dir)
    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    out_dir = tmp_path / "report"
    summary, jsonl = save_artifacts(records, str(out_dir))
    assert os.path.isfile(summary)
    assert os.path.isfile(jsonl)
    # summary.csv has a header + one row per record.
    with open(summary) as f:
        lines = f.read().splitlines()
    assert len(lines) == len(records) + 1
    # records.jsonl has exactly len(records) lines, each valid JSON.
    with open(jsonl) as f:
        for line in f:
            payload = json.loads(line)
            assert "cell_name" in payload
            assert "sigma_data2_buffer" in payload


def test_load_records_round_trips(tmp_path: Path) -> None:
    """Save then load yields TrialRecord instances with matching scalars."""
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    _build_optuna_dbs(optuna_dir)
    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    out_dir = tmp_path / "report"
    _, jsonl = save_artifacts(records, str(out_dir))
    loaded = load_records(jsonl)
    assert len(loaded) == len(records)
    # Spot-check round-trip on the scalars + a trajectory.
    for orig, lo in zip(records, loaded):
        assert isinstance(lo, TrialRecord)
        assert lo.cell_name == orig.cell_name
        assert lo.trial_number == orig.trial_number
        assert lo.stage2_elbo_surrogate == orig.stage2_elbo_surrogate
        assert lo.sigma_data2_buffer == orig.sigma_data2_buffer


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def test_plot_sigma_data_drift_writes_nonempty_png(tmp_path: Path) -> None:
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    out_png = tmp_path / "plots" / "sigma_data_drift.png"
    plot_sigma_data_drift(records, str(out_png))
    assert out_png.is_file()
    assert out_png.stat().st_size > 0


def test_plot_wallclock_writes_nonempty_png(tmp_path: Path) -> None:
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    out_png = tmp_path / "plots" / "wallclock.png"
    plot_wallclock_to_target(records, str(out_png))
    assert out_png.is_file()
    assert out_png.stat().st_size > 0


def test_write_headline_table_emits_markdown(tmp_path: Path) -> None:
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    out_md = tmp_path / "headline.md"
    write_headline_table(records, str(out_md))
    text = out_md.read_text()
    assert text.startswith("# Phase E")
    assert "stage2_elbo_surrogate" in text
    for cell in _TEST_CELLS + [_TEST_CONTROL]:
        assert cell in text


# ---------------------------------------------------------------------------
# The user's explicit invariant: plots iterate without re-aggregating.
# ---------------------------------------------------------------------------


def test_plot_only_reads_artifacts_no_aggregation(tmp_path: Path) -> None:
    """Once artifacts are saved, plot fns must not need the sweep dirs.

    Save artifacts, then ``rm -rf`` the sweep dirs.  The plot pipeline
    should still produce all three outputs from the JSONL alone.
    """
    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    _build_optuna_dbs(optuna_dir)
    records = aggregate(
        str(sweeps_root), optuna_dir=str(optuna_dir), study_prefix="phase_d",
    )
    out_dir = tmp_path / "report"
    _, jsonl = save_artifacts(records, str(out_dir))

    # Detonate every aggregation input.
    shutil.rmtree(sweeps_root)
    shutil.rmtree(optuna_dir)

    # All plot helpers should still work — they only read records.jsonl.
    loaded = load_records(jsonl)
    plot_sigma_data_drift(loaded, str(out_dir / "plots" / "sigma_data_drift.png"))
    plot_wallclock_to_target(loaded, str(out_dir / "plots" / "wallclock.png"))
    write_headline_table(loaded, str(out_dir / "plots" / "headline.md"))
    assert (out_dir / "plots" / "sigma_data_drift.png").is_file()
    assert (out_dir / "plots" / "wallclock.png").is_file()
    assert (out_dir / "plots" / "headline.md").is_file()


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_all_end_to_end(tmp_path: Path) -> None:
    """``python -m experiments.init_centering.report all ...`` end-to-end."""
    from experiments.init_centering.report import main

    sweeps_root, optuna_dir = _build_fake_sweep_layout(tmp_path)
    _build_optuna_dbs(optuna_dir)
    out_dir = tmp_path / "report"
    rc = main([
        "all",
        "--sweeps-root", str(sweeps_root),
        "--optuna-dir", str(optuna_dir),
        "--study-prefix", "phase_d",
        "--out", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / SUMMARY_FILENAME).is_file()
    assert (out_dir / RECORDS_FILENAME).is_file()
    assert (out_dir / "plots" / "sigma_data_drift.png").is_file()
    assert (out_dir / "plots" / "wallclock_to_target.png").is_file()
    assert (out_dir / "plots" / "headline_table.md").is_file()
