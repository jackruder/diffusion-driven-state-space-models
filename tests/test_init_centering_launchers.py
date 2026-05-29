"""Tests for the generic init-centering launcher (``launch_study``).

``launch_study`` renders Optuna-multirun sbatch scripts from the study's
registered ``init_<cell>__<dataset>`` points. Data + dims are baked into each
preset, so a rendered job carries only the sweep wiring + (paper mode) the size
override — never a ``experiment.data.mode`` override. ``--submit`` (which
requires ``--write-dir``) shells out to ``sbatch`` per file.
"""

from __future__ import annotations

import pytest

from experiments.init_centering import launch_study
from experiments.init_centering.study import INIT_CENTERING_STUDY


def _dry_run(capsys, argv: list[str]) -> str:
    rc = launch_study.main([*argv, "--dry-run"])
    assert rc == 0
    return capsys.readouterr().out


def test_tiny_renders_every_point(capsys) -> None:
    out = _dry_run(capsys, ["--mode", "tiny"])
    headers = [ln for ln in out.splitlines() if ln.startswith("# --- ")]
    assert len(headers) == len(INIT_CENTERING_STUDY.points) == 24
    # Multi-objective sweep matching the cells' PilotMOObjective.
    assert "+sweep=init_ablation_moo" in out


def test_tiny_bakes_data_no_override(capsys) -> None:
    """A tiny job references the registered preset; data is baked, not overridden."""
    out = _dry_run(capsys, ["--mode", "tiny", "--cell", "init_mlp_pinned_per_t"])
    assert "experiment=init_mlp_pinned_per_t__1d" in out
    assert "experiment=init_mlp_pinned_per_t__mv" in out
    # Data lives in the preset now — no per-job data override is emitted.
    assert "experiment.data.mode=" not in out
    assert "experiment.model.latent_dim=" not in out  # tiny size: no size override


def test_datasets_filter_restricts_points(capsys) -> None:
    out = _dry_run(capsys, ["--mode", "tiny", "--datasets", "mv"])
    headers = [ln for ln in out.splitlines() if ln.startswith("# --- ")]
    assert len(headers) == 12  # 12 cells, mv only
    assert all(h.endswith("__mv ---") for h in headers), headers


def test_baseline_form_filter(capsys) -> None:
    out = _dry_run(capsys, ["--mode", "tiny", "--baseline-forms", "mlp"])
    headers = [ln for ln in out.splitlines() if ln.startswith("# --- ")]
    # mlp has 2 modes × 2 tracking × 2 datasets = 8.
    assert len(headers) == 8
    assert all("init_mlp_" in h for h in headers), headers


def test_paper_emits_size_override(capsys) -> None:
    out = _dry_run(
        capsys, ["--mode", "paper", "--top-cells", "init_mlp_pinned_per_t"]
    )
    headers = [ln for ln in out.splitlines() if ln.startswith("# --- ")]
    assert len(headers) == 2  # one cell × 2 datasets
    # Paper doubles latent_dim: 1d 1->2, mv 4->8.
    assert "experiment.model.latent_dim=2" in out
    assert "experiment.model.latent_dim=8" in out


def test_paper_requires_top_cells() -> None:
    with pytest.raises(SystemExit):
        launch_study.main(["--mode", "paper", "--dry-run"])


def test_unknown_top_cell_errors() -> None:
    with pytest.raises(SystemExit):
        launch_study.main(["--mode", "paper", "--top-cells", "init_nope", "--dry-run"])


def test_submit_requires_write_dir() -> None:
    with pytest.raises(SystemExit):
        launch_study.main(["--mode", "tiny", "--submit"])


def test_submit_shells_out_once_per_written_file(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def _fake_submit(path: str) -> str:
        calls.append(path)
        return "Submitted batch job 999"

    # run_launcher binds submit_sbatch via experiments._launch.
    monkeypatch.setattr("experiments._launch.submit_sbatch", _fake_submit)

    write_dir = tmp_path / "sbatch"
    rc = launch_study.main([
        "--mode", "tiny", "--datasets", "1d",
        "--baseline-forms", "mlp",
        "--write-dir", str(write_dir),
        "--storage-dir", str(tmp_path / "optuna"),
        "--sweeps-root", str(tmp_path / "sweeps"),
        "--submit",
    ])
    assert rc == 0
    written = sorted(str(p) for p in write_dir.glob("*.sbatch"))
    assert len(written) == 4  # mlp × 2 modes × 2 tracking × 1 dataset
    assert sorted(calls) == written
