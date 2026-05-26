"""Smoke tests for the tiny-ablation + paper-headline launchers.

The launchers don't submit anything — they only render sbatch text.
These tests verify that (a) the cross-product of (cell, dataset) jobs
materialises with the right count, and (b) each rendered sbatch
script contains the expected Hydra overrides (sweep params + dataset
+ size).
"""

from __future__ import annotations

import pytest

from experiments.init_centering.cells import cell_name, iter_cells
from experiments.init_centering.launch_ablation_tiny import (
    TINY_DATASETS,
    all_tiny_jobs,
    render_tiny_sbatch,
)
from experiments.init_centering.launch_paper_headline import (
    PAPER_DATASETS,
    all_paper_jobs,
    render_paper_sbatch,
)


def test_tiny_jobs_cover_full_cross_product() -> None:
    """36 jobs = 18 cells × 2 datasets."""
    jobs = all_tiny_jobs()
    n_cells = sum(1 for _ in iter_cells())
    assert len(jobs) == n_cells * len(TINY_DATASETS)
    cells_seen = {cell for cell, *_ in jobs}
    assert cells_seen == {cell_name(*c) for c in iter_cells()}


def test_tiny_sbatch_carries_dataset_and_latent_dim_overrides(tmp_path) -> None:
    """Rendered script must override experiment.data + data_dim + latent_dim."""
    storage = str(tmp_path / "optuna")
    sweeps = str(tmp_path / "sweeps")
    cell = cell_name("mlp", "pinned", "per_t")
    script = render_tiny_sbatch(
        cell, "nonlin_bimodal_lift_mv", 8, 4, "mv",
        study_prefix="ablation_test",
        n_trials=2,
        storage_dir=storage,
        sweeps_root=sweeps,
    )
    assert "experiment.data=nonlin_bimodal_lift_mv" in script
    assert "experiment.model.data_dim=8" in script
    assert "experiment.model.latent_dim=4" in script
    assert "+sweep=init_ablation" in script
    assert "hydra.sweeper.n_trials=2" in script
    # Cell-scoped study name so studies don't collide.
    assert f"ablation_test_{cell}__mv" in script


def test_paper_jobs_cross_product_size() -> None:
    """N user-chosen cells × 2 datasets = 2N jobs."""
    top = [cell_name("mlp", "pinned", "per_t"), cell_name("linear", "learnable", "fixed")]
    jobs = all_paper_jobs(top)
    assert len(jobs) == len(top) * len(PAPER_DATASETS)


def test_paper_sbatch_uses_paper_latent_dim(tmp_path) -> None:
    """Paper-headline jobs override to the *doubled* latent_dim."""
    storage = str(tmp_path / "optuna")
    sweeps = str(tmp_path / "sweeps")
    cell = cell_name("mlp", "pinned", "per_t")
    # 1D dataset: tiny is latent_dim=1, paper is latent_dim=2.
    script_1d = render_paper_sbatch(
        cell, "nonlin_bimodal_lift_1d", 1, 2, "1d",
        study_prefix="paper_test",
        n_trials=80,
        storage_dir=storage,
        sweeps_root=sweeps,
    )
    assert "experiment.model.latent_dim=2" in script_1d
    # MV dataset: tiny is latent_dim=4, paper is latent_dim=8.
    script_mv = render_paper_sbatch(
        cell, "nonlin_bimodal_lift_mv", 8, 8, "mv",
        study_prefix="paper_test",
        n_trials=80,
        storage_dir=storage,
        sweeps_root=sweeps,
    )
    assert "experiment.model.latent_dim=8" in script_mv
    assert "hydra.sweeper.n_trials=80" in script_mv


def test_paper_launcher_validates_unknown_cells(tmp_path, monkeypatch) -> None:
    """Pass an unknown cell name → SystemExit with informative message."""
    from experiments.init_centering import launch_paper_headline as lph

    with pytest.raises(SystemExit, match="Unknown cell"):
        lph.main([
            "--top-cells", "init_nonexistent_cell",
            "--dry-run",
        ])
