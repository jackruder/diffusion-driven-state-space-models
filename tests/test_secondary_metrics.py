"""Tests for the two trivial-subset secondary metrics (task #18, F10)."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import pytest

from ddssm.eval.metrics import (
    METRIC_REGISTRY,
    EvalContext,
    eval_q_aux_kl_trajectory,
    eval_log_sigma_p2_collapse,
)


def _make_csv(tmp_path: Path, rows: list[dict]) -> str:
    """Write a minimal metrics.csv with the given rows; return the path."""
    csv_path = tmp_path / "metrics.csv"
    if not rows:
        with open(csv_path, "w") as f:
            f.write("step,loss/rate/init/kl_aux\n")
        return str(csv_path)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return str(csv_path)


def _ctx_with_run_dir(run_dir: str) -> EvalContext:
    return EvalContext(
        model=None,
        loader=None,
        device=torch.device("cpu"),
        run_dir=run_dir,
    )


# q_aux_kl_trajectory (metric #5)


def test_q_aux_kl_trajectory_registered() -> None:
    assert "q_aux_kl_trajectory" in METRIC_REGISTRY


def test_q_aux_kl_trajectory_reads_csv_trajectory(tmp_path: Path) -> None:
    rows = [
        {"step": "1", "loss/rate/init/kl_aux": "0.001"},
        {"step": "2", "loss/rate/init/kl_aux": "0.5"},
        {"step": "3", "loss/rate/init/kl_aux": "0.8"},
        {"step": "4", "loss/rate/init/kl_aux": "0.75"},
    ]
    _make_csv(tmp_path, rows)
    out = eval_q_aux_kl_trajectory(_ctx_with_run_dir(str(tmp_path)))
    assert out["q_aux_kl_trajectory_available"] is True
    assert out["q_aux_kl_trajectory_steps"] == [1, 2, 3, 4]
    assert out["q_aux_kl_trajectory_values"][-1] == pytest.approx(0.75)
    assert out["q_aux_kl_trajectory_peak"] == pytest.approx(0.8)
    assert out["q_aux_kl_trajectory_final"] == pytest.approx(0.75)
    assert out["q_aux_kl_trajectory_collapsed"] is False


def test_q_aux_kl_trajectory_flags_posterior_collapse(tmp_path: Path) -> None:
    rows = [
        {"step": "1", "loss/rate/init/kl_aux": "0.001"},
        {"step": "2", "loss/rate/init/kl_aux": "0.5"},
        {"step": "3", "loss/rate/init/kl_aux": "1e-4"},
    ]
    _make_csv(tmp_path, rows)
    out = eval_q_aux_kl_trajectory(_ctx_with_run_dir(str(tmp_path)))
    assert out["q_aux_kl_trajectory_collapsed"] is True


def test_q_aux_kl_trajectory_unavailable_without_run_dir() -> None:
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"))
    out = eval_q_aux_kl_trajectory(ctx)
    assert out["q_aux_kl_trajectory_available"] is False


def test_q_aux_kl_trajectory_unavailable_without_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    with open(csv_path, "w") as f:
        f.write("step,loss/total\n1,0.5\n2,0.4\n")
    out = eval_q_aux_kl_trajectory(_ctx_with_run_dir(str(tmp_path)))
    assert out["q_aux_kl_trajectory_available"] is False


# log_sigma_p2_collapse (metric #6)


def test_log_sigma_p2_collapse_registered() -> None:
    assert "log_sigma_p2_collapse" in METRIC_REGISTRY


def test_log_sigma_p2_collapse_unavailable_without_model() -> None:
    out = eval_log_sigma_p2_collapse(
        EvalContext(model=None, loader=None, device=torch.device("cpu"))
    )
    assert out["log_sigma_p2_collapse_available"] is False


@pytest.mark.slow
def test_log_sigma_p2_collapse_runs_on_smoke_model() -> None:
    """Snapshot shape + finiteness on a fresh init-centering smoke model."""
    from hydra_zen import instantiate

    from ddssm.experiment.stores import store
    from ddssm.experiment.registry import register_experiments

    register_experiments()
    cfg = store["experiment"]["experiment", "init_smoke_high_surface"]
    exp = instantiate(cfg)
    val_loader = exp.data.val_loader()
    ctx = EvalContext(
        model=exp.model.module,
        loader=val_loader,
        device=torch.device("cpu"),
        batch_transform=exp.data.batch_transform,
    )
    out = eval_log_sigma_p2_collapse(ctx, max_batches=1)
    assert out["log_sigma_p2_collapse_available"] is True
    assert len(out["log_sigma_p2_per_t_per_d"]) == len(out["log_sigma_p2_t_indices"])
    if out["log_sigma_p2_t_indices"]:
        assert len(out["log_sigma_p2_per_t_per_d"][0]) == int(exp.model.module.latent_dim)
    import math

    assert math.isfinite(out["log_sigma_p2_mean"])
    assert math.isfinite(out["log_sigma_p2_std"])
