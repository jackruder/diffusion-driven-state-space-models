"""Evaluation metrics and divergence detection for DDSSM training runs.

These helpers were factored out of the planned Hydra entry point so they can
be reused by any future training driver (hydra-zen, plain script, notebook).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch


def read_csv_column(csv_path: Path, col: str) -> list[float]:
    """Read a single numeric column from a CSV produced by ``CSVLogger``.

    Non-numeric or empty cells are skipped silently so this works on
    in-progress CSVs that may have header-only or partial rows.
    """
    values: list[float] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(col, "")
            if not raw:
                continue
            try:
                values.append(float(raw))
            except (ValueError, TypeError):
                continue
    return values


def find_train_csv(run_dir: Path) -> Path | None:
    """Locate ``train_metrics.csv`` inside a run directory."""
    expected = run_dir / "csv_logs" / "train_metrics.csv"
    if expected.is_file():
        return expected
    candidates = sorted(run_dir.rglob("train_metrics.csv"))
    return candidates[-1] if candidates else None


def pick_recon_column(csv_path: Path) -> str | None:
    """Return the most appropriate reconstruction-loss column name."""
    with open(csv_path) as f:
        headers = csv.DictReader(f).fieldnames or []
    for candidate in ("loss/distortion/rec", "loss/total"):
        if candidate in headers:
            return candidate
    for h in headers:
        if "recon" in h.lower() or "distortion" in h.lower():
            return h
    return None


def check_recon_divergence(
    run_dir: Path,
    spike_factor: float = 5.0,
    tail_fraction: float = 0.2,
    min_rows: int = 10,
) -> tuple[bool, str]:
    """Detect post-hoc divergence in a finished run.

    Compares the mean of the tail of the reconstruction-loss column against
    the median of the first half. Returns ``(diverged, reason)``. Useful as
    an Optuna pruning signal.

    The spike check is skipped when the first-half median is non-positive
    (e.g. for log-prob style losses); only the non-finite check applies in
    that case.
    """
    csv_path = find_train_csv(run_dir)
    if csv_path is None:
        return False, "no train_metrics.csv found"

    col = pick_recon_column(csv_path)
    if col is None:
        return False, "no recon/distortion column in CSV"

    values = read_csv_column(csv_path, col)
    if len(values) < min_rows:
        return False, f"only {len(values)} rows (<{min_rows}), skipping check"

    if any(not math.isfinite(v) for v in values):
        return True, f"non-finite {col} detected"

    n = len(values)
    half = n // 2
    tail_start = max(int(n * (1.0 - tail_fraction)), half)
    first_half_sorted = sorted(values[:half])
    median_first = first_half_sorted[len(first_half_sorted) // 2]
    mean_tail = sum(values[tail_start:]) / len(values[tail_start:])

    if median_first > 0 and mean_tail > spike_factor * median_first:
        return True, (
            f"{col} spike: tail mean={mean_tail:.4f} > "
            f"{spike_factor}x first-half median={median_first:.4f}"
        )

    return False, "ok"


def mae_metrics(
    pred_mean: torch.Tensor, y_future: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean absolute error, globally and per future timestep."""
    abs_err = (pred_mean - y_future).abs()
    return abs_err.mean(), abs_err.mean(dim=(0, 1))


def crps_sum_metrics(
    pred_samples: torch.Tensor, y_future: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """CRPS-sum at 19 quantile levels (0.05..0.95), globally and per timestep.

    ``pred_samples`` is expected to have shape ``(B, S, D, L2)`` and
    ``y_future`` shape ``(B, D, L2)``. Sums across the channel dimension are
    computed before quantilisation, matching the standard probabilistic
    forecasting convention.
    """
    levels = torch.arange(0.05, 1.0, 0.05, device=pred_samples.device)
    pred_sum = pred_samples.sum(dim=2)  # (B, S, L2)
    y_sum = y_future.sum(dim=1)  # (B, L2)
    qs = torch.quantile(pred_sum, levels, dim=1).permute(1, 2, 0)  # (B, L2, Q)
    indicator = (y_sum.unsqueeze(-1) < qs).float()
    crps = 2 * ((levels - indicator) * (y_sum.unsqueeze(-1) - qs)).mean(dim=-1)
    return crps.mean(), crps.mean(dim=0)


def crps_sum_latent_metrics(
    z_samples: torch.Tensor, z_gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """CRPS-sum on latent samples vs. ground-truth latents.

    Structurally identical to :func:`crps_sum_metrics`; the only
    difference is that ``z_samples`` and ``z_gt`` are latent-space
    tensors of shapes ``(B, S, d, L2)`` and ``(B, d, L2)``, summed
    across the latent dimension ``d``.

    Used by ``ddssm.eval.metrics.crps_sum_latent`` for the model-v2
    init-experiment headline metric on the latent path.
    """
    levels = torch.arange(0.05, 1.0, 0.05, device=z_samples.device)
    z_sum = z_samples.sum(dim=2)  # (B, S, L2)
    y_sum = z_gt.sum(dim=1)  # (B, L2)
    qs = torch.quantile(z_sum, levels, dim=1).permute(1, 2, 0)  # (B, L2, Q)
    indicator = (y_sum.unsqueeze(-1) < qs).float()
    crps = 2 * ((levels - indicator) * (y_sum.unsqueeze(-1) - qs)).mean(dim=-1)
    return crps.mean(), crps.mean(dim=0)
