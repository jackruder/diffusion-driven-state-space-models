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
    pred_mean: torch.Tensor,
    y_future: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean absolute error, globally and per future timestep.

    ``mask`` (``(B, D, L2)``, 1 = observed) restricts both reductions to
    observed target entries; missing/imputed entries would otherwise be
    scored as if real. Timesteps with no observed entries report 0.
    """
    abs_err = (pred_mean - y_future).abs()
    if mask is None:
        return abs_err.mean(), abs_err.mean(dim=(0, 1))
    m = mask.to(abs_err.dtype)
    abs_err = abs_err * m
    global_mae = abs_err.sum() / m.sum().clamp_min(1.0)
    per_t = abs_err.sum(dim=(0, 1)) / m.sum(dim=(0, 1)).clamp_min(1.0)
    return global_mae, per_t


def rmse_metrics(
    pred_mean: torch.Tensor,
    y_future: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Root mean squared error, globally and per future timestep.

    The global value is the true RMSE (sqrt of the overall MSE); the per-t
    vector is the sqrt of each timestep's MSE (so it does NOT linearly average
    back to the global value — sqrt is nonlinear).

    ``mask`` (``(B, D, L2)``, 1 = observed) restricts both reductions to
    observed target entries. Timesteps with no observed entries report 0.
    """
    sq_err = (pred_mean - y_future) ** 2  # (B, D, L2)
    if mask is None:
        return sq_err.mean().sqrt(), sq_err.mean(dim=(0, 1)).sqrt()
    m = mask.to(sq_err.dtype)
    sq_err = sq_err * m
    global_rmse = (sq_err.sum() / m.sum().clamp_min(1.0)).sqrt()
    per_t = (sq_err.sum(dim=(0, 1)) / m.sum(dim=(0, 1)).clamp_min(1.0)).sqrt()
    return global_rmse, per_t


def _crps_sum_pinball(
    pred_sum: torch.Tensor,
    y_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-batch-element pinball numerator and ND denominator.

    Args:
        pred_sum: Channel-summed forecast samples, shape ``(B, S, L2)``.
        y_sum: Channel-summed targets, shape ``(B, L2)``.

    Returns:
        A pair ``(ql_sum, denom)`` where:

        * ``ql_sum`` — pinball numerator, shape ``(B, L2)``.  Each entry is
          ``Σ_τ 2·QL(τ)`` averaged over the 19 quantile levels, summed across
          time within each batch element when the caller wants cross-batch
          pooling, or kept per-``(B, L2)`` for ``per_t`` reporting.
        * ``denom`` — ND denominator, shape ``(B, 1)``:
          ``Σ_t |Σ_d y_{b,d,t}|`` per batch element, clamped away from zero.
    """
    levels = torch.arange(0.05, 1.0, 0.05, device=pred_sum.device)
    qs = torch.quantile(pred_sum, levels, dim=1).permute(1, 2, 0)  # (B, L2, Q)
    indicator = (y_sum.unsqueeze(-1) < qs).float()
    # Mean over the 19 levels of 2·QL(τ) (CSDI calc_quantile_CRPS), per (B, L2).
    ql = 2 * ((levels - indicator) * (y_sum.unsqueeze(-1) - qs)).sum(dim=-1)
    ql = ql / levels.numel()  # (B, L2)
    denom = y_sum.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)  # (B, 1)
    return ql, denom


def crps_sum_components(
    pred_samples: torch.Tensor,
    y_future: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the raw numerator and denominator for ratio-of-means ND.

    This is the building block for cross-batch pooling in
    :func:`crps_sum_metrics` and :mod:`ddssm.eval.metrics`.  Callers that
    span multiple batches should **accumulate** the two tensors separately
    across batches and compute ``num.sum() / denom.sum()`` once at the end,
    which yields the true ratio-of-means ND.

    Args:
        pred_samples: ``(B, S, D, L2)`` forecast sample tensor.
        y_future: ``(B, D, L2)`` target tensor.
        mask: Optional ``(B, D, L2)`` observation mask (1 = observed).
            Masked entries are zeroed in both target and samples before the
            channel sum, so they contribute neither to the pinball numerator
            nor to the ND denominator (CSDI ``eval_points`` convention).

    Returns:
        ``(num, denom)`` where

        * ``num`` has shape ``(B, L2)``: the per-``(batch, time)`` pinball
          loss averaged over the 19 quantile levels.
        * ``denom`` has shape ``(B, 1)``: ``Σ_t |Σ_d y_{b,d,t}|`` per batch
          element, clamped to ``≥ 1e-8``.
    """
    if mask is not None:
        m = mask.to(y_future.dtype)
        pred_samples = pred_samples * m.unsqueeze(1)
        y_future = y_future * m
    pred_sum = pred_samples.sum(dim=2)  # (B, S, L2)
    y_sum = y_future.sum(dim=1)  # (B, L2)
    return _crps_sum_pinball(pred_sum, y_sum)


def crps_sum_metrics(
    pred_samples: torch.Tensor,
    y_future: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ND-normalized CRPS-sum at 19 quantile levels (0.05..0.95).

    ``pred_samples`` is expected to have shape ``(B, S, D, L2)`` and
    ``y_future`` shape ``(B, D, L2)``. Sums across the channel dimension are
    computed before quantilisation, matching the standard probabilistic
    forecasting convention.

    ``mask`` (``(B, D, L2)``, 1 = observed) zeroes missing entries in both
    the target and the forecast samples before the channel sum (the CSDI
    ``eval_points`` convention), so unobserved entries contribute neither
    to the pinball loss nor to the ND denominator.

    Per batch element the pinball loss is **averaged over the 19 quantile
    levels** (``Σ_τ 2·QL(τ) / 19``), matching CSDI's ``calc_quantile_CRPS`` /
    the published CRPS-sum convention — this is what every baseline table
    reports. (A Δτ=0.05 Riemann sum, which only spans [0.05,0.95], reads ≈5%
    lower and is NOT comparable to published numbers.)

    The global ND is the **ratio-of-means**: total pinball numerator across
    all batch elements and timesteps divided by the total ND denominator
    ``Σ_{b,t} |Σ_d y_{b,d,t}|``.  This matches the GluonTS / CSDI published
    convention and avoids overweighting low-magnitude windows.  The
    ``per_t`` vector is the per-timestep pinball sum (across the batch)
    divided by the **global** denominator, so ``per_t.sum() == global``.

    For cross-batch pooling (accumulating over a data loader) use
    :func:`crps_sum_components` to obtain ``(num, denom)`` per batch, then
    compute ``num_all.sum() / denom_all.sum()`` at the end — **do not**
    average per-batch globals, which would reintroduce mean-of-ratios bias.
    """
    num, denom = crps_sum_components(pred_samples, y_future, mask=mask)
    # num: (B, L2), denom: (B, 1) — ratio-of-means: pool across B and t.
    total_denom = denom.sum().clamp_min(1e-8)
    global_nd = num.sum() / total_denom
    # per_t: pinball sum across B divided by the global denominator so
    # per_t.sum() == global_nd.
    per_t = num.sum(dim=0) / total_denom
    return global_nd, per_t


def crps_sum_latent_metrics(
    z_samples: torch.Tensor, z_gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """CRPS-sum on latent samples vs. ground-truth latents.

    Structurally identical to :func:`crps_sum_metrics`; the only
    difference is that ``z_samples`` and ``z_gt`` are latent-space
    tensors of shapes ``(B, S, d, L2)`` and ``(B, d, L2)``, summed
    across the latent dimension ``d``.

    The global ND uses the ratio-of-means convention (see
    :func:`crps_sum_metrics`): total pinball numerator divided by total
    ND denominator across all batch elements and timesteps.  The ``per_t``
    vector is each timestep's numerator (summed over B) divided by the
    global denominator, so ``per_t.sum() == global``.

    Used by ``ddssm.eval.metrics.eval_crps_sum_latent`` for the model-v2
    init-experiment headline metric on the latent path.
    """
    z_sum = z_samples.sum(dim=2)  # (B, S, L2)
    y_sum = z_gt.sum(dim=1)  # (B, L2)
    num, denom = _crps_sum_pinball(z_sum, y_sum)
    total_denom = denom.sum().clamp_min(1e-8)
    global_nd = num.sum() / total_denom
    per_t = num.sum(dim=0) / total_denom
    return global_nd, per_t
