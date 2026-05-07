"""Stateless metric functions and the registry that exposes them by name.

Each metric takes an :class:`EvalContext` plus its own keyword args
and returns a JSON-serialisable result (typically a ``dict``). The
runner walks the names listed in :class:`EvalSpec.metrics`, looks them
up here, and merges results into a single ``metrics.json``.

The model-level building blocks (``mae_metrics``, ``crps_sum_metrics``)
live in ``ddssm.eval_metrics`` and are reused here unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import math
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..eval_metrics import crps_sum_metrics, mae_metrics


@dataclass
class EvalContext:
    """Inputs available to every metric function.

    ``loader`` and ``model`` may be ``None`` for metrics that depend
    only on the training CSV log (e.g. tail-mean of ``loss/total``).
    """

    model: torch.nn.Module | None
    loader: DataLoader | None
    device: torch.device
    batch_transform: Callable[[dict, torch.device], dict] | None = None
    csv_path: str | None = None
    T_split: int | None = None
    num_samples: int = 1
    run_dir: str | None = None


MetricFn = Callable[[EvalContext], Dict[str, Any]]
METRIC_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator-style registration for new metrics."""

    def _wrap(fn: MetricFn) -> MetricFn:
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric {name!r} already registered")
        METRIC_REGISTRY[name] = fn
        return fn

    return _wrap


# ---------------------------------------------------------------------------
# Forecasting metrics: walk the loader, run model.forecast, accumulate.
# ---------------------------------------------------------------------------


def _iter_forecast_batches(ctx: EvalContext):
    """Yield ``(pred_samples, pred_mean, y_future)`` per batch.

    Splits each batch at ``ctx.T_split`` (required for these metrics).
    """
    if ctx.model is None or ctx.loader is None or ctx.T_split is None:
        raise ValueError(
            "Forecast metrics require model, loader, and T_split to be set "
            "on the EvalContext."
        )
    model = ctx.model
    device = ctx.device
    L1 = int(ctx.T_split)
    transform = ctx.batch_transform

    with torch.no_grad():
        for batch in ctx.loader:
            if transform is not None:
                batch = transform(batch, device)
            else:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            x_hist = batch["observed_data"][..., :L1]
            x_mask = batch["observation_mask"][..., :L1]
            past_time = batch["timepoints"][:, :L1]
            future_time = batch["timepoints"][:, L1:]
            y_future = batch["observed_data"][..., L1:]

            covariates = batch.get("covariates", None)
            past_cov = covariates[..., :L1] if covariates is not None else None
            future_cov = covariates[..., L1:] if covariates is not None else None
            static_cov = batch.get("static_covariates", None)

            out = model.forecast(
                x_hist=x_hist,
                x_mask=x_mask,
                past_time=past_time,
                future_time=future_time,
                past_covariates=past_cov,
                future_covariates=future_cov,
                static_covariates=static_cov,
                num_samples=int(ctx.num_samples),
            )
            yield out["pred_samples"], out["pred_mean"], y_future


@register_metric("energy_score")
def eval_energy_score(ctx: EvalContext) -> Dict[str, Any]:
    """Energy score (proper scoring rule) averaged over forecast batches.

    ES(F, y) = E[||X - y||] - 0.5 * E[||X - X'||]
    where X, X' are i.i.d. forecast samples and expectation is taken over S draws.
    (D, L2) dimensions are collapsed into a single vector before computing norms.
    """
    scores = []
    for pred_samples, _, y_future in _iter_forecast_batches(ctx):
        B, S, D, L2 = pred_samples.shape
        s_flat = pred_samples.reshape(B, S, -1)           # (B, S, D*L2)
        y_flat = y_future.reshape(B, -1).unsqueeze(1)     # (B, 1, D*L2)
        term1 = torch.norm(s_flat - y_flat, dim=-1).mean(dim=1)        # (B,)
        diff = s_flat.unsqueeze(2) - s_flat.unsqueeze(1)               # (B,S,S,D*L2)
        term2 = torch.norm(diff, dim=-1).mean(dim=(1, 2))              # (B,)
        scores.append(float((term1 - 0.5 * term2).mean().item()))
    if not scores:
        return {"energy_score": float("nan")}
    return {"energy_score": float(np.mean(scores))}


@register_metric("mae")
def eval_mae(ctx: EvalContext) -> Dict[str, Any]:
    """Mean absolute error of the forecast mean against the true future."""
    g_acc, t_acc = [], []
    for _, pred_mean, y_future in _iter_forecast_batches(ctx):
        g, t = mae_metrics(pred_mean, y_future)
        g_acc.append(float(g.item()))
        t_acc.append(t.detach().cpu().numpy())
    if not g_acc:
        return {"mae": float("nan"), "mae_per_t": []}
    return {
        "mae": float(np.mean(g_acc)),
        "mae_per_t": np.mean(np.stack(t_acc, axis=0), axis=0).tolist(),
    }


@register_metric("crps_sum")
def eval_crps_sum(ctx: EvalContext) -> Dict[str, Any]:
    """Sum-aggregated CRPS over forecast samples (channel-summed)."""
    g_acc, t_acc = [], []
    for pred_samples, _, y_future in _iter_forecast_batches(ctx):
        g, t = crps_sum_metrics(pred_samples, y_future)
        g_acc.append(float(g.item()))
        t_acc.append(t.detach().cpu().numpy())
    if not g_acc:
        return {"crps_sum": float("nan"), "crps_sum_per_t": []}
    return {
        "crps_sum": float(np.mean(g_acc)),
        "crps_sum_per_t": np.mean(np.stack(t_acc, axis=0), axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Reconstruction MSE: compares posterior reconstruction to observed values.
# ---------------------------------------------------------------------------


@register_metric("recon_mse")
def eval_recon_mse(ctx: EvalContext) -> Dict[str, Any]:
    """MSE between the decoded posterior mean and the observed sequence."""
    if ctx.model is None or ctx.loader is None:
        raise ValueError("recon_mse requires model and loader.")
    model, device = ctx.model, ctx.device
    transform = ctx.batch_transform
    sums, counts = 0.0, 0

    with torch.no_grad():
        for batch in ctx.loader:
            if transform is not None:
                batch = transform(batch, device)
            x = batch["observed_data"]
            mask = batch["observation_mask"]
            t = batch["timepoints"]
            _l, _r, _d, _m, stats = model(x, mask, t, train=False)
            zs = stats["zs"][:, 0]  # (B, d, T)

            from ..net_utils import time_embedding
            te = time_embedding(t, model.emb_time_dim, device=device)

            T = x.shape[-1]
            recon = torch.zeros_like(x)
            for tt in range(T):
                t_idx = torch.full((x.shape[0],), tt, device=device, dtype=torch.long)
                z_hist = zs[..., : tt + 1]
                if z_hist.shape[-1] > model.j:
                    z_hist = z_hist[..., -model.j :]
                mu_x, _ = model.decoder(z_hist, te, t_idx)
                recon[..., tt] = mu_x

            err = (recon - x) ** 2
            if mask is not None:
                err = err * mask
                counts += int(mask.sum().item())
            else:
                counts += int(err.numel())
            sums += float(err.sum().item())

    return {"recon_mse": sums / max(counts, 1)}


# ---------------------------------------------------------------------------
# Bimodal JSD: per-sample one-step JSD against the analytic bimodal truth.
#
# Specific to the synthetic ``bimodal`` mode (z_t = 0.9 z_{t-1} + 4 s_t,
# s_t ∈ {-1, +1}, x_t = z_t + N(0, 0.2)). The analytic conditional one-step
# distribution centred at -a*x_prev is a 50/50 mixture of N(-step_size, sigma)
# and N(+step_size, sigma).
# ---------------------------------------------------------------------------

_JSD_EPS = 1e-12


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _jsd_discrete(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, _JSD_EPS, None); p = p / p.sum()
    q = np.clip(q, _JSD_EPS, None); q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _hist_mass(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(vals, bins=edges, density=False)
    h = h.astype(np.float64)
    return np.ones_like(h) / h.size if h.sum() <= 0 else h / h.sum()


def _bimodal_truth_mass(centers: np.ndarray, x_prev: float, *,
                        a: float, step_size: float, sigma: float,
                        center_coef: float) -> np.ndarray:
    """Discretised analytic one-step truth, centred at ``-center_coef * x_prev``."""
    shift = (a - center_coef) * x_prev
    pdf = 0.5 * _normal_pdf(centers, shift - step_size, sigma) \
        + 0.5 * _normal_pdf(centers, shift + step_size, sigma)
    pdf = np.clip(pdf, _JSD_EPS, None)
    return pdf / pdf.sum()


@register_metric("bimodal_jsd")
def eval_bimodal_jsd(
    ctx: EvalContext,
    *,
    npz_path: str | None = None,
    edges_min: float = -10.0,
    edges_max: float = 10.0,
    n_bins: int = 300,
    step_size: float = 4.0,
    sigma: float = 0.2,
    a: float = 0.9,
    center_coef: float = 0.9,
) -> Dict[str, Any]:
    """Per-sample one-step Jensen–Shannon divergence vs analytic bimodal truth.

    Specific to ``mode="bimodal"`` synthetic data. For each batch element the
    metric takes the FIRST future timestep's forecast samples
    (``pred_samples[:, :, 0, 0]``), centres them at ``-center_coef * x_prev``,
    bins into a histogram, and compares against the analytic 50/50 mixture
    of Gaussians at the same centring.

    Note: only the t+1 horizon is scored regardless of ``L2``. To run a strict
    one-step analysis (matching the legacy ``bimodal_jsd.py`` script), override
    ``T_split=T-1`` in the EvalSpec.

    Args:
        npz_path: If set (relative or absolute), write per-sample data
            (x_prev, forecast samples, model_mass, truth_mass) to this NPZ
            for downstream comparison plotting (``plot_bimodal_compare.py``).
        edges_min, edges_max, n_bins: Histogram bin definition.
        step_size, sigma, a: Bimodal DGP constants (defaults match
            ``SyntheticDataset(mode="bimodal")``).
        center_coef: Centring coefficient. ``0.9`` removes the AR drift exactly.

    Returns:
        ``bimodal_jsd_mean / std / sem / median`` plus
        ``bimodal_jsd_best_idx / worst_idx / median_idx_1 / median_idx_2``.
    """
    if ctx.model is None or ctx.loader is None or ctx.T_split is None:
        raise ValueError("bimodal_jsd requires model, loader, and T_split.")
    model, device = ctx.model, ctx.device
    L1 = int(ctx.T_split)
    transform = ctx.batch_transform

    edges = np.linspace(edges_min, edges_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    jsds: list[float] = []
    x_prevs: list[float] = []
    sample_buf: list[np.ndarray] = []
    model_mass_buf: list[np.ndarray] = []
    truth_mass_buf: list[np.ndarray] = []

    with torch.no_grad():
        for batch in ctx.loader:
            if transform is not None:
                batch = transform(batch, device)

            obs = batch["observed_data"]
            mask = batch["observation_mask"]
            past_time = batch["timepoints"][:, :L1]
            future_time = batch["timepoints"][:, L1:]

            out = model.forecast(
                x_hist=obs[..., :L1],
                x_mask=mask[..., :L1],
                past_time=past_time,
                future_time=future_time,
                num_samples=int(ctx.num_samples),
            )
            pred_samples = out["pred_samples"]  # (B, S, D, L2)
            B = pred_samples.shape[0]
            x_prev = obs[:, 0, L1 - 1].detach().cpu().numpy()
            xhat = pred_samples[:, :, 0, 0].detach().cpu().numpy()  # (B, S)

            for b in range(B):
                ctr = xhat[b] - center_coef * x_prev[b]
                p = _hist_mass(ctr, edges)
                q = _bimodal_truth_mass(centers, float(x_prev[b]),
                                        a=a, step_size=step_size, sigma=sigma,
                                        center_coef=center_coef)
                jsds.append(_jsd_discrete(p, q))
                x_prevs.append(float(x_prev[b]))
                sample_buf.append(xhat[b].astype(np.float32, copy=True))
                model_mass_buf.append(p.astype(np.float32, copy=True))
                truth_mass_buf.append(q.astype(np.float32, copy=True))

    jsd_arr = np.asarray(jsds, dtype=np.float64)
    n = int(jsd_arr.size)
    if n == 0:
        return {"bimodal_jsd_mean": float("nan"), "bimodal_jsd_n": 0}

    order = np.argsort(jsd_arr)
    if n == 1:
        median_idx_1 = median_idx_2 = 0
    elif n % 2 == 0:
        median_idx_1, median_idx_2 = int(order[n // 2 - 1]), int(order[n // 2])
    else:
        median_idx_1 = int(order[n // 2])
        median_idx_2 = int(order[min(n // 2 + 1, n - 1)])

    if npz_path is not None:
        out_path = npz_path
        if not os.path.isabs(out_path) and ctx.run_dir is not None:
            out_path = os.path.join(ctx.run_dir, out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez_compressed(
            out_path,
            sample_idx=np.arange(n, dtype=np.int64),
            x_prev=np.asarray(x_prevs, dtype=np.float32),
            xhat_samples=np.stack(sample_buf, axis=0),
            edges=edges.astype(np.float32),
            centers=centers.astype(np.float32),
            model_mass=np.stack(model_mass_buf, axis=0),
            truth_mass=np.stack(truth_mass_buf, axis=0),
            center_coef=np.float32(center_coef),
            step_size=np.float32(step_size),
            sigma=np.float32(sigma),
            a=np.float32(a),
        )

    return {
        "bimodal_jsd_n": n,
        "bimodal_jsd_mean": float(jsd_arr.mean()),
        "bimodal_jsd_std": float(jsd_arr.std()),
        "bimodal_jsd_sem": float(jsd_arr.std() / math.sqrt(n)),
        "bimodal_jsd_median": float(np.median(jsd_arr)),
        "bimodal_jsd_best_idx": int(order[0]),
        "bimodal_jsd_worst_idx": int(order[-1]),
        "bimodal_jsd_median_idx_1": median_idx_1,
        "bimodal_jsd_median_idx_2": median_idx_2,
    }


# ---------------------------------------------------------------------------
# CSV-derived metrics: cheap post-hoc summaries of the training log.
# ---------------------------------------------------------------------------


@register_metric("loss_tail")
def eval_loss_tail(ctx: EvalContext, *, column: str = "loss/total", tail_frac: float = 0.1) -> Dict[str, Any]:
    """Mean of the final ``tail_frac`` of values in a CSV column."""
    if not ctx.csv_path:
        return {column.replace("/", "_") + "_tail": float("nan")}
    import csv as _csv

    values: list[float] = []
    try:
        with open(ctx.csv_path, "r", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                raw = row.get(column, "")
                if raw in ("", None):
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    values.append(v)
    except OSError:
        return {column.replace("/", "_") + "_tail": float("nan")}
    if not values:
        return {column.replace("/", "_") + "_tail": float("nan")}
    n = max(1, int(len(values) * float(tail_frac)))
    return {column.replace("/", "_") + "_tail": float(sum(values[-n:]) / n)}
