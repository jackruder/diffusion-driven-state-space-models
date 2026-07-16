"""Stateless metric functions and the registry that exposes them by name.

Each metric takes an :class:`EvalContext` plus its own keyword args
and returns a JSON-serialisable result (typically a ``dict``). The
runner walks the names listed in :class:`EvalSpec.metrics`, looks them
up here, and merges results into a single ``metrics.json``.

The model-level building blocks (``mae_metrics``, ``crps_sum_metrics``)
live in ``ddssm.eval.eval_metrics`` and are reused here unchanged.
"""

from __future__ import annotations

import os
import csv
import math
from typing import Any
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ddssm.eval.eval_metrics import (
    crps_sum_components,
    crps_sum_metrics,
    mae_metrics,
    rmse_metrics,
)
from ddssm.adapters.base import MetricNotSupported


@dataclass
class EvalContext:
    """Inputs available to every metric function.

    ``loader`` and ``model`` may be ``None`` for metrics that depend
    only on the training CSV log (e.g. tail-mean of ``loss/total``).

    ``model`` is the :class:`~ddssm.adapters.base.ModelAdapter`, not the raw
    ``nn.Module``. Shared metrics use the adapter surface directly
    (``model.forecast`` / ``model.log_prob``); family-internal (DDSSM-only)
    metrics reach the owned module via :meth:`require_module`.
    """

    model: Any | None
    loader: DataLoader | None
    device: torch.device
    batch_transform: Callable[[dict, torch.device], dict] | None = None
    csv_path: str | None = None
    T_split: int | None = None
    num_samples: int = 1
    run_dir: str | None = None
    # Per-series z-score stats ``(D,)`` for de-normalizing forecasts back to the
    # original data scale before obs-space metrics (CSDI's calc_quantile_CRPS
    # de-normalizes; per-series scaling makes the channel-sum non-comparable
    # otherwise). ``None`` ⇒ data was not normalized (e.g. synthetic) → no-op.
    means: torch.Tensor | None = None
    stds: torch.Tensor | None = None

    def require_module(self, cls: type) -> torch.nn.Module:
        """Return the adapter's owned module iff it is a ``cls``; else raise.

        The single gating prelude for family-internal (DDSSM-only) metrics: it
        replaces both the raw ``.module`` access and any capability metadata.
        A non-matching family raises :class:`MetricNotSupported`, which the eval
        runner catches to skip + omit the metric -- NOT ``AttributeError``,
        which would mask real bugs deeper in a metric. Callers pass ``cls``
        (lazy-imported at the call site) so this helper stays cycle-free.
        """
        module = self.model.module
        if not isinstance(module, cls):
            raise MetricNotSupported(
                f"{type(self.model).__name__} does not provide a {cls.__name__} module"
            )
        return module


MetricFn = Callable[[EvalContext], dict[str, Any]]
METRIC_REGISTRY: dict[str, MetricFn] = {}


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
    """Yield ``(pred_samples, pred_mean, y_future, y_mask)`` per batch.

    Splits each batch at ``ctx.T_split`` (required for these metrics).
    ``y_mask`` is the future slice of the observation mask (1 = observed);
    consumers must reduce over observed entries only, else missing/imputed
    targets are scored as if real.
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

    # De-normalize forecasts back to the original data scale (CSDI convention),
    # so obs-space metrics are comparable to published tables. ``(D,)`` per-series
    # stats broadcast over (B[,S],D,L2).
    denorm = ctx.means is not None and ctx.stds is not None
    if denorm:
        mean_d = ctx.means.to(device).reshape(1, -1, 1)  # (1, D, 1)
        std_d = ctx.stds.to(device).reshape(1, -1, 1)

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
            y_mask = batch["observation_mask"][..., L1:]

            covariates = batch.get("covariates")
            past_cov = covariates[..., :L1] if covariates is not None else None
            future_cov = covariates[..., L1:] if covariates is not None else None
            static_cov = batch.get("static_covariates")

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
            pred_samples, pred_mean = out["pred_samples"], out["pred_mean"]
            if denorm:
                pred_samples = pred_samples * std_d.unsqueeze(1) + mean_d.unsqueeze(1)
                pred_mean = pred_mean * std_d + mean_d
                y_future = y_future * std_d + mean_d
            yield pred_samples, pred_mean, y_future, y_mask


@register_metric("energy_score")
def eval_energy_score(ctx: EvalContext) -> dict[str, Any]:
    """Energy score (proper scoring rule) averaged over forecast batches.

    ES(F, y) = E[||X - y||] - 0.5 * E[||X - X'||]
    where X, X' are i.i.d. forecast samples and expectation is taken over S draws.
    (D, L2) dimensions are collapsed into a single vector before computing norms.

    Missing target entries are zeroed in both the samples and the target
    (projection onto the observed coordinates), so they contribute to
    neither norm — the score is the energy score of the observed marginal.
    """
    scores = []
    for pred_samples, _, y_future, y_mask in _iter_forecast_batches(ctx):
        m = y_mask.to(pred_samples.dtype)
        pred_samples = pred_samples * m.unsqueeze(1)
        y_future = y_future * m
        B, S, D, L2 = pred_samples.shape
        s_flat = pred_samples.reshape(B, S, -1)  # (B, S, D*L2)
        y_flat = y_future.reshape(B, -1).unsqueeze(1)  # (B, 1, D*L2)
        term1 = torch.norm(s_flat - y_flat, dim=-1).mean(dim=1)  # (B,)
        diff = s_flat.unsqueeze(2) - s_flat.unsqueeze(1)  # (B,S,S,D*L2)
        pair = torch.norm(diff, dim=-1)  # (B,S,S)
        # Unbiased U-statistic for E||X-X'||: exclude the zero diagonal
        # (the i==i pairs). Averaging over the full S×S matrix would
        # underestimate by (S-1)/S, biasing the score high by a margin
        # that shrinks with S. The diagonal is zero, so summing the whole
        # matrix and dividing by S(S-1) drops it cleanly.
        if S > 1:
            term2 = pair.sum(dim=(1, 2)) / (S * (S - 1))  # (B,)
        else:
            term2 = torch.zeros_like(term1)
        scores.append(float((term1 - 0.5 * term2).mean().item()))
    if not scores:
        return {"energy_score": float("nan")}
    return {"energy_score": float(np.mean(scores))}


def _masked_weighted_reduce(
    g_acc: list[float],
    t_acc: list[np.ndarray],
    w_g: list[float],
    w_t: list[np.ndarray],
    *,
    square: bool = False,
) -> tuple[float, list[float]] | None:
    """Pool per-batch (global, per_t) values weighted by observed counts.

    Plain ``np.mean`` over batches would weight a nearly-all-missing batch
    the same as a fully-observed one; weighting by mask counts recovers the
    exact pooled mean over observed entries. ``square=True`` pools on the
    squared scale (for RMSE, where sqrt is nonlinear).
    """
    total_w = float(np.sum(w_g))
    if total_w <= 0:
        return None
    g_arr = np.asarray(g_acc, dtype=np.float64)
    t_arr = np.stack(t_acc, axis=0).astype(np.float64)
    w_t_arr = np.stack(w_t, axis=0).astype(np.float64)
    if square:
        g_arr = g_arr**2
        t_arr = t_arr**2
    g = float(np.average(g_arr, weights=np.asarray(w_g, dtype=np.float64)))
    per_t = (t_arr * w_t_arr).sum(axis=0) / np.maximum(w_t_arr.sum(axis=0), 1.0)
    if square:
        g = math.sqrt(g)
        per_t = np.sqrt(per_t)
    return g, per_t.tolist()


@register_metric("mae")
def eval_mae(ctx: EvalContext) -> dict[str, Any]:
    """Mean absolute error of the forecast mean against the true future.

    Reduced over observed target entries only (per the observation mask).
    """
    g_acc, t_acc, w_g, w_t = [], [], [], []
    for _, pred_mean, y_future, y_mask in _iter_forecast_batches(ctx):
        g, t = mae_metrics(pred_mean, y_future, mask=y_mask)
        g_acc.append(float(g.item()))
        t_acc.append(t.detach().cpu().numpy())
        w_g.append(float(y_mask.sum().item()))
        w_t.append(y_mask.sum(dim=(0, 1)).detach().cpu().numpy())
    reduced = _masked_weighted_reduce(g_acc, t_acc, w_g, w_t) if g_acc else None
    if reduced is None:
        return {"mae": float("nan"), "mae_per_t": []}
    return {"mae": reduced[0], "mae_per_t": reduced[1]}


@register_metric("rmse")
def eval_rmse(ctx: EvalContext) -> dict[str, Any]:
    """Root mean squared error of the forecast mean against the true future.

    Reduced over observed target entries only (per the observation mask).
    """
    g_acc, t_acc, w_g, w_t = [], [], [], []
    for _, pred_mean, y_future, y_mask in _iter_forecast_batches(ctx):
        g, t = rmse_metrics(pred_mean, y_future, mask=y_mask)
        g_acc.append(float(g.item()))
        t_acc.append(t.detach().cpu().numpy())
        w_g.append(float(y_mask.sum().item()))
        w_t.append(y_mask.sum(dim=(0, 1)).detach().cpu().numpy())
    reduced = (
        _masked_weighted_reduce(g_acc, t_acc, w_g, w_t, square=True)
        if g_acc
        else None
    )
    if reduced is None:
        return {"rmse": float("nan"), "rmse_per_t": []}
    return {"rmse": reduced[0], "rmse_per_t": reduced[1]}


@register_metric("crps_sum")
def eval_crps_sum(ctx: EvalContext) -> dict[str, Any]:
    """Sum-aggregated CRPS over forecast samples (channel-summed).

    Missing target entries are zeroed in both target and samples before the
    channel sum (CSDI ``eval_points`` convention).

    The global ND is the **ratio-of-means**: the total pinball numerator
    accumulated across all batches and timesteps divided by the total ND
    denominator ``Σ_{b,t} |Σ_d y_{b,d,t}|``.  Numerator and denominator are
    accumulated separately across batches and divided once, which avoids the
    mean-of-ratios bias that would overweight low-magnitude windows.

    ``crps_sum_per_t`` is each timestep's pinball sum (across all batch
    elements) divided by the same global denominator, so
    ``sum(crps_sum_per_t) == crps_sum``.
    """
    num_acc: list[np.ndarray] = []  # (B, L2) per batch
    denom_acc: list[float] = []  # scalar per batch (sum of (B, 1))
    for pred_samples, _, y_future, y_mask in _iter_forecast_batches(ctx):
        num, denom = crps_sum_components(pred_samples, y_future, mask=y_mask)
        num_acc.append(num.detach().cpu().numpy())  # (B, L2)
        denom_acc.append(float(denom.sum().item()))
    if not num_acc:
        return {"crps_sum": float("nan"), "crps_sum_per_t": []}
    # Ratio-of-means: pool numerator and denominator separately.
    total_denom = max(float(np.sum(denom_acc)), 1e-8)
    # Stack over batches → (N_batches*B, L2); sum across all elements, then
    # divide once by the global denominator.
    num_all = np.concatenate(num_acc, axis=0)  # (total_B, L2)
    per_t = (num_all.sum(axis=0) / total_denom).tolist()
    global_nd = float(num_all.sum() / total_denom)
    return {"crps_sum": global_nd, "crps_sum_per_t": per_t}


# ---------------------------------------------------------------------------
# Analytic LGSSM optimum: the Kalman-filter predictive forecast NLL — the floor a
# trained model can approach but not beat on the easy (lgssm) cell.
# ---------------------------------------------------------------------------

# Generative params of the ``lgssm`` synthetic mode (data/synthetic.py): a scalar
# AR(1) latent z_t = a·z_{t-1} + N(0, q) observed as x_t = z_t + N(0, r), applied
# independently per channel from a DETERMINISTIC start z_0 = 0. These MUST track the
# generator; test_kalman_forecast_nll guards against drift by checking the empirical
# NLL against the analytic Gaussian entropy floor.
_LGSSM_A = 0.9
_LGSSM_PROC_SIGMA = 0.1
_LGSSM_OBS_SIGMA = 0.1


@register_metric("kalman_forecast_nll")
def eval_kalman_forecast_nll(
    ctx: EvalContext,
    *,
    a: float = _LGSSM_A,
    proc_sigma: float = _LGSSM_PROC_SIGMA,
    obs_sigma: float = _LGSSM_OBS_SIGMA,
) -> Dict[str, Any]:
    """Bayes-optimal (Kalman) per-step forecast NLL on LGSSM data.

    Conditions on the first ``ctx.T_split`` observations, propagates the EXACT
    Gaussian predictive forward over the horizon, and scores ``-log N(x_t; μ_t, V_t)``
    at the TRUE future observations — the optimal-forecaster reference for the
    ``lgssm`` cell (a number the model approaches from above). Data-only (no model);
    assumes the un-normalized scalar-AR(1) process applied per channel.
    """
    if ctx.loader is None or ctx.T_split is None:
        raise ValueError(
            "kalman_forecast_nll requires loader and T_split on the EvalContext."
        )
    device = ctx.device
    transform = ctx.batch_transform
    L1 = int(ctx.T_split)
    a = float(a)
    q = float(proc_sigma) ** 2
    r = float(obs_sigma) ** 2

    per_t_acc: list[np.ndarray] = []
    weights: list[int] = []
    with torch.no_grad():
        for batch in ctx.loader:
            if transform is not None:
                batch = transform(batch, device)
            else:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            x = batch["observed_data"]            # (B, D, T)
            B, D, T = x.shape
            if T <= L1:
                continue
            H = T - L1

            # Kalman filter over the history. z_0 = 0 is known, so the variance
            # recursion is data-independent (a scalar); only the mean depends on x.
            z_filt = torch.zeros(B, D, device=device, dtype=x.dtype)
            P_f = 0.0
            for t in range(1, L1):
                P_pred = a * a * P_f + q
                K = P_pred / (P_pred + r)
                z_filt = a * z_filt + K * (x[..., t] - a * z_filt)
                P_f = (1.0 - K) * P_pred

            # Forecast: propagate the filtered posterior forward, no updates.
            zhat = a * z_filt                     # latent predictive mean at t=L1
            P_lat = a * a * P_f + q               # latent predictive var at t=L1
            per_t = torch.empty(H, device=device, dtype=x.dtype)
            for h in range(H):
                Vx = P_lat + r                    # obs predictive var at t=L1+h
                resid = x[..., L1 + h] - zhat
                per_t[h] = (0.5 * (resid * resid / Vx + math.log(2.0 * math.pi * Vx))).mean()
                zhat = a * zhat
                P_lat = a * a * P_lat + q
            per_t_acc.append(per_t.detach().cpu().numpy())
            weights.append(B)

    if not per_t_acc:
        return {"kalman_forecast_nll": float("nan"), "kalman_forecast_nll_per_t": []}
    w = np.asarray(weights, dtype=float)
    per_t = np.average(np.stack(per_t_acc, axis=0), axis=0, weights=w)
    return {
        "kalman_forecast_nll": float(per_t.mean()),
        "kalman_forecast_nll_per_t": per_t.tolist(),
    }


# ---------------------------------------------------------------------------
# Reconstruction MSE: compares posterior reconstruction to observed values.
# ---------------------------------------------------------------------------


@register_metric("recon_mse")
def eval_recon_mse(ctx: EvalContext) -> dict[str, Any]:
    """MSE between the decoded posterior mean and the observed sequence."""
    from ddssm.model.dssd import DDSSM_base

    if ctx.model is None or ctx.loader is None:
        raise ValueError("recon_mse requires model and loader.")
    model = ctx.require_module(DDSSM_base)
    device = ctx.device
    transform = ctx.batch_transform
    sums, counts = 0.0, 0

    with torch.no_grad():
        for batch in ctx.loader:
            if transform is not None:
                batch = transform(batch, device)
            x = batch["observed_data"]
            mask = batch["observation_mask"]
            t = batch["timepoints"]
            _components, _metrics, stats = model(x, mask, t, train=False)
            zs = stats["zs"][:, 0]  # (B, d, T)

            from ddssm.nn.net_utils import time_embedding

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
    p = np.clip(p, _JSD_EPS, None)
    p = p / p.sum()
    q = np.clip(q, _JSD_EPS, None)
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _hist_mass(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(vals, bins=edges, density=False)
    h = h.astype(np.float64)
    return np.ones_like(h) / h.size if h.sum() <= 0 else h / h.sum()


def _bimodal_truth_mass(
    centers: np.ndarray,
    x_prev: float,
    *,
    a: float,
    step_size: float,
    sigma: float,
    center_coef: float,
) -> np.ndarray:
    """Discretised analytic one-step truth, centred at ``-center_coef * x_prev``."""
    shift = (a - center_coef) * x_prev
    pdf = 0.5 * _normal_pdf(centers, shift - step_size, sigma) + 0.5 * _normal_pdf(
        centers, shift + step_size, sigma
    )
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
) -> dict[str, Any]:
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
                q = _bimodal_truth_mass(
                    centers,
                    float(x_prev[b]),
                    a=a,
                    step_size=step_size,
                    sigma=sigma,
                    center_coef=center_coef,
                )
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
# Obs-space sliced JSD: model's decoded forecast one-step samples vs. K resamples
# of the DGP's analytic transition kernel lifted through the SAME random MLP the
# data generator used. Specific to ``nonlinear-bimodal-lift-mv`` — the only mode
# that exposes both ``gt_latent`` and ``gt_signs``. Frame-invariant successor of
# the (purged) latent-frame JSD.
# ---------------------------------------------------------------------------


@register_metric("obs_space_jsd")
def eval_obs_space_jsd(
    ctx: EvalContext,
    *,
    origins: tuple[int, ...] = (8, 16, 24),
    n_random_projections: int = 32,
    n_bins: int = 60,
    max_batches: int = 4,
    proj_seed: int = 42,
) -> dict[str, Any]:
    """Sliced JSD between decoded forecast samples and DGP-lifted samples.

    For each batch element, at each forecast origin ``t`` in ``origins``:

    1. Model side: call ``model.forecast(x_hist=obs[..., :t], ...,
       num_samples=S)`` with ``S = ctx.num_samples`` and take the first
       future step ``pred_samples[:, :, :, 0]`` → ``(B, S, D)``.
    2. Truth side: from the stored ``gt_latent[:, :, t-1]`` and
       ``gt_signs[:, :, t-1]``, draw ``K = S`` fresh sign transitions and
       process-noise samples to produce ``z_t``, then lift through the
       deterministic DGP MLP + ``σ_x`` obs-noise → ``(B, K=S, D)``.
    3. Project both sample sets onto ``n_random_projections`` random unit
       vectors + ``D`` axis-aligned directions, bin each 1D projection into
       ``n_bins`` shared bins (edges from combined min/max), and compute
       the discrete JSD between the two histograms.

    Aggregation is mean-over-batch then mean-over-directions. Reports
    ``obs_space_jsd_mean`` (all origins × directions × batch),
    ``obs_space_jsd_per_origin`` (dict), and ``obs_space_jsd_per_dim``
    (list of D floats aggregating only the axis-aligned directions).

    Skipped cleanly (returns ``{obs_space_jsd_available: False, reason}``)
    unless the loader's dataset is ``nonlinear-bimodal-lift-mv`` and both
    ``gt_latent`` and ``gt_signs`` are present.
    """
    from ddssm.data.synthetic import (
        NLBL_DELTA,
        NLBL_MV_OBS_D,
        NLBL_MV_SIGN_PERSISTENCE,
        NLBL_SIGMA_X,
        NLBL_SIGMA_Z,
        _mv_lift_matrices,
        _mv_mixing_matrix,
    )

    if ctx.model is None or ctx.loader is None:
        return {
            "obs_space_jsd_available": False,
            "obs_space_jsd_reason": "no model or loader",
        }
    dataset = getattr(ctx.loader, "dataset", None)
    mode = getattr(dataset, "mode", None) if dataset is not None else None
    if mode != "nonlinear-bimodal-lift-mv":
        return {
            "obs_space_jsd_available": False,
            "obs_space_jsd_reason": f"unsupported mode {mode!r}",
        }

    D_obs = int(NLBL_MV_OBS_D)
    W1, b1, W2, b2 = _mv_lift_matrices()  # (H, d), (H,), (D, H), (D,)
    A = _mv_mixing_matrix()  # (d, d)

    # 40 directions: n_random unit vectors in R^D + D axis-aligned.
    rng = np.random.default_rng(int(proj_seed))
    rand_dirs = rng.standard_normal(size=(int(n_random_projections), D_obs))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12
    axis_dirs = np.eye(D_obs, dtype=np.float64)
    directions = np.concatenate([rand_dirs, axis_dirs], axis=0)  # (n_dir, D)
    n_random = int(n_random_projections)
    n_dirs = directions.shape[0]

    model = ctx.model
    device = ctx.device
    transform = ctx.batch_transform
    S = int(ctx.num_samples)

    # Accumulators: per-origin lists of per-(batch-element, direction) JSD floats.
    per_origin_dir: dict[int, list[np.ndarray]] = {int(o): [] for o in origins}

    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx.loader):
            if batch_idx >= int(max_batches):
                break
            if transform is not None:
                batch = transform(batch, device)
            else:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            gt_latent = batch.get("gt_latent")
            gt_signs = batch.get("gt_signs")
            if gt_latent is None or gt_signs is None:
                return {
                    "obs_space_jsd_available": False,
                    "obs_space_jsd_reason": "missing gt_latent or gt_signs",
                }

            obs = batch["observed_data"]
            mask = batch["observation_mask"]
            timepoints = batch["timepoints"]
            B_ = obs.shape[0]
            T_full = obs.shape[-1]

            for origin in origins:
                origin = int(origin)
                if origin < 1 or origin >= T_full:
                    continue

                past_time = timepoints[:, :origin]
                future_time = timepoints[:, origin:]

                out = model.forecast(
                    x_hist=obs[..., :origin],
                    x_mask=mask[..., :origin],
                    past_time=past_time,
                    future_time=future_time,
                    num_samples=S,
                )
                # First future step: (B, S, D)
                model_samples = (
                    out["pred_samples"][:, :, :, 0].detach().cpu().numpy()
                )

                # Truth side: z_{t-1}, s_{t-1} are stored at index origin-1.
                z_prev = gt_latent[:, :, origin - 1].detach().cpu().numpy()  # (B, d)
                s_prev = gt_signs[:, :, origin - 1].detach().cpu().numpy()  # (B, d)
                latent_d = z_prev.shape[-1]

                # Draw K=S sign transitions per batch element per dim.
                keep = (
                    rng.random(size=(B_, S, latent_d)) < NLBL_MV_SIGN_PERSISTENCE
                )
                s_new = np.where(keep, s_prev[:, None, :], -s_prev[:, None, :])
                z_noise = rng.standard_normal(size=(B_, S, latent_d))
                # z_t = tanh(A z_{t-1}) + delta * s_t + sigma_z * eta
                # z_prev @ A.T -> (B, d); broadcast over sample axis.
                Az = z_prev @ A.T  # (B, d)
                z_t = (
                    np.tanh(Az)[:, None, :]
                    + NLBL_DELTA * s_new
                    + NLBL_SIGMA_Z * z_noise
                )  # (B, S, d)
                # Lift: h = tanh(z_t @ W1.T + b1); x = h @ W2.T + b2 + sigma_x * xi
                h = np.tanh(z_t @ W1.T + b1)  # (B, S, H)
                x_noise = rng.standard_normal(size=(B_, S, D_obs))
                truth_samples = h @ W2.T + b2 + NLBL_SIGMA_X * x_noise  # (B, S, D)

                # Project + JSD per (batch-element, direction).
                # model_samples: (B, S, D); truth_samples: (B, S, D)
                # projections: (n_dir, D)
                m_proj = np.einsum("bsd,kd->bks", model_samples, directions)
                t_proj = np.einsum("bsd,kd->bks", truth_samples, directions)
                jsd_bd = np.zeros((B_, n_dirs), dtype=np.float64)
                for b in range(B_):
                    for k in range(n_dirs):
                        mv = m_proj[b, k]
                        tv = t_proj[b, k]
                        lo = float(min(mv.min(), tv.min()))
                        hi = float(max(mv.max(), tv.max()))
                        if hi <= lo:
                            hi = lo + 1e-6
                        edges = np.linspace(lo, hi, n_bins + 1)
                        p = _hist_mass(mv, edges)
                        q = _hist_mass(tv, edges)
                        jsd_bd[b, k] = _jsd_discrete(p, q)
                per_origin_dir[origin].append(jsd_bd)

    # Reduce.
    per_origin_mean: dict[int, float] = {}
    per_origin_arrays: list[np.ndarray] = []
    for origin in origins:
        origin = int(origin)
        arrs = per_origin_dir.get(origin, [])
        if not arrs:
            per_origin_mean[origin] = float("nan")
            continue
        stacked = np.concatenate(arrs, axis=0)  # (total_B, n_dirs)
        per_origin_arrays.append(stacked)
        # Mean over batch elements, mean over directions.
        per_origin_mean[origin] = float(stacked.mean())

    if not per_origin_arrays:
        return {
            "obs_space_jsd_available": False,
            "obs_space_jsd_reason": "no batches processed",
        }
    all_stacked = np.concatenate(per_origin_arrays, axis=0)  # (Σ_B, n_dirs)
    headline = float(all_stacked.mean())
    # Per-dim: only the axis-aligned block (columns n_random .. n_random + D).
    per_dim = [
        float(all_stacked[:, n_random + di].mean()) for di in range(D_obs)
    ]

    return {
        "obs_space_jsd_available": True,
        "obs_space_jsd_mean": headline,
        "obs_space_jsd_per_origin": per_origin_mean,
        "obs_space_jsd_per_dim": per_dim,
    }


# ---------------------------------------------------------------------------
# CSV-derived metrics: cheap post-hoc summaries of the training log.
# ---------------------------------------------------------------------------


@register_metric("loss_tail")
def eval_loss_tail(
    ctx: EvalContext, *, column: str = "loss/total", tail_frac: float = 0.1
) -> dict[str, Any]:
    """Mean of the final ``tail_frac`` of values in a CSV column."""
    if not ctx.csv_path:
        return {column.replace("/", "_") + "_tail": float("nan")}
    import csv as _csv

    values: list[float] = []
    try:
        with open(ctx.csv_path, newline="") as f:
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


# ---------------------------------------------------------------------------
# Model-v2 headline metrics (init-experiment.org § Headline metrics).
#
# The metrics that the init-centering ablation grid uses to score each cell.
# The marginal log-likelihood ``nll`` (the doc's headline metric #1, the
# cell-fair ranking objective) is registered alongside the ELBO surrogate.
# ---------------------------------------------------------------------------


@register_metric("wallclock_to_target")
def eval_wallclock_to_target(
    ctx: EvalContext,
    *,
    target_column: str = "loss/total",
    target_value: float = 0.0,
    direction: str = "<=",
    time_column: str = "time/elapsed_s",
    step_column: str = "step",
) -> dict[str, Any]:
    """Wall-clock (and step) at which a metric first crossed a threshold.

    Walks ``ctx.csv_path`` in order; returns the ``(step, elapsed_s)``
    of the first row where ``target_column`` crosses ``target_value``
    in the given ``direction`` ("<=" or ">="), or ``null`` for both
    fields if no row crosses.

    Per ``init-experiment.org`` § Headline metrics, metric 5
    ("wall-clock time to a target metric").
    """
    if direction not in ("<=", ">="):
        raise ValueError(f"direction must be '<=' or '>='; got {direction!r}")
    null_result = {
        "wallclock_to_target_step": None,
        "wallclock_to_target_seconds": None,
        "wallclock_to_target_column": target_column,
        "wallclock_to_target_value": float(target_value),
        "wallclock_to_target_direction": direction,
    }
    if not ctx.csv_path:
        return null_result
    import csv as _csv

    try:
        with open(ctx.csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                raw = row.get(target_column, "")
                if raw in ("", None):
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v):
                    continue
                crossed = v <= target_value if direction == "<=" else v >= target_value
                if not crossed:
                    continue
                step_raw = row.get(step_column, "")
                time_raw = row.get(time_column, "")
                try:
                    step_int = (
                        int(float(step_raw)) if step_raw not in ("", None) else None
                    )
                except (TypeError, ValueError):
                    step_int = None
                try:
                    elapsed = float(time_raw) if time_raw not in ("", None) else None
                except (TypeError, ValueError):
                    elapsed = None
                return {
                    "wallclock_to_target_step": step_int,
                    "wallclock_to_target_seconds": elapsed,
                    "wallclock_to_target_column": target_column,
                    "wallclock_to_target_value": float(target_value),
                    "wallclock_to_target_direction": direction,
                }
    except OSError:
        return null_result
    return null_result


@register_metric("wallclock_to_relative_target")
def eval_wallclock_to_relative_target(
    ctx: EvalContext,
    *,
    target_column: str = "loss/total",
    descent_frac: float = 0.9,
    time_column: str = "time/elapsed_s",
    step_column: str = "step",
) -> dict[str, Any]:
    """Wall-clock at which a trial first reached a fraction of its own descent.

    Unlike :func:`eval_wallclock_to_target` (which compares against a
    fixed threshold), this metric is *self-referential*: each trial
    defines its own target as

        target_value = init_loss - descent_frac * (init_loss - final_loss)

    where ``init_loss`` is the first finite value of ``target_column``
    and ``final_loss`` is the last. Reports the step + seconds at
    which the trial first crossed that value (descending). Always
    defined for any non-degenerate trial, so useful as a
    multi-objective diagnostic when paired with the absolute
    :func:`eval_wallclock_to_target`.

    Note: a trial whose loss bounces (descends then comes back up)
    will report the time at which it first reached the implied
    target — which may be mid-trajectory rather than at the end.
    That is intentional: it measures descent efficiency, not final
    fit quality.
    """
    null_result = {
        "wallclock_to_relative_target_step": None,
        "wallclock_to_relative_target_seconds": None,
        "wallclock_to_relative_target_column": target_column,
        "wallclock_to_relative_target_descent_frac": float(descent_frac),
        "wallclock_to_relative_target_init_loss": None,
        "wallclock_to_relative_target_final_loss": None,
        "wallclock_to_relative_target_implied_target": None,
    }
    if not ctx.csv_path:
        return null_result
    import csv as _csv

    init_v: float | None = None
    final_v: float | None = None
    # First pass: find init and final values.
    try:
        with open(ctx.csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                raw = row.get(target_column, "")
                if raw in ("", None):
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v):
                    continue
                if init_v is None:
                    init_v = v
                final_v = v
    except OSError:
        return null_result
    if init_v is None or final_v is None or init_v <= final_v:
        # No descent (init <= final means loss didn't go down).
        return null_result
    implied_target = init_v - float(descent_frac) * (init_v - final_v)
    null_result["wallclock_to_relative_target_init_loss"] = float(init_v)
    null_result["wallclock_to_relative_target_final_loss"] = float(final_v)
    null_result["wallclock_to_relative_target_implied_target"] = float(implied_target)
    # Second pass: find first crossing.
    try:
        with open(ctx.csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                raw = row.get(target_column, "")
                if raw in ("", None):
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v) or v > implied_target:
                    continue
                step_raw = row.get(step_column, "")
                time_raw = row.get(time_column, "")
                try:
                    step_int = (
                        int(float(step_raw)) if step_raw not in ("", None) else None
                    )
                except (TypeError, ValueError):
                    step_int = None
                try:
                    elapsed = float(time_raw) if time_raw not in ("", None) else None
                except (TypeError, ValueError):
                    elapsed = None
                return {
                    **null_result,
                    "wallclock_to_relative_target_step": step_int,
                    "wallclock_to_relative_target_seconds": elapsed,
                }
    except OSError:
        return null_result
    return null_result


@register_metric("stage2_elbo_surrogate")
def eval_stage2_elbo_surrogate(
    ctx: EvalContext,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Stage-2 ELBO surrogate on a held-out split.

    Walks ``ctx.loader``, calls ``model.forward(train=False, ...)`` on
    each batch (the existing core code path computes every piece of
    ``L_total`` per ``model-v2.org`` § Assembled losses for stage 2),
    accumulates the rate + distortion sub-components, and returns the
    seven scalar pieces of the surrogate plus the total.

    Per ``init-experiment.org`` § Headline metrics, metric 2. The
    cell-fair ranking alternative is ``nll`` (metric 1, registered
    below); this surrogate is faster but not cell-invariant.
    """
    from ddssm.model.dssd import DDSSM_base

    if ctx.model is None or ctx.loader is None:
        return {"stage2_elbo_surrogate": float("nan")}
    model = ctx.require_module(DDSSM_base)

    sums = {
        "stage2_elbo_surrogate": 0.0,
        "recon": 0.0,
        "init_loss": 0.0,
        "init_kl_aux": 0.0,
        "init_entropy": 0.0,
        "trans_kl": 0.0,
    }
    n_batches = 0
    transform = ctx.batch_transform
    device = ctx.device
    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx.loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            if transform is not None:
                batch = transform(batch, device)
            else:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            components, metrics, _ = model(
                batch["observed_data"],
                batch["observation_mask"],
                batch["timepoints"],
                covariates=batch.get("covariates"),
                static_covariates=batch.get("static_covariates"),
                train=False,
            )
            # Reconstruct a "loss/total"-style surrogate from the
            # unweighted LossComponents: distortion + the ELBO rate
            # terms.
            loss = components.recon + components.init_kl + components.trans_kl
            sums["stage2_elbo_surrogate"] += float(loss.item())
            sums["recon"] += float(
                metrics.get("loss/distortion/rec", 0.0).item()
                if hasattr(metrics.get("loss/distortion/rec", 0.0), "item")
                else metrics.get("loss/distortion/rec", 0.0)
            )
            sums["init_loss"] += float(
                metrics.get("loss/rate/init/loss_init", 0.0).item()
                if hasattr(metrics.get("loss/rate/init/loss_init", 0.0), "item")
                else metrics.get("loss/rate/init/loss_init", 0.0)
            )
            sums["init_kl_aux"] += float(
                metrics.get("loss/rate/init/kl_aux", 0.0).item()
                if hasattr(metrics.get("loss/rate/init/kl_aux", 0.0), "item")
                else metrics.get("loss/rate/init/kl_aux", 0.0)
            )
            sums["init_entropy"] += float(
                metrics.get("loss/rate/init/entropy", 0.0).item()
                if hasattr(metrics.get("loss/rate/init/entropy", 0.0), "item")
                else metrics.get("loss/rate/init/entropy", 0.0)
            )
            sums["trans_kl"] += float(
                metrics.get("loss/rate/trans/kl", 0.0).item()
                if hasattr(metrics.get("loss/rate/trans/kl", 0.0), "item")
                else metrics.get("loss/rate/trans/kl", 0.0)
            )
            n_batches += 1

    if n_batches == 0:
        return {"stage2_elbo_surrogate": float("nan")}
    out = {f"stage2_elbo_surrogate_{k}": v / n_batches for k, v in sums.items()}
    # The total surrogate (loss/total under diffusion) is also the headline value.
    out["stage2_elbo_surrogate"] = out.pop(
        "stage2_elbo_surrogate_stage2_elbo_surrogate"
    )
    out["stage2_elbo_surrogate_n_batches"] = int(n_batches)
    return out


@register_metric("nll")
def eval_nll(
    ctx: EvalContext,
    *,
    num_iwae_samples: int | None = None,
    divergence_mode: str = "exact",
    num_hutchinson_probes: int = 1,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    method: str = "dopri5",
    seed: int | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Marginal NLL ``-log p_ψ(x_{1:T})`` via the prob-flow ODE + IWAE.

    Walks ``ctx.loader`` and calls :meth:`DDSSM_base.log_prob` on each
    batch. The result is the cell-fair ranking objective (model-v2.org §
    "Exact likelihood evaluation"): unlike ``stage2_elbo_surrogate``, it
    does not depend on the EDM preconditioning scale or the per-cell
    regulariser shape.

    Args:
        num_iwae_samples: trajectory samples ``K`` for the IWAE
            estimator. ``None`` (default) falls back to ``model.S``.
        divergence_mode: ``"exact"`` (D reverse-mode passes per ODE
            evaluation, zero variance) or ``"hutchinson"`` (one
            reverse-mode pass against a Rademacher probe vector ``v``
            held fixed over the ODE solve, unbiased on the log-density
            scale).
        num_hutchinson_probes: number of independent Hutchinson runs to
            average over (variance ÷ N). Each call uses a fresh probe
            vector. Ignored when ``divergence_mode == "exact"``.
        rtol, atol, method: torchdiffeq solver tolerances and method.
        seed: optional seed for reproducible Hutchinson draws.
        max_batches: cap on the number of batches walked from
            ``ctx.loader``. ``None`` walks the full loader.

    Returns:
        Dict with ``nll`` (mean over sequences of ``-log p_ψ(x_{1:T})``),
        bookkeeping (``nll_n_batches``, ``nll_n_sequences``), and the
        knob values used (``nll_num_iwae_samples``,
        ``nll_num_hutchinson_probes``, ``nll_divergence_mode``).
    """
    from ddssm.model.dssd import DDSSM_base

    if ctx.model is None or ctx.loader is None:
        return {"nll": float("nan")}

    # Validate arguments before touching the module — tests pass ``model=object()``
    # as a placeholder to check that ``ValueError`` fires on bad knobs without
    # requiring a real adapter.
    if divergence_mode not in {"exact", "hutchinson"}:
        raise ValueError(
            f"divergence_mode must be 'exact' or 'hutchinson'; got {divergence_mode!r}"
        )
    if num_hutchinson_probes < 1:
        raise ValueError(
            f"num_hutchinson_probes must be >= 1; got {num_hutchinson_probes}"
        )

    model = ctx.require_module(DDSSM_base)

    n_probes = num_hutchinson_probes if divergence_mode == "hutchinson" else 1
    generator = (
        torch.Generator(device=ctx.device).manual_seed(int(seed))
        if seed is not None
        else None
    )

    device = ctx.device
    transform = ctx.batch_transform

    total_neg_logp = 0.0
    n_sequences = 0
    n_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx.loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            if transform is not None:
                batch = transform(batch, device)
            else:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            probes = torch.zeros(
                batch["observed_data"].shape[0], device=device, dtype=torch.float64
            )
            for _ in range(n_probes):
                log_p = model.log_prob(
                    batch["observed_data"],
                    batch["observation_mask"],
                    batch["timepoints"],
                    covariates=batch.get("covariates"),
                    static_covariates=batch.get("static_covariates"),
                    K=num_iwae_samples,
                    rtol=rtol,
                    atol=atol,
                    method=method,
                    divergence_mode=divergence_mode,
                    generator=generator,
                )
                probes = probes + log_p.to(dtype=torch.float64)
            mean_log_p = probes / float(n_probes)
            total_neg_logp += float((-mean_log_p).sum().item())
            n_sequences += int(mean_log_p.shape[0])
            n_batches += 1

    if n_sequences == 0:
        return {"nll": float("nan")}

    return {
        "nll": total_neg_logp / n_sequences,
        "nll_n_batches": n_batches,
        "nll_n_sequences": n_sequences,
        "nll_num_iwae_samples": num_iwae_samples,
        "nll_num_hutchinson_probes": int(n_probes),
        "nll_divergence_mode": divergence_mode,
    }


@register_metric("sigma_data_drift")
def eval_sigma_data_drift(
    ctx: EvalContext,
    *,
    max_batches: int = 4,
) -> dict[str, Any]:
    """σ_data²(t) snapshot + the two-component decomposition.

    Per ``model-v2.org`` § Data-variance tracking:

      ``σ_data²(t) = (1/D) ( E[‖σ_t‖²] + tr Var[μ̂_t] )``

    where ``μ̂_t = μ_t − μ_p(z_{t-1})`` is the centered residual mean.
    The first component captures the *average posterior variance* and
    the second the *spread of residual means*.  This metric returns
    both components per t plus the buffer's per-t values, summing
    over a few held-out batches for the empirical components.

    Per ``init-experiment.org`` § Headline metrics, metric 6 ("σ_data
    drift trajectory + two-component decomposition").  This is the
    snapshot variant — the trajectory plot in Phase E reads from
    ``diag/sigma_data2/t=N`` columns in ``metrics.csv``.
    """
    from ddssm.model.dssd import DDSSM_base

    if ctx.model is None or ctx.loader is None:
        return {"sigma_data_drift_available": False}
    model = ctx.require_module(DDSSM_base)
    # Within-DDSSM soft guard: a DDSSM without a sigma_data buffer (e.g. a
    # non-diffusion transition) has nothing to report -- not a family mismatch.
    if getattr(model, "sigma_data", None) is None:
        return {"sigma_data_drift_available": False}
    device = ctx.device
    transform = ctx.batch_transform

    sigma_data2 = model.sigma_data.sigma_data2.detach().cpu().tolist()

    # Empirical components from a few held-out batches.
    j = int(model.j)
    d = int(model.latent_dim)
    comp1_per_t: dict[int, list[float]] = {}  # t -> list of E[||σ_t||^2]
    comp2_per_t: dict[int, list[float]] = {}  # t -> list of tr Var[μ̂_t]

    from ddssm.nn.net_utils import time_embedding

    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx.loader):
            if batch_idx >= int(max_batches):
                break
            if transform is not None:
                batch = transform(batch, device)
            else:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            obs = batch["observed_data"]
            mask = batch["observation_mask"]
            tp = batch["timepoints"]
            te = time_embedding(tp, model.emb_time_dim, device=device)
            zs, _, stats = model._encode_latents(
                observed_data=obs,
                time_embed=te,
                observation_mask=mask,
            )
            B, S, _, T = zs.shape
            if j >= T:
                continue
            mus = stats["mus"]
            logvars = stats["logvars"]
            sigma2 = logvars.exp()  # (B, S, d, T)

            # For each transition target t in 1..T-1 (0-based code idx
            # t = j..T-1), compute:
            #   component 1 = mean over (b, s) of sum_d sigma_t^2
            #   component 2 = var over (b, s) of (mu_t - mu_p(z_hist))
            #                 summed over d (trace).
            baseline = getattr(model, "baseline", None)
            for t in range(j, T):
                z_hist = zs[:, :, :, t - j : t]  # (B, S, d, j)
                z_hist_flat = z_hist.reshape(B * S, d, j)
                mu_t = mus[:, :, :, t].reshape(B * S, d)
                lv_t = sigma2[:, :, :, t].reshape(B * S, d)

                if baseline is not None:
                    mu_p = baseline.mean(z_hist_flat)
                    mu_hat = mu_t - mu_p
                else:
                    mu_hat = mu_t

                # Component 1: average ||sigma_t||^2.
                c1 = lv_t.sum(dim=-1).mean().item() / d
                # Component 2: tr Var across the BS axis of mu_hat,
                # summed over d, then divided by d (matches the
                # 1/D normalisation in the doc).
                c2 = float(mu_hat.var(dim=0, unbiased=False).sum().item() / d)

                t_ext = t + 1  # 1-based
                comp1_per_t.setdefault(t_ext, []).append(c1)
                comp2_per_t.setdefault(t_ext, []).append(c2)

    # Reduce.
    sorted_ts = sorted(comp1_per_t.keys())
    comp1_list = [float(np.mean(comp1_per_t[t])) for t in sorted_ts]
    comp2_list = [float(np.mean(comp2_per_t[t])) for t in sorted_ts]
    decomp_sum = [a + b for a, b in zip(comp1_list, comp2_list)]
    return {
        "sigma_data_drift_available": True,
        "sigma_data2_buffer": sigma_data2,
        "sigma_data2_t_indices": sorted_ts,
        "sigma_data2_component1_per_t": comp1_list,
        "sigma_data2_component2_per_t": comp2_list,
        "sigma_data2_decomposition_sum_per_t": decomp_sum,
    }


@register_metric("q_aux_kl_trajectory")
def eval_q_aux_kl_trajectory(
    ctx: EvalContext,
    *,
    collapse_threshold: float = 1e-3,
) -> dict[str, Any]:
    """KL[q_Φ(z_0|z_1) ‖ N(0, I)] trajectory over training (secondary metric #5).

    Reads the per-step ``loss/rate/init/kl_aux`` column from
    ``run_dir/metrics.csv`` and returns the trajectory + summary
    statistics. Per ``init-experiment.org`` § Secondary metrics: this
    is the diagnostic for q_Φ posterior collapse. The KL should rise
    off zero early in stage 1 and stabilise; collapsing back to zero
    indicates q_Φ has ignored z_1 and the VHP has degenerated to the
    homoskedastic prior.

    Returns ``{q_aux_kl_trajectory_available: False}`` if the run dir
    or CSV column is missing.
    """
    if ctx.run_dir is None:
        return {
            "q_aux_kl_trajectory_available": False,
            "q_aux_kl_trajectory_reason": "no run_dir",
        }
    csv_path = os.path.join(ctx.run_dir, "metrics.csv")
    if not os.path.isfile(csv_path):
        return {
            "q_aux_kl_trajectory_available": False,
            "q_aux_kl_trajectory_reason": "no metrics.csv",
        }
    steps: list[int] = []
    values: list[float] = []
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            if "loss/rate/init/kl_aux" not in (reader.fieldnames or []):
                return {
                    "q_aux_kl_trajectory_available": False,
                    "q_aux_kl_trajectory_reason": "kl_aux column missing",
                }
            for row in reader:
                v_raw = row.get("loss/rate/init/kl_aux", "")
                if v_raw == "" or v_raw is None:
                    continue
                try:
                    v = float(v_raw)
                except (TypeError, ValueError):
                    continue
                try:
                    s = int(row.get("step", 0))
                except (TypeError, ValueError):
                    continue
                steps.append(s)
                values.append(v)
    except OSError:
        return {
            "q_aux_kl_trajectory_available": False,
            "q_aux_kl_trajectory_reason": "csv read error",
        }

    if not values:
        return {
            "q_aux_kl_trajectory_available": False,
            "q_aux_kl_trajectory_reason": "empty",
        }
    final = float(values[-1])
    mean_v = float(np.mean(values))
    peak = float(max(values))
    # Posterior collapse heuristic: peak rose above threshold but final
    # value collapsed back below it.
    collapsed = bool(peak > collapse_threshold and final < collapse_threshold)
    return {
        "q_aux_kl_trajectory_available": True,
        "q_aux_kl_trajectory_steps": steps,
        "q_aux_kl_trajectory_values": values,
        "q_aux_kl_trajectory_final": final,
        "q_aux_kl_trajectory_mean": mean_v,
        "q_aux_kl_trajectory_peak": peak,
        "q_aux_kl_trajectory_collapsed": collapsed,
    }


@register_metric("log_sigma_p2_collapse")
def eval_log_sigma_p2_collapse(
    ctx: EvalContext,
    *,
    max_batches: int = 4,
    outlier_z_threshold: float = 2.0,
) -> dict[str, Any]:
    """Per-(t, d) ``log σ_p²(z_{t-1})`` diagnostic (secondary metric #6).

    Per ``init-experiment.org`` § Secondary metrics: the global anchor
    ``R_σp`` can be satisfied by some (t, d) cells collapsing while
    others inflate (balancing in aggregate). This metric surfaces that
    pathology by computing the per-(t, d) batch mean of
    ``log σ_p²(z_{t-1})_d`` over an evaluation trajectory sample and
    flagging cells whose absolute mean exceeds ``outlier_z_threshold``
    standard deviations from zero.

    Requires ``model.baseline`` with a ``mean_and_logvar`` head (all
    four baseline forms provide it, including the parameter-free
    Zero/Persistence via their state-conditional σ_p head).
    """
    from ddssm.model.dssd import DDSSM_base

    if ctx.model is None or ctx.loader is None:
        return {"log_sigma_p2_collapse_available": False}
    model = ctx.require_module(DDSSM_base)
    # Within-DDSSM soft guard: a DDSSM without a baseline head has nothing to
    # report (not a cross-family mismatch).
    if getattr(model, "baseline", None) is None:
        return {"log_sigma_p2_collapse_available": False}
    device = ctx.device
    transform = ctx.batch_transform
    j = int(model.j)
    d = int(model.latent_dim)
    baseline = model.baseline

    from ddssm.nn.net_utils import time_embedding

    # Accumulate per-(t, d) sums across batches.
    sums: dict[int, np.ndarray] = {}  # t -> (d,)
    counts: dict[int, int] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx.loader):
            if batch_idx >= int(max_batches):
                break
            if transform is not None:
                batch = transform(batch, device)
            else:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            obs = batch["observed_data"]
            mask = batch["observation_mask"]
            tp = batch["timepoints"]
            te = time_embedding(tp, model.emb_time_dim, device=device)
            zs, _, _ = model._encode_latents(
                observed_data=obs,
                time_embed=te,
                observation_mask=mask,
            )
            B, S, _, T = zs.shape
            if j >= T:
                continue
            for t in range(j, T):
                z_hist = zs[:, :, :, t - j : t].reshape(B * S, d, j)
                _, log_sigma_p2 = baseline.mean_and_logvar(z_hist)
                # Mean across the (B * S) axis → per-d.
                per_d = log_sigma_p2.mean(dim=0).detach().cpu().numpy()
                t_ext = t + 1  # 1-based
                if t_ext in sums:
                    sums[t_ext] += per_d
                    counts[t_ext] += 1
                else:
                    sums[t_ext] = per_d.copy()
                    counts[t_ext] = 1

    if not sums:
        return {"log_sigma_p2_collapse_available": False}
    sorted_ts = sorted(sums.keys())
    per_t_per_d: list[list[float]] = []
    for t in sorted_ts:
        mean_per_d = sums[t] / counts[t]
        per_t_per_d.append([float(v) for v in mean_per_d])

    flat = np.array(per_t_per_d, dtype=np.float64)  # (T, d)
    overall_mean = float(flat.mean())
    overall_std = float(flat.std(ddof=0))
    # Outliers: (t, d) whose |value| exceeds threshold × std (deviation from 0).
    outliers: list[tuple[int, int, float]] = []
    if overall_std > 0:
        for ti, t in enumerate(sorted_ts):
            for di in range(d):
                v = flat[ti, di]
                if abs(v) > outlier_z_threshold * overall_std:
                    outliers.append((int(t), int(di), float(v)))

    return {
        "log_sigma_p2_collapse_available": True,
        "log_sigma_p2_t_indices": sorted_ts,
        "log_sigma_p2_per_t_per_d": per_t_per_d,
        "log_sigma_p2_mean": overall_mean,
        "log_sigma_p2_std": overall_std,
        "log_sigma_p2_n_outliers": len(outliers),
        "log_sigma_p2_outliers": [
            {"t": t, "d": di, "log_sigma_p2": v} for t, di, v in outliers
        ],
    }


@register_metric("crps_sum_latent")
def eval_crps_sum_latent(
    ctx: EvalContext,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """CRPS-sum on latent samples vs. ground-truth latents.

    Mirrors the obs-space ``crps_sum`` metric but operates on the
    latent path: the encoder + transition produce ``(B, S, d, T)``
    samples of z (via ``model.forecast``'s latent rollout), and the
    ground-truth latents come from the data module's
    ``expose_gt_latents`` surface.

    Returns ``{crps_sum_latent_mean, crps_sum_latent_per_t}`` when
    GT latents are available; ``{crps_sum_latent_available: False}``
    otherwise.  Per ``init-experiment.org`` § Headline metrics,
    metric 4 ("CRPS-sum across forecast horizons, in both latent
    and observation spaces").

    The global ND uses the **ratio-of-means** convention: pinball
    numerator and ND denominator are accumulated separately across
    batches and divided once at the end (matching ``eval_crps_sum``
    and the GluonTS / CSDI published convention).  ``crps_sum_latent_per_t``
    is each timestep's numerator (summed over all batch elements) divided by
    the global denominator, so ``sum(crps_sum_latent_per_t) == crps_sum_latent_mean``.
    """
    from ddssm.model.dssd import DDSSM_base

    if ctx.model is None or ctx.loader is None or ctx.T_split is None:
        return {"crps_sum_latent_available": False}
    from ddssm.eval.eval_metrics import _crps_sum_pinball

    model = ctx.require_module(DDSSM_base)
    device = ctx.device
    transform = ctx.batch_transform
    L1 = int(ctx.T_split)
    num_acc: list[np.ndarray] = []  # (B, L2) per batch
    denom_acc: list[float] = []  # scalar per batch
    n_batches = 0

    from ddssm.nn.net_utils import time_embedding

    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx.loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            if transform is not None:
                batch = transform(batch, device)
            else:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            gt_latent = batch.get("gt_latent")
            if gt_latent is None:
                return {"crps_sum_latent_available": False}

            T = batch["observed_data"].shape[-1]
            x_hist = batch["observed_data"][..., :L1]
            x_mask = batch["observation_mask"][..., :L1]
            past_time = batch["timepoints"][:, :L1]
            future_time = batch["timepoints"][:, L1:]

            # Encode the past, then roll out latents into the future
            # via the transition (this is what ``model.forecast`` does
            # internally; we re-implement it lightly to expose latent
            # samples rather than decoder samples).
            te_past = time_embedding(past_time, model.emb_time_dim, device=device)
            zs_past, _, stats = model.encoder.sample_paths(
                observed_data=x_hist,
                time_embed=te_past,
                S=int(ctx.num_samples),
                cond_mask=x_mask if getattr(model.encoder, "use_mask", False) else None,
            )
            B, S, d, _ = zs_past.shape
            j = int(model.j)

            # Pad if past < j.
            if j <= L1:
                z_hist = zs_past[..., -j:]
            else:
                pad = torch.zeros(B, S, d, j - L1, device=device, dtype=zs_past.dtype)
                z_hist = torch.cat([pad, zs_past], dim=-1)
            z_hist_flat = z_hist.reshape(B * S, d, j)

            # Roll out future latents.  Use a no-context sample for the
            # transition (covariates/time-embeddings are stub-zero —
            # acceptable for the simple synthetic modes the GT-latent
            # surface covers).
            L2 = future_time.size(1)
            future_zs = torch.zeros(B, S, d, L2, device=device, dtype=zs_past.dtype)
            for t_step in range(L2):
                ctx_dict = {
                    "hist_time_emb": torch.zeros(
                        B * S,
                        j,
                        model.emb_time_dim,
                        device=device,
                    ),
                    "target_time_emb": torch.zeros(
                        B * S,
                        1,
                        model.emb_time_dim,
                        device=device,
                    ),
                }
                z_next = model.transition.sample(z_hist_flat, S=1, ctx=ctx_dict)
                z_next = z_next.squeeze(1)  # (BS, d)
                future_zs[:, :, :, t_step] = z_next.view(B, S, d)
                if j > 1:
                    z_hist_flat = torch.cat(
                        [z_hist_flat[:, :, 1:], z_next.unsqueeze(-1)],
                        dim=-1,
                    )
                else:
                    z_hist_flat = z_next.unsqueeze(-1)

            # Ground-truth latents over the same future range.
            z_gt = gt_latent[..., L1:T]  # (B, d, L2)
            z_sum = future_zs.sum(dim=2)  # (B, S, L2)
            y_sum = z_gt.sum(dim=1)  # (B, L2)
            num, denom = _crps_sum_pinball(z_sum, y_sum)
            num_acc.append(num.detach().cpu().numpy())
            denom_acc.append(float(denom.sum().item()))
            n_batches += 1

    if n_batches == 0:
        return {"crps_sum_latent_available": False}
    # Ratio-of-means: divide total pinball numerator by total denominator.
    total_denom = max(float(np.sum(denom_acc)), 1e-8)
    num_all = np.concatenate(num_acc, axis=0)  # (total_B, L2)
    global_nd = float(num_all.sum() / total_denom)
    per_t = (num_all.sum(axis=0) / total_denom).tolist()
    return {
        "crps_sum_latent_available": True,
        "crps_sum_latent_mean": global_nd,
        "crps_sum_latent_per_t": per_t,
    }


