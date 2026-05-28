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

import csv
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


# ---------------------------------------------------------------------------
# Model-v2 headline metrics (init-experiment.org § Headline metrics).
#
# Five metrics that the init-centering ablation grid uses to score each
# cell.  PF-ODE NLL (the doc's headline metric #1) is gated and deferred;
# the others are below.
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
) -> Dict[str, Any]:
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
        with open(ctx.csv_path, "r", newline="") as f:
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
                    step_int = int(float(step_raw)) if step_raw not in ("", None) else None
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
) -> Dict[str, Any]:
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
        with open(ctx.csv_path, "r", newline="") as f:
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
        with open(ctx.csv_path, "r", newline="") as f:
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
                    step_int = int(float(step_raw)) if step_raw not in ("", None) else None
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
) -> Dict[str, Any]:
    """Stage-2 ELBO surrogate on a held-out split.

    Walks ``ctx.loader``, calls ``model.forward(train=False, ...)`` on
    each batch (the existing core code path computes every piece of
    ``L_total`` per ``model-v2.org`` § Assembled losses for stage 2),
    accumulates the rate + distortion sub-components, and returns the
    seven scalar pieces of the surrogate plus the total.

    Per ``init-experiment.org`` § Headline metrics, metric 2.  Until
    PF-ODE NLL (metric 1) lands, this is the comparison objective for
    the ablation grid.
    """
    if ctx.model is None or ctx.loader is None:
        return {"stage2_elbo_surrogate": float("nan")}

    sums = {
        "stage2_elbo_surrogate": 0.0,
        "recon": 0.0,
        "init_loss": 0.0,
        "init_kl_aux": 0.0,
        "init_entropy": 0.0,
        "trans_kl": 0.0,
        "r_sigma_p": 0.0,
        "r_mu_p": 0.0,
    }
    n_batches = 0
    transform = ctx.batch_transform
    model = ctx.model
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
                covariates=batch.get("covariates", None),
                static_covariates=batch.get("static_covariates", None),
                train=False,
                report_scaled=False,
            )
            # Reconstruct the pre-ADR-0004 "loss/total" the eval used
            # to read from forward()'s first return value: distortion
            # + rate (with regularizers at their hparam weights).
            hp = getattr(model.config, "hyperparams", None)
            l_sp = float(getattr(hp, "lambda_sigma_p", 0.0)) if hp is not None else 0.0
            l_mp = float(getattr(model, "anchor_lambda", 0.0) or 0.0)
            loss = (
                components.recon
                + components.init_kl
                + components.trans_kl
                + l_sp * components.r_sigma_p
                + l_mp * components.r_mu_p
            )
            sums["stage2_elbo_surrogate"] += float(loss.item())
            sums["recon"] += float(metrics.get("loss/distortion/rec", 0.0).item() if hasattr(metrics.get("loss/distortion/rec", 0.0), "item") else metrics.get("loss/distortion/rec", 0.0))
            sums["init_loss"] += float(metrics.get("loss/rate/init/loss_init", 0.0).item() if hasattr(metrics.get("loss/rate/init/loss_init", 0.0), "item") else metrics.get("loss/rate/init/loss_init", 0.0))
            sums["init_kl_aux"] += float(metrics.get("loss/rate/init/kl_aux", 0.0).item() if hasattr(metrics.get("loss/rate/init/kl_aux", 0.0), "item") else metrics.get("loss/rate/init/kl_aux", 0.0))
            sums["init_entropy"] += float(metrics.get("loss/rate/init/entropy", 0.0).item() if hasattr(metrics.get("loss/rate/init/entropy", 0.0), "item") else metrics.get("loss/rate/init/entropy", 0.0))
            sums["trans_kl"] += float(metrics.get("loss/rate/trans/kl", 0.0).item() if hasattr(metrics.get("loss/rate/trans/kl", 0.0), "item") else metrics.get("loss/rate/trans/kl", 0.0))
            sums["r_sigma_p"] += float(metrics.get("loss/rate/trans/r_sigma_p", 0.0).item() if hasattr(metrics.get("loss/rate/trans/r_sigma_p", 0.0), "item") else metrics.get("loss/rate/trans/r_sigma_p", 0.0))
            sums["r_mu_p"] += float(metrics.get("loss/rate/trans/r_mu_p", 0.0).item() if hasattr(metrics.get("loss/rate/trans/r_mu_p", 0.0), "item") else metrics.get("loss/rate/trans/r_mu_p", 0.0))
            n_batches += 1

    if n_batches == 0:
        return {"stage2_elbo_surrogate": float("nan")}
    out = {f"stage2_elbo_surrogate_{k}": v / n_batches for k, v in sums.items()}
    # The total surrogate (loss/total under V3) is also the headline value.
    out["stage2_elbo_surrogate"] = out.pop("stage2_elbo_surrogate_stage2_elbo_surrogate")
    out["stage2_elbo_surrogate_n_batches"] = int(n_batches)
    return out


@register_metric("sigma_data_drift")
def eval_sigma_data_drift(
    ctx: EvalContext,
    *,
    max_batches: int = 4,
) -> Dict[str, Any]:
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
    if (
        ctx.model is None
        or ctx.loader is None
        or getattr(ctx.model, "sigma_data", None) is None
    ):
        return {"sigma_data_drift_available": False}
    model = ctx.model
    device = ctx.device
    transform = ctx.batch_transform

    sigma_data2 = model.sigma_data.sigma_data2.detach().cpu().tolist()

    # Empirical components from a few held-out batches.
    j = int(model.j)
    d = int(model.latent_dim)
    comp1_per_t: dict[int, list[float]] = {}  # t -> list of E[||σ_t||^2]
    comp2_per_t: dict[int, list[float]] = {}  # t -> list of tr Var[μ̂_t]

    from ..net_utils import time_embedding

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
            if T <= j:
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
) -> Dict[str, Any]:
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
        return {"q_aux_kl_trajectory_available": False, "q_aux_kl_trajectory_reason": "no run_dir"}
    csv_path = os.path.join(ctx.run_dir, "metrics.csv")
    if not os.path.isfile(csv_path):
        return {"q_aux_kl_trajectory_available": False, "q_aux_kl_trajectory_reason": "no metrics.csv"}
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
        return {"q_aux_kl_trajectory_available": False, "q_aux_kl_trajectory_reason": "csv read error"}

    if not values:
        return {"q_aux_kl_trajectory_available": False, "q_aux_kl_trajectory_reason": "empty"}
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
) -> Dict[str, Any]:
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
    Zero/Identity via their state-conditional σ_p head).
    """
    if (
        ctx.model is None
        or ctx.loader is None
        or getattr(ctx.model, "baseline", None) is None
    ):
        return {"log_sigma_p2_collapse_available": False}
    model = ctx.model
    device = ctx.device
    transform = ctx.batch_transform
    j = int(model.j)
    d = int(model.latent_dim)
    baseline = model.baseline

    from ..net_utils import time_embedding

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
                observed_data=obs, time_embed=te, observation_mask=mask,
            )
            B, S, _, T = zs.shape
            if T <= j:
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
) -> Dict[str, Any]:
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
    """
    if ctx.model is None or ctx.loader is None or ctx.T_split is None:
        return {"crps_sum_latent_available": False}
    from ..eval_metrics import crps_sum_latent_metrics

    model = ctx.model
    device = ctx.device
    transform = ctx.batch_transform
    L1 = int(ctx.T_split)
    means: list[float] = []
    per_t_accum: list[torch.Tensor] = []
    n_batches = 0

    from ..net_utils import time_embedding

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
            gt_latent = batch.get("gt_latent", None)
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
                        B * S, j, model.emb_time_dim, device=device,
                    ),
                    "target_time_emb": torch.zeros(
                        B * S, 1, model.emb_time_dim, device=device,
                    ),
                }
                z_next = model.transition.sample(z_hist_flat, S=1, ctx=ctx_dict)
                z_next = z_next.squeeze(1)  # (BS, d)
                future_zs[:, :, :, t_step] = z_next.view(B, S, d)
                if j > 1:
                    z_hist_flat = torch.cat(
                        [z_hist_flat[:, :, 1:], z_next.unsqueeze(-1)], dim=-1,
                    )
                else:
                    z_hist_flat = z_next.unsqueeze(-1)

            # Ground-truth latents over the same future range.
            z_gt = gt_latent[..., L1:T]  # (B, d, L2)
            crps_mean, crps_per_t = crps_sum_latent_metrics(
                z_samples=future_zs, z_gt=z_gt,
            )
            means.append(float(crps_mean.item()))
            per_t_accum.append(crps_per_t.cpu())
            n_batches += 1

    if n_batches == 0:
        return {"crps_sum_latent_available": False}
    per_t = torch.stack(per_t_accum, dim=0).mean(dim=0).tolist()
    return {
        "crps_sum_latent_available": True,
        "crps_sum_latent_mean": float(np.mean(means)),
        "crps_sum_latent_per_t": per_t,
    }


@register_metric("gt_latent_jsd")
def eval_gt_latent_jsd(
    ctx: EvalContext,
    *,
    max_batches: int = 2,
    n_bins: int = 60,
    edges_min: float = -3.0,
    edges_max: float = 3.0,
) -> Dict[str, Any]:
    """JSD between model-transition samples and the GT transition kernel.

    For each ``t ∈ [1, T-1]``, draws ``S`` samples from the model's
    learned ``p_ψ(z_t | z_{t-1})`` (with GT ``z_{t-1}`` as the
    conditioning), plus ``S`` samples from the analytic ground-truth
    transition kernel via :mod:`ddssm.eval.synthetic_kernels`.  Bins
    both into shared histograms and computes Jensen-Shannon divergence
    per t.

    Per ``init-experiment.org`` § Headline metrics, metric 3
    ("Transition JSD on ground-truth latents").  This metric is the
    *only* one that isolates the transition model from the encoder —
    both sample sets are conditioned on the same GT z_{t-1}.

    Skips with ``{gt_latent_jsd_available: False}`` when GT latents
    aren't exposed by the loader or when the synthetic mode lacks a
    registered closed-form kernel.
    """
    if ctx.model is None or ctx.loader is None:
        return {"gt_latent_jsd_available": False}
    from .synthetic_kernels import KERNEL_REGISTRY

    # Look up the data module's mode from the loader's dataset.
    mode = _infer_synthetic_mode(ctx.loader)
    if mode is None or mode not in KERNEL_REGISTRY:
        return {"gt_latent_jsd_available": False, "gt_latent_jsd_reason": f"no kernel for mode={mode!r}"}

    kernel = KERNEL_REGISTRY[mode]
    model = ctx.model
    device = ctx.device
    transform = ctx.batch_transform
    j = int(model.j)
    S = int(ctx.num_samples)

    edges = np.linspace(edges_min, edges_max, n_bins + 1)
    per_t_jsd: dict[int, list[float]] = {}
    n_batches = 0

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
            gt_latent = batch.get("gt_latent", None)
            if gt_latent is None:
                return {"gt_latent_jsd_available": False, "gt_latent_jsd_reason": "no gt_latent"}
            B, d, T = gt_latent.shape
            for t in range(j, T):
                # GT conditioning history (B, d, j).
                z_hist_gt = gt_latent[:, :, t - j : t]
                # Tile across S samples.
                z_hist_flat = z_hist_gt.unsqueeze(1).expand(B, S, d, j).reshape(B * S, d, j)

                ctx_dict = {
                    "hist_time_emb": torch.zeros(B * S, j, model.emb_time_dim, device=device),
                    "target_time_emb": torch.zeros(B * S, 1, model.emb_time_dim, device=device),
                }
                z_next_model = model.transition.sample(z_hist_flat, S=1, ctx=ctx_dict)
                z_next_model = z_next_model.squeeze(1).view(B, S, d).cpu().numpy()
                z_next_gt = kernel(z_hist_gt.cpu().numpy(), S, batch_idx=batch_idx, t=t)  # (B, S, d)

                # Per-dim JSD, averaged.
                jsds_d = []
                for di in range(d):
                    for b in range(B):
                        p = _hist_mass(z_next_model[b, :, di], edges)
                        q = _hist_mass(z_next_gt[b, :, di], edges)
                        jsds_d.append(_jsd_discrete(p, q))
                per_t_jsd.setdefault(t + 1, []).append(float(np.mean(jsds_d)))
            n_batches += 1

    if not per_t_jsd:
        return {"gt_latent_jsd_available": False}
    sorted_ts = sorted(per_t_jsd.keys())
    per_t_list = [float(np.mean(per_t_jsd[t])) for t in sorted_ts]
    return {
        "gt_latent_jsd_available": True,
        "gt_latent_jsd_mean": float(np.mean(per_t_list)),
        "gt_latent_jsd_per_t": per_t_list,
        "gt_latent_jsd_t_indices": sorted_ts,
        "gt_latent_jsd_n_batches": int(n_batches),
    }


def _infer_synthetic_mode(loader) -> str | None:
    """Best-effort lookup of the synthetic dataset's ``mode`` attribute."""
    try:
        ds = loader.dataset
        if hasattr(ds, "mode"):
            return str(ds.mode)
    except AttributeError:
        return None
    return None
