"""Stateless plot functions and the registry that exposes them by name.

Each plot function takes a :class:`PlotContext` plus its own keyword
args, draws to a Matplotlib ``Figure``, and saves to ``save_path``.
The viz runner walks the names listed in :class:`VizSpec.plots` and
calls each one in turn.

The plotting logic is split into composable pieces so a paper figure
that needs (say) only the 1-D forecast panel can be produced without
also drawing the 2-D spatial path.
"""

from __future__ import annotations

import csv as _csv
from typing import Any
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ddssm.adapters.base import MetricNotSupported


@dataclass
class PlotContext:
    """Inputs available to every plot function.

    ``model`` and ``loader`` may be unused by purely CSV-driven plots
    (e.g. training-loss curves), so both are nullable.

    ``model`` is the :class:`~ddssm.adapters.base.ModelAdapter`, not the raw
    ``nn.Module``. Forecast-based plots call the adapter surface
    (``model.forecast``); DDSSM-only plots reach the owned module via
    :meth:`require_module`.
    """

    model: Any | None
    loader: DataLoader | None
    device: torch.device
    batch_transform: Callable[[dict, torch.device], dict] | None = None
    csv_path: str | None = None
    T_split: int | None = None
    num_samples: int = 10

    def require_module(self, cls: type) -> torch.nn.Module:
        """Return the adapter's owned module iff it is a ``cls``; else raise.

        The gating prelude for DDSSM-only plots (mirrors
        :meth:`ddssm.eval.metrics.EvalContext.require_module`). A non-matching
        family raises :class:`MetricNotSupported`, which the viz runner catches
        to skip the plot -- NOT ``AttributeError`` (which would mask real bugs).
        Callers pass ``cls`` (lazy-imported at the call site) so this helper
        stays cycle-free.
        """
        module = self.model.module
        if not isinstance(module, cls):
            raise MetricNotSupported(
                f"{type(self.model).__name__} does not provide a {cls.__name__} module"
            )
        return module


PlotFn = Callable[..., None]
PLOT_REGISTRY: dict[str, PlotFn] = {}


def register_plot(name: str) -> Callable[[PlotFn], PlotFn]:
    """Decorator registering a plot function under ``name`` in the registry.

    Args:
        name: Registry key a ``VizSpec`` uses to select this plot.

    Returns:
        The decorator, which returns the function unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def _wrap(fn: PlotFn) -> PlotFn:
        if name in PLOT_REGISTRY:
            raise ValueError(f"Plot {name!r} already registered")
        PLOT_REGISTRY[name] = fn
        return fn

    return _wrap


# ---------------------------------------------------------------------------
# Helper: pull one batch (or specific sample indices) out of the loader and
# run the model's reconstruction + forecast paths.
# ---------------------------------------------------------------------------


def _gather_batch(ctx: PlotContext, sample_indices: list[int] | None):
    if ctx.loader is None:
        raise ValueError("Plot requires a non-empty loader.")
    if sample_indices is not None:
        from torch.utils.data.dataloader import default_collate

        items = [ctx.loader.dataset[i] for i in sample_indices]
        return default_collate(items)
    return next(iter(ctx.loader))


def _run_recon_and_forecast(
    ctx: PlotContext, batch: dict, T_split: int, num_samples: int
):
    from ddssm.model.dssd import DDSSM_base

    if ctx.model is None:
        raise ValueError("Reconstruction/forecast plots need a non-None model.")
    # DDSSM-only prelude: the recon path calls ``model(...)`` and reaches
    # ``.emb_time_dim`` / ``.j`` / ``.decoder``, so a forecast-only adapter is
    # gated here (raises ``MetricNotSupported`` → the viz runner skips the plot).
    model = ctx.require_module(DDSSM_base)
    device = ctx.device
    if ctx.batch_transform is not None:
        batch = ctx.batch_transform(batch, device)

    observed = batch["observed_data"]
    mask = batch["observation_mask"]
    timepoints = batch["timepoints"]

    with torch.no_grad():
        _components, _metrics, stats = model(observed, mask, timepoints, train=False)
        zs = stats["zs"]
        z_sample = zs[:, 0, :, :]

        from ddssm.nn.net_utils import time_embedding

        te = time_embedding(timepoints, model.emb_time_dim, device=device)

        recons = []
        T = observed.shape[-1]
        for t in range(T):
            t_idx = torch.full((observed.shape[0],), t, device=device, dtype=torch.long)
            z_hist = z_sample[..., : t + 1]
            if z_hist.shape[-1] > model.j:
                z_hist = z_hist[..., -model.j :]
            mu_x, _ = model.decoder(z_hist, te, t_idx)
            recons.append(mu_x)
        recons = torch.stack(recons, dim=-1)

        x_hist = observed[..., :T_split].contiguous()
        x_mask = mask[..., :T_split].contiguous()
        t_past = timepoints[:, :T_split].contiguous()
        t_fut = timepoints[:, T_split:].contiguous()

        out = model.forecast(
            x_hist=x_hist,
            x_mask=x_mask,
            past_time=t_past,
            future_time=t_fut,
            num_samples=int(num_samples),
        )

    return {
        "observed": observed.cpu().numpy(),
        "recon": recons.cpu().numpy(),
        "pred_mean": out["pred_mean"].cpu().numpy(),
        "pred_samples": out["pred_samples"].cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Plot 1: 1-D forecast panel (one row per sample).
# ---------------------------------------------------------------------------


@register_plot("forecast_1d")
def plot_forecast_1d(
    ctx: PlotContext,
    save_path: str,
    *,
    sample_indices: list[int] | None = None,
    n_show: int = 8,
    time_start_at_zero: bool = False,
    show_title: bool = False,
    font_size: int = 18,
) -> None:
    """One subplot per sample showing observed / recon / forecast samples + mean."""
    if ctx.T_split is None:
        raise ValueError("plot_forecast_1d requires PlotContext.T_split.")

    batch = _gather_batch(ctx, sample_indices)
    arrs = _run_recon_and_forecast(ctx, batch, int(ctx.T_split), ctx.num_samples)

    observed = arrs["observed"]
    recon = arrs["recon"]
    pred_mean = arrs["pred_mean"]
    pred_samples = arrs["pred_samples"]
    B = (
        min(n_show, observed.shape[0])
        if sample_indices is None
        else len(sample_indices)
    )
    T_split = int(ctx.T_split)
    T_total = observed.shape[-1]
    x_obs = np.arange(T_total) if time_start_at_zero else np.arange(1, T_total + 1)

    plt.rcParams.update({"font.size": font_size})
    fig, axes = plt.subplots(B, 1, figsize=(12, 4 * B), sharex=False, squeeze=False)
    axes = axes.flatten()

    for i in range(B):
        ax = axes[i]
        ax.plot(x_obs, observed[i, 0, :], "k-", label="Observed", alpha=0.6)
        ax.plot(x_obs, recon[i, 0, :], "b--", label="Reconstruction", alpha=0.8)

        last_obs_idx = T_split - 1
        x_last = x_obs[last_obs_idx]
        x_fut = np.concatenate([[x_last], x_obs[T_split:]])
        last_val = observed[i, 0, last_obs_idx]

        for s in range(pred_samples.shape[1]):
            y_s = np.concatenate([[last_val], pred_samples[i, s, 0, :]])
            ax.plot(x_fut, y_s, color="red", alpha=0.15, linewidth=1)
        y_mean = np.concatenate([[last_val], pred_mean[i, 0, :]])
        ax.plot(x_fut, y_mean, "r-", label="Forecast Mean", linewidth=2)
        ax.axvline(x=x_last, color="gray", linestyle=":", label="Context Split")
        if i == 0:
            ax.legend(fontsize=font_size, loc="upper left")
        if show_title:
            ax.set_title(
                f"Sample {sample_indices[i] if sample_indices else i}",
                fontsize=font_size,
            )
        ax.set_ylabel("Value", fontsize=font_size + 3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


# ---------------------------------------------------------------------------
# Plot 2: 2-D spatial trajectory.
# ---------------------------------------------------------------------------


@register_plot("forecast_2d_spatial")
def plot_forecast_2d_spatial(
    ctx: PlotContext,
    save_path: str,
    *,
    sample_indices: list[int] | None = None,
    n_show: int = 8,
    obstacle_box: tuple[float, float, float, float] | None = (-0.6, -0.6, 1.2, 1.2),
    xlim: tuple[float, float] = (-2.2, 2.5),
    ylim: tuple[float, float] = (-2.0, 2.0),
    font_size: int = 18,
) -> None:
    """X-vs-Y trajectory plot with sample paths and forecast mean."""
    if ctx.T_split is None:
        raise ValueError("plot_forecast_2d_spatial requires PlotContext.T_split.")

    batch = _gather_batch(ctx, sample_indices)
    arrs = _run_recon_and_forecast(ctx, batch, int(ctx.T_split), ctx.num_samples)
    observed = arrs["observed"]
    recon = arrs["recon"]
    pred_mean = arrs["pred_mean"]
    pred_samples = arrs["pred_samples"]
    if observed.shape[1] < 2:
        raise ValueError("forecast_2d_spatial expects D >= 2.")

    B = (
        min(n_show, observed.shape[0])
        if sample_indices is None
        else len(sample_indices)
    )
    T_split = int(ctx.T_split)

    plt.rcParams.update({"font.size": font_size})
    fig, axes = plt.subplots(B, 1, figsize=(12, 6 * B), squeeze=False)
    axes = axes.flatten()

    import matplotlib.patches as patches

    for i in range(B):
        ax = axes[i]
        if obstacle_box is not None:
            x0, y0, w, h = obstacle_box
            rect = patches.Rectangle(
                (x0, y0),
                w,
                h,
                linewidth=1,
                edgecolor="black",
                facecolor="gray",
                alpha=0.3,
                label="Obstacle",
            )
            ax.add_patch(rect)
        ax.plot(
            observed[i, 0, :],
            observed[i, 1, :],
            "k-",
            label="Observed",
            alpha=0.6,
            marker=".",
            markersize=3,
        )
        ax.plot(
            recon[i, 0, :], recon[i, 1, :], "b--", label="Reconstruction", alpha=0.7
        )
        ax.plot(
            observed[i, 0, T_split - 1],
            observed[i, 1, T_split - 1],
            "go",
            label="Context End",
            markersize=8,
        )
        for s in range(pred_samples.shape[1]):
            xs = np.concatenate([
                [observed[i, 0, T_split - 1]],
                pred_samples[i, s, 0, :],
            ])
            ys = np.concatenate([
                [observed[i, 1, T_split - 1]],
                pred_samples[i, s, 1, :],
            ])
            ax.plot(xs, ys, color="red", alpha=0.15, linewidth=1)
        xs_mean = np.concatenate([[observed[i, 0, T_split - 1]], pred_mean[i, 0, :]])
        ys_mean = np.concatenate([[observed[i, 1, T_split - 1]], pred_mean[i, 1, :]])
        ax.plot(xs_mean, ys_mean, "r-", label="Forecast Mean", linewidth=2)
        ax.set_aspect("equal", "box")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


# ---------------------------------------------------------------------------
# Plot 3: forecast-distribution histogram at one (series, dim, t_future) point.
# ---------------------------------------------------------------------------


@register_plot("forecast_distribution")
def plot_forecast_distribution(
    ctx: PlotContext,
    save_path: str,
    *,
    series_idx: int = 0,
    dim_idx: int = 0,
    t_future_idx: int = 0,
    n_bins: int = 50,
    title: str | None = None,
) -> None:
    """Histogram of forecast samples at one ``(series, dim, t_future)`` point.

    Picks a single batch element + dimension + future timestep and draws a
    density histogram of the ``num_samples`` forecast draws, plus vertical
    lines for the truth and the forecast mean. Useful for inspecting whether
    the predictive distribution is unimodal/multimodal at a specific horizon
    (the canonical bimodal-diff vs bimodal-gauss inspection).
    """
    if ctx.T_split is None:
        raise ValueError("plot_forecast_distribution requires PlotContext.T_split.")
    batch = _gather_batch(ctx, sample_indices=None)
    arrs = _run_recon_and_forecast(ctx, batch, int(ctx.T_split), ctx.num_samples)

    pred_samples = arrs["pred_samples"]  # (B, S, D, L2)
    pred_mean = arrs["pred_mean"]  # (B, D, L2)
    observed = arrs["observed"]  # (B, D, T_total)
    T_split = int(ctx.T_split)

    if not (0 <= series_idx < pred_samples.shape[0]):
        raise IndexError(
            f"series_idx={series_idx} out of range [0, {pred_samples.shape[0]})"
        )
    if not (0 <= dim_idx < pred_samples.shape[2]):
        raise IndexError(f"dim_idx={dim_idx} out of range [0, {pred_samples.shape[2]})")
    if not (0 <= t_future_idx < pred_samples.shape[3]):
        raise IndexError(
            f"t_future_idx={t_future_idx} out of range [0, {pred_samples.shape[3]})"
        )

    vals = pred_samples[series_idx, :, dim_idx, t_future_idx]
    mu = float(pred_mean[series_idx, dim_idx, t_future_idx])
    y = float(observed[series_idx, dim_idx, T_split + t_future_idx])

    fig = plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=n_bins, density=True, alpha=0.6, label="forecast samples")
    plt.axvline(y, color="red", linestyle="--", linewidth=2, label=f"truth={y:.3f}")
    plt.axvline(mu, color="black", linestyle="-", linewidth=2, label=f"mean={mu:.3f}")
    if title is None:
        title = f"series={series_idx} dim={dim_idx} t+{t_future_idx + 1}"
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: training-metric curves from CSV (no model needed).
# ---------------------------------------------------------------------------


@register_plot("metrics_csv")
def plot_metrics_csv(
    ctx: PlotContext,
    save_path: str,
    *,
    keys: list[str] | None = None,
    log_y: bool = False,
) -> None:
    """Line plots of selected CSV columns versus step."""
    if not ctx.csv_path:
        raise ValueError("plot_metrics_csv requires PlotContext.csv_path.")
    keys = keys or ["loss/total"]

    series: dict[str, list[float]] = {k: [] for k in keys}
    steps: list[int] = []
    with open(ctx.csv_path) as f:
        reader = _csv.DictReader(f)
        valid = [k for k in keys if reader.fieldnames and k in reader.fieldnames]
        if not valid:
            raise ValueError(f"None of {keys} found in {ctx.csv_path}")
        for row in reader:
            try:
                s = int(row["step"])
            except (KeyError, ValueError):
                continue
            try:
                vals = [float(row[k]) for k in valid]
            except (KeyError, ValueError):
                continue
            steps.append(s)
            for k, v in zip(valid, vals):
                series[k].append(v)

    plt.figure(figsize=(10, 6))
    for k in valid:
        plt.plot(steps, series[k], label=k)
    if log_y:
        plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
