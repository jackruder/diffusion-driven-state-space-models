"""Variance plot registry for probe outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ProbePlotContext:
    """Inputs available to every probe-plot function.

    Attributes:
        rows: Per-sample probe rows.
        summary: Probe-loop summary dict.
        metrics: Computed metric outputs keyed by metric name.
    """

    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    metrics: dict[str, Any]


ProbePlotFn = Callable[..., None]
PROBE_PLOT_REGISTRY: dict[str, ProbePlotFn] = {}


def register_probe_plot(name: str) -> Callable[[ProbePlotFn], ProbePlotFn]:
    """Decorator registering a probe-plot function under ``name``.

    Args:
        name: Registry key a ``ProbeSpec`` uses to select this plot.

    Returns:
        The decorator, which returns the function unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def _wrap(fn: ProbePlotFn) -> ProbePlotFn:
        if name in PROBE_PLOT_REGISTRY:
            raise ValueError(f"Probe plot {name!r} already registered.")
        PROBE_PLOT_REGISTRY[name] = fn
        return fn

    return _wrap


# Shared visual scheme so the three plots in this module are
# interpretable side-by-side:
#
#   Colour → objective       (ESM = blue, DSM = orange)
#   Style  → k-sampling mode (uniform = solid, lsgm_is = dashed)
#
# When ``force_per_k=True`` the k-sampling mode is unused (k is
# forced), so the two style variants land on top of each other — the
# dashed pattern showing through the solid line is the visual cue
# that the pair is degenerate.

_OBJ_COLOR = {"esm": "#1f77b4", "dsm": "#ff7f0e"}
_MODE_STYLE = {
    "uniform": "-",
    "lsgm_is": "--",
    "adaptive_is": "-.",
    "adaptive_is_full": ":",
}
_KIND_COLOR = {"loss": "#2ca02c", "grad": "#d62728"}


def _cell_style(cell: str) -> tuple[str, str]:
    objective, _, mode = cell.partition(":")
    return _OBJ_COLOR.get(objective, "#444444"), _MODE_STYLE.get(mode, ":")


def _plot_empty(save_path: str, message: str, figsize=(8, 5)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _plot_var_per_k(
    ctx: ProbePlotContext,
    save_path: str,
    *,
    metric_key: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Draw one log-scale variance-vs-τ curve per cell from ``metric_key``."""
    var = ctx.metrics.get(metric_key, {})
    if not var:
        _plot_empty(save_path, f"No data for {metric_key!r}")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    # Sort so DSM (high-variance) is drawn first and ESM appears on top.
    cells_sorted = sorted(
        var.keys(),
        key=lambda c: (
            0 if c.split(":")[0] == "dsm" else 1,
            0 if c.endswith("uniform") else 1,
        ),
    )
    for cell in cells_sorted:
        kvals = var[cell]
        if not kvals:
            continue
        items = sorted(kvals.items(), key=lambda kv: int(kv[0]))
        xs = np.array([int(k) for k, _ in items], dtype=np.int64)
        ys = np.array([float(v) for _, v in items], dtype=np.float64)
        colour, style = _cell_style(cell)
        ax.plot(
            xs, ys,
            color=colour, linestyle=style,
            linewidth=1.5, alpha=0.85,
            label=cell,
        )
    ax.set_xlabel(
        r"$\tau$-bin index $k$    (small $k$ → low noise, large $k$ → high noise)"
    )
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    leg = ax.legend(fontsize=8, title="objective : k-sampling", loc="best")
    leg.get_title().set_fontsize(8)
    # Tag the pair-equivalence on the figure so an overlap is read as
    # information ("modes don't differ under forced k") rather than a
    # bug.
    fig.text(
        0.01, 0.01,
        "Under forced k the k-sampling mode is not exercised — "
        "uniform / lsgm_is curves coincide.",
        fontsize=7, style="italic", color="dimgrey",
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(save_path)
    plt.close(fig)


@register_probe_plot("var_grad_vs_tau")
def plot_var_grad_vs_tau(
    ctx: ProbePlotContext, save_path: str,
    *, ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    **_: Any,
) -> None:
    """Gradient variance per τ-bin (forced k), one curve per cell."""
    _plot_var_per_k(
        ctx, save_path,
        metric_key="grad_var_per_tau",
        title=r"Gradient variance per $\tau$-bin (forced $k$)",
        ylabel=(
            r"$\overline{\mathrm{Var}_{\mathrm{seed,batch}}}\,"
            r"(\nabla_\theta \mathcal{L}_p)$"
            "\n(mean across score-net parameters)"
        ),
        ylim=ylim, xlim=xlim,
    )


@register_probe_plot("var_loss_vs_tau")
def plot_var_loss_vs_tau(
    ctx: ProbePlotContext, save_path: str,
    *, ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    **_: Any,
) -> None:
    """Loss variance per τ-bin (forced k), one curve per cell."""
    _plot_var_per_k(
        ctx, save_path,
        metric_key="loss_var_per_tau",
        title=r"Loss variance per $\tau$-bin (forced $k$)",
        ylabel=r"$\mathrm{Var}_{\mathrm{seed,batch}}\,(\mathcal{L}_p)$",
        ylim=ylim, xlim=xlim,
    )


@register_probe_plot("ratio_vs_tau")
def plot_ratio_vs_tau(
    ctx: ProbePlotContext, save_path: str,
    *, ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    **_: Any,
) -> None:
    """Per-k ESM/DSM variance ratio (loss + grad), as a function of τ."""
    rpt = ctx.metrics.get("ratio_per_tau", {})
    if not rpt or not (rpt.get("loss") or rpt.get("grad")):
        _plot_empty(save_path, "No ratio_per_tau data available")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    any_line = False
    for kind in ("loss", "grad"):
        per_mode = rpt.get(kind, {})
        for mode in ("uniform", "lsgm_is", "adaptive_is", "adaptive_is_full"):
            kvals = per_mode.get(mode, {})
            if not kvals:
                continue
            items = sorted(kvals.items(), key=lambda kv: int(kv[0]))
            xs = np.array([int(k) for k, _ in items], dtype=np.int64)
            ys = np.array([float(v) for _, v in items], dtype=np.float64)
            ax.plot(
                xs, ys,
                color=_KIND_COLOR[kind],
                linestyle=_MODE_STYLE[mode],
                linewidth=1.5, alpha=0.85,
                label=f"{kind} ratio ({mode})",
            )
            any_line = True

    if not any_line:
        _plot_empty(save_path, "No finite ratio values to plot")
        return

    ax.axhline(
        1.0, color="grey", linestyle=":", linewidth=1.2, alpha=0.7,
        label="parity (ESM = DSM)",
    )
    ax.set_xlabel(r"$\tau$-bin index $k$")
    ax.set_ylabel("ESM / DSM variance ratio")
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(r"ESM vs DSM variance ratio per $\tau$-bin")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.text(
        0.01, 0.01,
        "< 1 → ESM has lower variance at this k;   "
        "> 1 → DSM has lower variance.    "
        "uniform / lsgm_is coincide under forced k.",
        fontsize=7, style="italic", color="dimgrey",
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(save_path)
    plt.close(fig)


@register_probe_plot("summary_table")
def plot_summary_table(ctx: ProbePlotContext, save_path: str, **_: Any) -> None:
    """Render the probe summary dict as a monospace text panel."""
    text = json.dumps(ctx.summary, indent=2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(0.01, 0.99, text, family="monospace", va="top", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


@register_probe_plot("var_grad_vs_step")
def plot_var_grad_vs_step(ctx: ProbePlotContext, save_path: str, **_: Any) -> None:
    """Placeholder for the R2 checkpoint-sweep grad-variance-vs-step plot."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, "R2 checkpoint-sweep plot placeholder",
            ha="center", va="center")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
