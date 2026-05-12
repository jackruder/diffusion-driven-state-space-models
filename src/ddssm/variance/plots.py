"""Variance plot registry for probe outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ProbePlotContext:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    metrics: dict[str, Any]


ProbePlotFn = Callable[..., None]
PROBE_PLOT_REGISTRY: dict[str, ProbePlotFn] = {}


def register_probe_plot(name: str) -> Callable[[ProbePlotFn], ProbePlotFn]:
    def _wrap(fn: ProbePlotFn) -> ProbePlotFn:
        if name in PROBE_PLOT_REGISTRY:
            raise ValueError(f"Probe plot {name!r} already registered.")
        PROBE_PLOT_REGISTRY[name] = fn
        return fn

    return _wrap


def _plot_var_per_k(
    ctx: ProbePlotContext,
    save_path: str,
    *,
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    var = ctx.metrics.get(metric_key, {})
    fig, ax = plt.subplots(figsize=(8, 5))
    if not var:
        ax.text(
            0.5, 0.5,
            f"No data for {metric_key!r}",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        return
    for cell, kvals in sorted(var.items()):
        xs = np.array([int(k) for k in kvals.keys()], dtype=np.int64)
        ys = np.array([float(v) for v in kvals.values()], dtype=np.float64)
        ax.plot(xs, ys, marker=".", markersize=4, linewidth=1, label=cell)
    ax.set_xlabel(r"$\tau$-bin index $k$  (small $k$ → low noise, large $k$ → high noise)")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, title="objective : k-sampling", loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


@register_probe_plot("var_grad_vs_tau")
def plot_var_grad_vs_tau(ctx: ProbePlotContext, save_path: str, **_: Any) -> None:
    _plot_var_per_k(
        ctx,
        save_path,
        metric_key="grad_var_per_tau",
        title=r"Gradient variance per $\tau$-bin (forced $k$)",
        ylabel=r"$\mathrm{Var}_{\mathrm{seed,batch}}\,(\nabla_\theta \mathcal{L}_p)$"
               "\n(mean over score-net parameters)",
    )


@register_probe_plot("var_loss_vs_tau")
def plot_var_loss_vs_tau(ctx: ProbePlotContext, save_path: str, **_: Any) -> None:
    _plot_var_per_k(
        ctx,
        save_path,
        metric_key="loss_var_per_tau",
        title=r"Loss variance per $\tau$-bin (forced $k$)",
        ylabel=r"$\mathrm{Var}_{\mathrm{seed,batch}}\,(\mathcal{L}_p)$"
               "\n(per-batch loss mean)",
    )


@register_probe_plot("ratio_vs_tau")
def plot_ratio_vs_tau(ctx: ProbePlotContext, save_path: str, **_: Any) -> None:
    """ESM / DSM variance ratio, per k-sampling mode.

    A horizontal bar chart with the ratio on the x-axis (log scale) and
    the k-sampling mode on the y-axis. A vertical dashed line at 1.0
    marks parity; values < 1 mean ESM has lower variance than DSM.
    """
    ratio = ctx.metrics.get("ratio_esm_dsm", {})
    labels = list(ratio.get("loss", {}).keys())   # ['uniform', 'lsgm_is']
    loss_vals = np.array([ratio["loss"][k] for k in labels], dtype=float)
    grad_vals = np.array(
        [ratio.get("grad", {}).get(k, np.nan) for k in labels], dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(labels))
    bar_h = 0.35
    ax.barh(y - bar_h / 2, loss_vals, height=bar_h,
            label="Loss-variance ratio", color="C0")
    ax.barh(y + bar_h / 2, grad_vals, height=bar_h,
            label="Grad-variance ratio", color="C1")
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1,
               alpha=0.7, label="parity (ESM = DSM)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel(
        "ESM / DSM variance ratio  (log scale)\n"
        "< 1  → ESM has lower variance;  > 1  → DSM has lower variance"
    )
    ax.set_ylabel("k-sampling mode")
    ax.set_title("ESM vs DSM variance ratio, by k-sampling mode")
    ax.grid(True, which="both", axis="x", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


@register_probe_plot("summary_table")
def plot_summary_table(ctx: ProbePlotContext, save_path: str, **_: Any) -> None:
    text = json.dumps(ctx.summary, indent=2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(0.01, 0.99, text, family="monospace", va="top", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


@register_probe_plot("var_grad_vs_step")
def plot_var_grad_vs_step(ctx: ProbePlotContext, save_path: str, **_: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, "R2 checkpoint-sweep plot placeholder",
            ha="center", va="center")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
