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


def _plot_var_per_tau(ctx: ProbePlotContext, save_path: str, value_key: str) -> None:
    var = ctx.metrics.get("var_per_tau", {})
    fig, ax = plt.subplots(figsize=(8, 5))
    for cell, kvals in sorted(var.items()):
        xs = np.array([int(k) for k in kvals.keys()], dtype=np.int64)
        ys = np.array([float(v) for v in kvals.values()], dtype=np.float64)
        if value_key == "ratio":
            continue
        ax.plot(xs, ys, label=cell)
    ax.set_xlabel("k index")
    ax.set_ylabel("Variance")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


@register_probe_plot("var_grad_vs_tau")
def plot_var_grad_vs_tau(ctx: ProbePlotContext, save_path: str, **_: Any) -> None:
    _plot_var_per_tau(ctx, save_path, "grad")


@register_probe_plot("var_loss_vs_tau")
def plot_var_loss_vs_tau(ctx: ProbePlotContext, save_path: str, **_: Any) -> None:
    _plot_var_per_tau(ctx, save_path, "loss")


@register_probe_plot("ratio_vs_tau")
def plot_ratio_vs_tau(ctx: ProbePlotContext, save_path: str, **_: Any) -> None:
    ratio = ctx.metrics.get("ratio_esm_dsm", {})
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(ratio.get("loss", {})))
    labels = list(ratio.get("loss", {}).keys())
    loss_vals = [ratio["loss"][k] for k in labels]
    grad_vals = [ratio.get("grad", {}).get(k, np.nan) for k in labels]
    ax.plot(xs, loss_vals, marker="o", label="loss ratio")
    ax.plot(xs, grad_vals, marker="x", label="grad ratio")
    ax.set_xticks(xs, labels)
    ax.set_yscale("log")
    ax.legend()
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
    ax.text(0.5, 0.5, "R2 checkpoint-sweep plot placeholder", ha="center", va="center")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
