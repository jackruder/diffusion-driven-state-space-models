"""Variance metric registry for the variance probe stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class ProbeContext:
    model: Any
    transitions: dict[str, Any]
    loader: Any
    device: Any
    spec: Any
    run_dir: str
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


ProbeMetricFn = Callable[..., dict[str, Any]]
PROBE_METRIC_REGISTRY: dict[str, ProbeMetricFn] = {}


def register_probe_metric(name: str) -> Callable[[ProbeMetricFn], ProbeMetricFn]:
    def _wrap(fn: ProbeMetricFn) -> ProbeMetricFn:
        if name in PROBE_METRIC_REGISTRY:
            raise ValueError(f"Probe metric {name!r} already registered.")
        PROBE_METRIC_REGISTRY[name] = fn
        return fn

    return _wrap


def _replica_rows(ctx: ProbeContext) -> list[dict[str, Any]]:
    return [r for r in ctx.rows if r["kind"] == "replica"]


@register_probe_metric("loss_var")
def metric_loss_var(ctx: ProbeContext) -> dict[str, Any]:
    rows = _replica_rows(ctx)
    by_cell: dict[str, list[float]] = {}
    for r in rows:
        key = f"{r['objective']}:{r['k_sampling_mode']}"
        by_cell.setdefault(key, []).append(float(r["L_p"]))
    return {
        "loss_var": {
            k: float(np.var(v)) for k, v in by_cell.items()
        }
    }


@register_probe_metric("grad_var")
def metric_grad_var(ctx: ProbeContext) -> dict[str, Any]:
    return {
        "grad_var": {
            k: float(v.get("grad_variance", np.nan))
            for k, v in ctx.summary.get("cells", {}).items()
        }
    }


@register_probe_metric("ratio_esm_dsm")
def metric_ratio_esm_dsm(ctx: ProbeContext) -> dict[str, Any]:
    loss_var = metric_loss_var(ctx)["loss_var"]
    grad_var = metric_grad_var(ctx)["grad_var"]
    out: dict[str, Any] = {"loss": {}, "grad": {}}
    for mode in ("uniform", "lsgm_is"):
        e_key = f"esm:{mode}"
        d_key = f"dsm:{mode}"
        out["loss"][mode] = float(loss_var.get(e_key, np.nan) / max(loss_var.get(d_key, np.nan), 1e-12))
        out["grad"][mode] = float(grad_var.get(e_key, np.nan) / max(grad_var.get(d_key, np.nan), 1e-12))
    return {"ratio_esm_dsm": out}


@register_probe_metric("var_per_tau")
def metric_var_per_tau(ctx: ProbeContext) -> dict[str, Any]:
    rows = [r for r in ctx.rows if r["kind"] == "forced_k"]
    bucket: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        key = f"{r['objective']}:{r['k_sampling_mode']}"
        bucket.setdefault(key, {}).setdefault(int(r["k_idx"]), []).append(float(r["L_p"]))
    out: dict[str, Any] = {}
    for cell, kmap in bucket.items():
        out[cell] = {str(k): float(np.var(vals)) for k, vals in sorted(kmap.items())}
    return {"var_per_tau": out}
