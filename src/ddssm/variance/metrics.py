"""Variance metric registry for the variance probe stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

MIN_DIVISOR = 1e-12


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
    """Across-replica variance of the loss ESTIMATOR (per cell).

    Replica rows are emitted one-per-MC-sample, so pooling their per-sample
    ``L_p`` and taking ``np.var`` would report within-batch sample spread, not
    the variance of the loss estimator across independent replicas. The right
    quantity is ``L_p_scalar`` — the per-replica batch-mean loss, identical
    across a replica's sample rows — varied over replicas. This mirrors the
    sibling ``grad_var`` (estimator variance of the gradient) and is what makes
    the ESM/DSM ratio in :func:`metric_ratio_esm_dsm` meaningful.
    """
    rows = _replica_rows(ctx)
    by_cell: dict[str, dict[tuple, float]] = {}
    for r in rows:
        key = f"{r['objective']}:{r['k_sampling_mode']}"
        replica = (r["seed"], r["batch_idx"], r["replica"])
        by_cell.setdefault(key, {})[replica] = float(
            r.get("L_p_scalar", r["L_p"])
        )
    return {
        "loss_var": {
            k: float(np.var(list(v.values()))) for k, v in by_cell.items()
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
        d_loss = loss_var.get(d_key, np.nan)
        d_grad = grad_var.get(d_key, np.nan)
        out["loss"][mode] = (
            float(np.nan) if not np.isfinite(d_loss) else float(loss_var.get(e_key, np.nan) / max(d_loss, MIN_DIVISOR))
        )
        out["grad"][mode] = (
            float(np.nan) if not np.isfinite(d_grad) else float(grad_var.get(e_key, np.nan) / max(d_grad, MIN_DIVISOR))
        )
    return {"ratio_esm_dsm": out}


def _loss_var_per_tau(ctx: ProbeContext) -> dict[str, dict[str, float]]:
    """Per-(cell, k) variance of the per-batch loss mean across seeds/batches.

    Returns NaN where fewer than two samples are available (otherwise
    ``np.var`` silently reports 0, which looks like a confidently flat
    estimator instead of an undersampled one).
    """
    rows = [r for r in ctx.rows if r["kind"] == "forced_k"]
    bucket: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        key = f"{r['objective']}:{r['k_sampling_mode']}"
        bucket.setdefault(key, {}).setdefault(int(r["k_idx"]), []).append(float(r["L_p"]))
    out: dict[str, dict[str, float]] = {}
    for cell, kmap in bucket.items():
        out[cell] = {
            str(k): float(np.var(vals)) if len(vals) >= 2 else float("nan")
            for k, vals in sorted(kmap.items())
        }
    return out


@register_probe_metric("loss_var_per_tau")
def metric_loss_var_per_tau(ctx: ProbeContext) -> dict[str, Any]:
    return {"loss_var_per_tau": _loss_var_per_tau(ctx)}


# Back-compat alias — old configs referenced ``var_per_tau``. Drops in
# the same value under the legacy key.
@register_probe_metric("var_per_tau")
def metric_var_per_tau(ctx: ProbeContext) -> dict[str, Any]:
    return {"var_per_tau": _loss_var_per_tau(ctx)}


@register_probe_metric("grad_var_per_tau")
def metric_grad_var_per_tau(ctx: ProbeContext) -> dict[str, Any]:
    """Per-(cell, k) gradient variance, populated by the force_per_k loop."""
    raw = ctx.summary.get("per_k_grad_var", {})
    out: dict[str, dict[str, float]] = {}
    for cell, kmap in raw.items():
        out[cell] = {str(k): float(v) for k, v in sorted(kmap.items())}
    return {"grad_var_per_tau": out}


@register_probe_metric("ratio_per_tau")
def metric_ratio_per_tau(ctx: ProbeContext) -> dict[str, Any]:
    """Per-k ESM/DSM ratio for both loss and gradient variances.

    This is the "vs τ" version of ``ratio_esm_dsm`` — instead of one
    scalar per (kind, mode) it returns ``{kind: {mode: {k: ratio}}}``.
    """
    loss_pt = _loss_var_per_tau(ctx)
    grad_raw = ctx.summary.get("per_k_grad_var", {})
    grad_pt = {
        cell: {str(k): float(v) for k, v in kmap.items()}
        for cell, kmap in grad_raw.items()
    }
    out: dict[str, dict[str, dict[str, float]]] = {"loss": {}, "grad": {}}
    for mode in ("uniform", "lsgm_is"):
        e_key = f"esm:{mode}"
        d_key = f"dsm:{mode}"
        for kind, src in (("loss", loss_pt), ("grad", grad_pt)):
            if e_key not in src or d_key not in src:
                continue
            ek = src[e_key]
            dk = src[d_key]
            ratios: dict[str, float] = {}
            for k_str, dv in dk.items():
                ev = ek.get(k_str, float("nan"))
                if not np.isfinite(dv) or dv == 0 or not np.isfinite(ev):
                    ratios[k_str] = float("nan")
                else:
                    ratios[k_str] = float(ev / dv)
            out[kind][mode] = dict(sorted(
                ratios.items(), key=lambda kv: int(kv[0])
            ))
    return {"ratio_per_tau": out}
