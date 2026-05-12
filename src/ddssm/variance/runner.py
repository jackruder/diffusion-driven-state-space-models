"""Variance runner: executes probe, metrics, and plot registries."""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from .metrics import PROBE_METRIC_REGISTRY, ProbeContext
from .plots import PROBE_PLOT_REGISTRY, ProbePlotContext
from .probe import _select_loader, run_probe

log = logging.getLogger(__name__)


@dataclass
class ProbeCell:
    objective: str
    k_sampling_mode: str


@dataclass
class ProbeMetricSpec:
    name: str
    kwargs: Any = field(default_factory=dict)


@dataclass
class ProbePlotSpec:
    name: str
    save_filename: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeSpec:
    # Keep as untyped ``list`` so OmegaConf/hydra-zen accepts both ProbeCell
    # and Builds_ProbeCell instances in structured configs.
    cells: list = field(default_factory=lambda: [
        ProbeCell("esm", "uniform"),
        ProbeCell("dsm", "uniform"),
        ProbeCell("esm", "lsgm_is"),
        ProbeCell("dsm", "lsgm_is"),
    ])
    R: int = 128
    B_var: int = 16
    n_batches: int = 1
    K_bins: int = 20
    force_per_k: bool = True
    split: str = "train"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    freeze: list[str] = field(default_factory=lambda: ["encoder", "decoder", "zinit", "embed_layer"])
    # Same rationale as ``cells`` above.
    metrics: list = field(default_factory=lambda: [
        ProbeMetricSpec("loss_var"),
        ProbeMetricSpec("grad_var"),
        ProbeMetricSpec("ratio_esm_dsm"),
        ProbeMetricSpec("loss_var_per_tau"),
        ProbeMetricSpec("grad_var_per_tau"),
    ])
    # Same rationale as ``cells`` above.
    plots: list = field(default_factory=lambda: [
        ProbePlotSpec("var_grad_vs_tau"),
        ProbePlotSpec("var_loss_vs_tau"),
        ProbePlotSpec("ratio_vs_tau"),
        ProbePlotSpec("summary_table"),
    ])
    raw_filename: str = "variance_raw.csv"
    summary_filename: str = "variance_summary.json"
    checkpoint_path: str | None = None


def _write_rows(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def variance(
    experiment,
    spec: ProbeSpec,
    *,
    device: torch.device,
    run_dir: str,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)
    ckpt = checkpoint_path or spec.checkpoint_path
    log.info("Variance probe → %s (checkpoint=%s)", run_dir, ckpt)
    log.info("Stage 1/3: running probe loop")
    rows, summary, transitions = run_probe(
        experiment,
        spec,
        device=device,
        checkpoint_path=ckpt,
    )
    raw_path = os.path.join(run_dir, spec.raw_filename)
    _write_rows(rows, raw_path)
    log.info("Wrote raw CSV → %s (%d rows)", raw_path, len(rows))

    loader = _select_loader(experiment, spec.split)

    ctx = ProbeContext(
        model=experiment.model,
        transitions=transitions,
        loader=loader,
        device=device,
        spec=spec,
        run_dir=run_dir,
        rows=rows,
        summary=summary,
    )

    log.info("Stage 2/3: computing %d metric(s)", len(spec.metrics))
    metric_out: dict[str, Any] = {}
    for metric in spec.metrics:
        if metric.name not in PROBE_METRIC_REGISTRY:
            raise KeyError(f"Unknown probe metric {metric.name!r}")
        log.info("  metric %s", metric.name)
        kwargs = dict(metric.kwargs or {})
        metric_out.update(PROBE_METRIC_REGISTRY[metric.name](ctx, **kwargs))

    summary_out = {
        "summary": summary,
        "metrics": metric_out,
        "raw_csv": os.path.basename(raw_path),
        "checkpoint_path": ckpt,
    }
    summary_path = os.path.join(run_dir, spec.summary_filename)
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2, default=float)
    log.info("Wrote summary JSON → %s", summary_path)

    log.info("Stage 3/3: generating %d plot(s)", len(spec.plots))
    plot_ctx = ProbePlotContext(rows=rows, summary=summary, metrics=metric_out)
    for plot in spec.plots:
        if plot.name not in PROBE_PLOT_REGISTRY:
            raise KeyError(f"Unknown probe plot {plot.name!r}")
        out_name = plot.save_filename or f"{plot.name}.png"
        out_path = os.path.join(run_dir, out_name)
        PROBE_PLOT_REGISTRY[plot.name](plot_ctx, out_path, **dict(plot.kwargs or {}))
        log.info("  saved %s", out_path)
    log.info("Variance probe complete.")
    return summary_out
