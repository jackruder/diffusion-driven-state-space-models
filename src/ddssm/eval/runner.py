"""Evaluation runner: walks an EvalSpec, writes metrics.json.

Loads a checkpoint into the experiment's already-built model, picks
the right loader from the data module (``test`` by default), and
invokes each metric named in :class:`EvalSpec.metrics`. Results are
merged into a single dict and persisted to ``metrics.json`` under the
Hydra run dir.

Train, evaluate, and visualize are independent stages -- this file is
not invoked by training. Use ``python -m ddssm.evaluate`` to drive it
from the CLI.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from .metrics import EvalContext, METRIC_REGISTRY

log = logging.getLogger(__name__)


@dataclass
class EvalSpec:
    """What to compute, on which split, with what defaults.

    Attributes:
        metrics: Names of registered metrics (see ``METRIC_REGISTRY``).
        split: Which DataModule loader to evaluate on
            (``"train"`` / ``"val"`` / ``"test"``).
        num_samples: Forecast sample count for sample-based metrics.
        T_split: Forecast split index for forecast-based metrics. If
            ``None``, falls back to ``data.metadata.forecast_split``;
            if that is also ``None`` the metrics that need it raise.
        output_filename: File name (relative to the run dir) for the
            JSON dump.
        kwargs: Per-metric keyword overrides keyed by metric name. The
            inner dict is forwarded as ``**kwargs`` to that metric.
            Example: ``{"bimodal_jsd": {"npz_path": "out.npz"}}``.
    """

    metrics: list[str] = field(default_factory=lambda: ["loss_tail"])
    split: str = "test"
    num_samples: int = 1
    T_split: int | None = None
    output_filename: str = "metrics.json"
    # ``Any`` (not ``dict`` or ``dict[str, Any]``) is required here.
    # OmegaConf sets struct=True on DictConfig values inside structured
    # configs regardless of the dict key/value type parameters.  Only
    # ``Any`` fully escapes struct checking, letting CLI overrides like
    # ``experiment.eval.kwargs={metric: {key: val}}`` add arbitrary keys.
    kwargs: Any = field(default_factory=dict)


def evaluate(
    experiment,
    spec: EvalSpec,
    *,
    device: torch.device,
    run_dir: str,
    checkpoint_path: str | None = None,
    csv_path: str | None = None,
) -> dict[str, Any]:
    """Run every metric named in ``spec`` and write the result to disk.

    Loads ``checkpoint_path`` into the experiment's model, selects the
    ``spec.split`` loader, walks ``spec.metrics``, merges their results,
    and writes them to ``spec.output_filename`` under ``run_dir``. If a
    W&B run can be resumed from ``run_dir``, the scalar results are also
    logged under the ``eval/`` namespace (best-effort).

    Args:
        experiment: The built :class:`Experiment` (model + data module).
        spec: Which metrics to compute, on which split, with defaults.
        device: Device to load the model onto and run metrics on.
        run_dir: Hydra run dir; the JSON is written here.
        checkpoint_path: Checkpoint to load; ``None`` uses the model as
            built (untrained weights).
        csv_path: Path to a training ``metrics.csv`` for CSV-derived
            metrics (e.g. ``loss_tail``).

    Returns:
        The merged metric-results dict (also persisted to disk).

    Raises:
        KeyError: If ``spec`` names a metric absent from
            ``METRIC_REGISTRY``.
    """
    from ..checkpoint import prepare_model

    model = prepare_model(
        experiment, checkpoint_path=checkpoint_path, device=device,
    )

    loader = experiment.data.loader(spec.split)
    T_split = experiment.data.metadata.forecast_split_or(spec.T_split)

    ctx = EvalContext(
        model=model,
        loader=loader,
        device=device,
        batch_transform=experiment.data.batch_transform,
        csv_path=csv_path,
        T_split=T_split,
        num_samples=int(spec.num_samples),
        run_dir=run_dir,
    )

    results: dict[str, Any] = {}
    for name in spec.metrics:
        if name not in METRIC_REGISTRY:
            raise KeyError(
                f"Unknown metric {name!r}. Registered metrics: "
                f"{sorted(METRIC_REGISTRY)}"
            )
        metric_kwargs = dict(spec.kwargs.get(name, {})) if spec.kwargs else {}
        log.info("Computing metric %s (kwargs=%s)", name, metric_kwargs)
        results.update(METRIC_REGISTRY[name](ctx, **metric_kwargs))

    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, spec.output_filename)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log.info("Wrote %s", out_path)

    # Cross-stage W&B reconnect: surface eval scalars on the original
    # training run under the ``eval/`` namespace and snapshot the JSON
    # as an artifact. Soft-fails so eval still returns the in-memory
    # dict even when W&B is unreachable.
    from ..loggers import resume_run_from_dir

    wandb_mod = resume_run_from_dir(
        run_dir, getattr(experiment, "wandb_config", None),
    )
    if wandb_mod is not None:
        scalar_payload: dict[str, float] = {}
        for k, v in results.items():
            try:
                scalar_payload[f"eval/{k}"] = float(v)
            except (TypeError, ValueError):
                continue
        try:
            if scalar_payload:
                wandb_mod.log(scalar_payload)
            artifact = wandb_mod.Artifact(name="eval-metrics", type="eval")
            artifact.add_file(out_path)
            wandb_mod.log_artifact(artifact)
        except Exception as e:  # noqa: BLE001 — best-effort
            log.warning("wandb eval log/upload failed: %s", e)
        try:
            wandb_mod.finish()
        except Exception:  # noqa: BLE001 — best-effort
            pass
    return results
