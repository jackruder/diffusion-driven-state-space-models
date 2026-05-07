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
    """

    metrics: list[str] = field(default_factory=lambda: ["loss_tail"])
    split: str = "test"
    num_samples: int = 1
    T_split: int | None = None
    output_filename: str = "metrics.json"


def _select_loader(experiment, split: str):
    if split == "train":
        return experiment.data.train_loader()
    if split == "val":
        return experiment.data.val_loader()
    if split == "test":
        return experiment.data.test_loader()
    raise ValueError(f"Unknown eval split: {split!r}")


def _resolve_T_split(spec: EvalSpec, experiment) -> int | None:
    if spec.T_split is not None:
        return int(spec.T_split)
    meta = getattr(experiment.data, "metadata", None)
    if meta is None:
        return None
    return getattr(meta, "forecast_split", None)


def _maybe_load_checkpoint(model: torch.nn.Module, ckpt_path: str | None, device: torch.device) -> None:
    if ckpt_path is None:
        log.warning("No checkpoint provided; evaluating randomly-initialised weights.")
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path!r}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state, strict=True)
    log.info("Loaded checkpoint from %s", ckpt_path)


def evaluate(
    experiment,
    spec: EvalSpec,
    *,
    device: torch.device,
    run_dir: str,
    checkpoint_path: str | None = None,
    csv_path: str | None = None,
) -> dict[str, Any]:
    """Run every metric named in ``spec`` and write the result to disk."""
    model = experiment.model.to(device)
    _maybe_load_checkpoint(model, checkpoint_path, device)
    model.eval()

    loader = _select_loader(experiment, spec.split)
    T_split = _resolve_T_split(spec, experiment)

    ctx = EvalContext(
        model=model,
        loader=loader,
        device=device,
        batch_transform=experiment.data.batch_transform,
        csv_path=csv_path,
        T_split=T_split,
        num_samples=int(spec.num_samples),
    )

    results: dict[str, Any] = {}
    for name in spec.metrics:
        if name not in METRIC_REGISTRY:
            raise KeyError(
                f"Unknown metric {name!r}. Registered metrics: "
                f"{sorted(METRIC_REGISTRY)}"
            )
        log.info("Computing metric %s", name)
        results.update(METRIC_REGISTRY[name](ctx))

    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, spec.output_filename)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log.info("Wrote %s", out_path)
    return results
