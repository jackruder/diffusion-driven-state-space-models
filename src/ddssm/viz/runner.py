"""Visualization runner: walks a VizSpec, saves PNGs.

Loads a checkpoint into the experiment's already-built model, picks
the right loader from the data module (``test`` by default), and
invokes each plot named in :class:`VizSpec.plots`. Each :class:`PlotSpec`
gives a registered name plus optional kwargs forwarded to the plot
function. Output files land under the Hydra run dir.

Train, evaluate, and visualize are independent stages -- nothing in
this file runs during training.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from .plots import PLOT_REGISTRY, PlotContext

log = logging.getLogger(__name__)


@dataclass
class PlotSpec:
    """One plot to produce: a registered name + per-plot kwargs.

    Attributes:
        name: Registry key (see ``PLOT_REGISTRY``).
        save_filename: Output PNG name (relative to the run dir). If
            empty, defaults to ``f"{name}.png"``.
        kwargs: Extra keyword arguments forwarded to the plot function.
    """

    name: str
    save_filename: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VizSpec:
    """What to plot, on which split, with what defaults.

    Attributes:
        plots: List of :class:`PlotSpec` to produce in order.
        split: DataModule loader to draw from (``"train"`` / ``"val"`` / ``"test"``).
        num_samples: Forecast sample count for sample-based plots.
        T_split: Forecast split index. ``None`` falls back to
            ``data.metadata.forecast_split``.
    """

    # Annotated as ``list`` (no inner type) so OmegaConf's strict
    # validator accepts the zen-built Builds_PlotSpec dataclass alongside
    # plain PlotSpec instances; both are accepted at runtime via ``isinstance``-loose iteration.
    plots: list = field(default_factory=list)
    split: str = "test"
    num_samples: int = 10
    T_split: int | None = None


def _select_loader(experiment, split: str):
    if split == "train":
        return experiment.data.train_loader()
    if split == "val":
        return experiment.data.val_loader()
    if split == "test":
        return experiment.data.test_loader()
    raise ValueError(f"Unknown viz split: {split!r}")


def _resolve_T_split(spec: VizSpec, experiment) -> int | None:
    if spec.T_split is not None:
        return int(spec.T_split)
    meta = getattr(experiment.data, "metadata", None)
    if meta is None:
        return None
    return getattr(meta, "forecast_split", None)


def _maybe_load_checkpoint(model: torch.nn.Module, ckpt_path: str | None, device: torch.device) -> None:
    if ckpt_path is None:
        log.warning("No checkpoint provided; visualizing randomly-initialised weights.")
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path!r}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state, strict=True)
    log.info("Loaded checkpoint from %s", ckpt_path)


def visualize(
    experiment,
    spec: VizSpec,
    *,
    device: torch.device,
    run_dir: str,
    checkpoint_path: str | None = None,
    csv_path: str | None = None,
) -> list[str]:
    """Run every plot named in ``spec`` and return the list of saved paths."""
    model = experiment.model.to(device)
    _maybe_load_checkpoint(model, checkpoint_path, device)
    model.eval()

    loader = _select_loader(experiment, spec.split)
    T_split = _resolve_T_split(spec, experiment)

    ctx = PlotContext(
        model=model,
        loader=loader,
        device=device,
        batch_transform=experiment.data.batch_transform,
        csv_path=csv_path,
        T_split=T_split,
        num_samples=int(spec.num_samples),
    )

    os.makedirs(run_dir, exist_ok=True)
    saved: list[str] = []
    for plot in spec.plots:
        if plot.name not in PLOT_REGISTRY:
            raise KeyError(
                f"Unknown plot {plot.name!r}. Registered: {sorted(PLOT_REGISTRY)}"
            )
        out_name = plot.save_filename or f"{plot.name}.png"
        out_path = os.path.join(run_dir, out_name)
        log.info("Plotting %s -> %s", plot.name, out_path)
        PLOT_REGISTRY[plot.name](ctx, out_path, **plot.kwargs)
        saved.append(out_path)
    return saved
