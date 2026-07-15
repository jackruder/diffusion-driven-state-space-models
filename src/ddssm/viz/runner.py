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

import os
from typing import Any
import logging
from dataclasses import field, dataclass

import torch

from ddssm.viz.plots import PLOT_REGISTRY, PlotContext
from ddssm.adapters.base import MetricNotSupported

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
        T_split: Forecast split index. ``None`` falls back to the data
            module default via ``data.metadata.forecast_split_or``.
    """

    # Annotated as ``list`` (no inner type) so OmegaConf's strict
    # validator accepts the zen-built Builds_PlotSpec dataclass alongside
    # plain PlotSpec instances; both are accepted at runtime via ``isinstance``-loose iteration.
    plots: list = field(default_factory=list)
    split: str = "test"
    num_samples: int = 10
    T_split: int | None = None


def visualize(
    experiment,
    spec: VizSpec,
    *,
    device: torch.device,
    run_dir: str,
    checkpoint_path: str | None = None,
    csv_path: str | None = None,
) -> list[str]:
    """Load a checkpoint, run every plot in ``spec``, and save PNGs.

    Args:
        experiment: The built :class:`~ddssm.experiment.Experiment`.
        spec: Which plots to draw, the split to draw from, and defaults.
        device: Torch device for the model forward passes.
        run_dir: Output directory the PNGs are written under.
        checkpoint_path: Checkpoint to load into the model. ``None`` uses
            the experiment's default checkpoint resolution.
        csv_path: Optional ``metrics.csv`` path for CSV-driven plots.

    Returns:
        Absolute paths of the saved PNGs, in spec order.

    Raises:
        KeyError: If a requested plot name is not in ``PLOT_REGISTRY``.
    """
    from ddssm.training.checkpoint import prepare_model

    # ``prepare_model`` returns the ModelAdapter. Forecast-based plots call the
    # adapter surface (``.forecast``) directly; DDSSM-only plots reach the raw
    # module via ``ctx.require_module(...)``. Keep the adapter on the context so
    # both paths (and the gating in ``require_module``) resolve.
    model = prepare_model(
        experiment,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    loader = experiment.data.loader(spec.split)
    T_split = experiment.data.metadata.forecast_split_or(spec.T_split)

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

    # Cross-stage W&B reconnect: when the train run wrote a run-id under
    # ``run_dir/.wandb_run_id`` and the experiment carries an enabled
    # ``wandb_config``, push each generated PNG to the original W&B run
    # so plots and training scalars live together.
    from ddssm.training.loggers import resume_run_from_dir

    wandb_mod = resume_run_from_dir(
        run_dir,
        getattr(experiment, "wandb_config", None),
    )

    saved: list[str] = []
    for plot in spec.plots:
        if plot.name not in PLOT_REGISTRY:
            raise KeyError(
                f"Unknown plot {plot.name!r}. Registered: {sorted(PLOT_REGISTRY)}"
            )
        out_name = plot.save_filename or f"{plot.name}.png"
        out_path = os.path.join(run_dir, out_name)
        log.info("Plotting %s -> %s", plot.name, out_path)
        # Method-level gating (same shape as the eval runner): a plot the
        # current model family can't support raises ``MetricNotSupported`` at
        # the point of need (``ctx.require_module(...)`` inside DDSSM-only
        # plots). Skip it — don't record a path — and keep rendering the rest.
        try:
            PLOT_REGISTRY[plot.name](ctx, out_path, **plot.kwargs)
        except MetricNotSupported as exc:
            log.warning("Skipping plot %s: %s", plot.name, exc)
            continue
        saved.append(out_path)
        if wandb_mod is not None:
            try:
                wandb_mod.log({f"viz/{plot.name}": wandb_mod.Image(out_path)})
            except Exception as e:
                log.warning("wandb image log failed for %s: %s", plot.name, e)
    if wandb_mod is not None:
        try:
            wandb_mod.finish()
        except Exception:
            pass
    return saved
