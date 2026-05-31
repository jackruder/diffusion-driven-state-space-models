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
    from ..checkpoint import prepare_model

    model = prepare_model(
        experiment, checkpoint_path=checkpoint_path, device=device,
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
    from ..loggers import resume_run_from_dir

    wandb_mod = resume_run_from_dir(
        run_dir, getattr(experiment, "wandb_config", None),
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
        PLOT_REGISTRY[plot.name](ctx, out_path, **plot.kwargs)
        saved.append(out_path)
        if wandb_mod is not None:
            try:
                wandb_mod.log({f"viz/{plot.name}": wandb_mod.Image(out_path)})
            except Exception as e:  # noqa: BLE001 — best-effort
                log.warning("wandb image log failed for %s: %s", plot.name, e)
    if wandb_mod is not None:
        try:
            wandb_mod.finish()
        except Exception:  # noqa: BLE001 — best-effort
            pass
    return saved
