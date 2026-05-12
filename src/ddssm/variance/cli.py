"""Hydra entry point for the variance probe stage.

Three modes, selected by CLI flags:

* default (no flags): probe a single checkpoint. If no checkpoint
  exists at ``<run_dir>/checkpoints/ckpt_latest.pth``, train first.

* ``+checkpoint=path``: probe the supplied checkpoint instead of
  the run-dir default.

* ``+per_step=true``: probe *every* ``ckpt_step{N}.pth`` under
  ``<run_dir>/checkpoints/``, write per-step probe outputs into
  ``<run_dir>/probe/step_{N:06d}/``, then stitch one GIF per plot
  into ``<run_dir>/`` so the variance landscape's evolution across
  training is visible at a glance. If no step checkpoints exist
  yet, training runs first.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

from .._experiment_registry import register_experiments
from .plots import PROBE_PLOT_REGISTRY, ProbePlotContext

register_experiments()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLOT_NAMES = (
    "var_grad_vs_tau",
    "var_loss_vs_tau",
    "ratio_vs_tau",
    "summary_table",
)

_CKPT_STEP_RE = re.compile(r"ckpt_step(\d+)\.pth$")


def _find_step_checkpoints(checkpoints_dir: str) -> list[tuple[int, str]]:
    """Return [(step, abs_path)] for every ``ckpt_stepN.pth`` in order."""
    if not os.path.isdir(checkpoints_dir):
        return []
    found = []
    for name in os.listdir(checkpoints_dir):
        m = _CKPT_STEP_RE.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(checkpoints_dir, name)))
    found.sort()
    return found


def _annotate_frame(png_path: str, step: int) -> Image.Image:
    """Open a probe-plot PNG and overlay a ``step N`` badge in the top-right."""
    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except (OSError, IOError):
        font = ImageFont.load_default()
    text = f"step {step}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 8
    x = img.width - tw - pad - 4
    y = pad
    # Solid white box behind text so it reads over any plot background.
    draw.rectangle(
        (x - 6, y - 4, x + tw + 6, y + th + 4),
        fill=(255, 255, 255),
    )
    draw.text((x, y), text, fill=(20, 20, 20), font=font)
    return img


def _collect_pos_values(d, out: list[float]) -> None:
    """Recurse a nested dict, appending finite positive numeric leaves to ``out``."""
    if isinstance(d, dict):
        for v in d.values():
            _collect_pos_values(v, out)
        return
    if isinstance(d, (list, tuple)):
        for v in d:
            _collect_pos_values(v, out)
        return
    try:
        x = float(d)
    except (TypeError, ValueError):
        return
    if math.isfinite(x) and x > 0:
        out.append(x)


def _global_ylim(metric_dicts: list[dict], metric_key: str,
                 pad: float = 3.0, anchors: tuple[float, ...] = ()) -> tuple[float, float] | None:
    """Compute a stable [low, high] log-axis bound across many metric snapshots.

    ``pad`` multiplies above the max and divides below the min so the
    extremes don't sit flush against the axis. ``anchors`` are points
    that must lie inside the returned range (e.g. parity=1.0 for ratio
    plots).
    """
    values: list[float] = []
    for m in metric_dicts:
        _collect_pos_values(m.get(metric_key, {}), values)
    if not values:
        return None
    lo, hi = min(values), max(values)
    for a in anchors:
        lo = min(lo, a)
        hi = max(hi, a)
    return (lo / pad, hi * pad)


def _global_bounds(per_step_data: list[tuple[int, dict]]) -> dict[str, dict]:
    """Per-plot fixed bounds so GIF frames don't jitter."""
    metrics_list = [data.get("metrics", {}) for _, data in per_step_data]
    bounds: dict[str, dict] = {}
    yl = _global_ylim(metrics_list, "grad_var_per_tau")
    if yl is not None:
        bounds["var_grad_vs_tau"] = {"ylim": yl}
    yl = _global_ylim(metrics_list, "loss_var_per_tau")
    if yl is not None:
        bounds["var_loss_vs_tau"] = {"ylim": yl}
    yl = _global_ylim(metrics_list, "ratio_per_tau", anchors=(1.0,))
    if yl is not None:
        bounds["ratio_vs_tau"] = {"ylim": yl}
    return bounds


def _compile_gif(
    plot_name: str, step_frames: list[tuple[int, str]], out_path: str,
    *, frame_ms: int = 600,
) -> None:
    """Stitch per-step PNGs of one plot type into an animated GIF."""
    frames = []
    for step, png_path in step_frames:
        if not os.path.exists(png_path):
            continue
        frames.append(_annotate_frame(png_path, step))
    if not frames:
        log.warning("No frames for %s — skipping GIF", plot_name)
        return
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )
    log.info("Saved animation %s (%d frames)", out_path, len(frames))


def _probe_per_step(experiment, device: torch.device, run_dir: str) -> dict:
    """Probe every ckpt_step*.pth in ``<run_dir>/checkpoints/``."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    step_ckpts = _find_step_checkpoints(ckpt_dir)
    if not step_ckpts:
        log.info(
            "No step checkpoints in %s — training first to populate them.",
            ckpt_dir,
        )
        experiment.train(device=device, run_dir=run_dir)
        step_ckpts = _find_step_checkpoints(ckpt_dir)
    if not step_ckpts:
        raise RuntimeError(
            f"Per-step probe requested but no ckpt_step*.pth files exist "
            f"under {ckpt_dir} even after training — check "
            f"experiment.training.checkpoint_every."
        )

    log.info(
        "Per-step probe: %d checkpoints (steps %s)",
        len(step_ckpts), [s for s, _ in step_ckpts],
    )
    per_step_dir = os.path.join(run_dir, "probe")
    os.makedirs(per_step_dir, exist_ok=True)

    step_outputs: list[tuple[int, str]] = []
    for i, (step, ckpt_path) in enumerate(step_ckpts, start=1):
        sub_dir = os.path.join(per_step_dir, f"step_{step:06d}")
        os.makedirs(sub_dir, exist_ok=True)
        log.info("[%d/%d] step %d → %s", i, len(step_ckpts), step, sub_dir)
        experiment.variance_probe(
            device=device,
            run_dir=sub_dir,
            checkpoint_path=ckpt_path,
        )
        step_outputs.append((step, sub_dir))

    # Re-render per-step plots with globally-fixed axes so each GIF
    # frame uses identical scales (otherwise the animation jitters as
    # matplotlib auto-scales each frame independently).
    log.info("Re-rendering per-step plots with fixed axes")
    per_step_data: list[tuple[int, dict]] = []
    for step, sub_dir in step_outputs:
        summary_path = os.path.join(sub_dir, "variance_summary.json")
        try:
            with open(summary_path) as f:
                per_step_data.append((step, json.load(f)))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            log.warning("Could not load %s for axis bounds: %s", summary_path, exc)

    bounds = _global_bounds(per_step_data)
    for step, data in per_step_data:
        sub_dir = next(d for s, d in step_outputs if s == step)
        ctx = ProbePlotContext(
            rows=[],
            summary=data.get("summary", {}),
            metrics=data.get("metrics", {}),
        )
        for plot_name in ("var_grad_vs_tau", "var_loss_vs_tau", "ratio_vs_tau"):
            kwargs = bounds.get(plot_name, {})
            png_path = os.path.join(sub_dir, f"{plot_name}.png")
            PROBE_PLOT_REGISTRY[plot_name](ctx, png_path, **kwargs)

    log.info("Stitching GIFs across %d step(s)", len(step_outputs))
    for plot_name in _PLOT_NAMES:
        frames = [(step, os.path.join(sub_dir, f"{plot_name}.png"))
                  for step, sub_dir in step_outputs]
        gif_path = os.path.join(run_dir, f"{plot_name}.gif")
        _compile_gif(plot_name, frames, gif_path)

    return {
        "per_step_dir": per_step_dir,
        "steps": [s for s, _ in step_outputs],
        "gifs": {
            name: os.path.join(run_dir, f"{name}.gif") for name in _PLOT_NAMES
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = HydraConfig.get().runtime.output_dir
    orig_cwd = hydra.utils.get_original_cwd()

    os.makedirs(run_dir, exist_ok=True)
    experiment = instantiate(cfg.experiment)

    # Per-step sweep mode — auto-trains if no step checkpoints exist
    # yet, then probes every ckpt_step*.pth.
    if bool(cfg.get("per_step", False)):
        return _probe_per_step(experiment, device, run_dir)

    # Default checkpoint location is ``<run_dir>/checkpoints/ckpt_latest.pth``.
    # Override with ``+checkpoint=path/to/other.pth`` for a one-off file.
    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(run_dir, "checkpoints", "ckpt_latest.pth")
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(orig_cwd, checkpoint_path)

    if not os.path.exists(checkpoint_path):
        log.info(
            "Checkpoint not found at %s — running training stage first.",
            checkpoint_path,
        )
        experiment.train(device=device, run_dir=run_dir)
        checkpoint_path = os.path.join(run_dir, "checkpoints", "ckpt_latest.pth")
        log.info("Training finished; probing from %s", checkpoint_path)

    return experiment.variance_probe(
        device=device,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
