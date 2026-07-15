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

import os
import re
import json
import math
import logging

from PIL import Image, ImageDraw, ImageFont
import hydra
import numpy as np
import torch
from hydra_zen import instantiate
from omegaconf import OmegaConf, DictConfig
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig

from ddssm.variance.plots import PROBE_PLOT_REGISTRY, ProbePlotContext
from ddssm.experiment.registry import register_experiments

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

_CKPT_STEP_RE = re.compile(r"ckpt(?:_stage_\d+)?_step(\d+)\.pth$")


def _find_step_checkpoints(checkpoints_dir: str) -> list[tuple[int, str]]:
    """Return [(step, abs_path)] for every step checkpoint in order.

    Matches both the single-stage ``ckpt_step{N}.pth`` layout and the
    multi-stage ``ckpt_stage_{i}_step{N}.pth`` layout that
    :class:`~ddssm.training.stages.StageOrchestrator` writes.
    """
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
    except OSError:
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


def _global_ylim(
    metric_dicts: list[dict],
    metric_key: str,
    pad: float = 3.0,
    anchors: tuple[float, ...] = (),
) -> tuple[float, float] | None:
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


def _render_ratio_trajectory(
    run_dir: str,
    per_step_data: list[tuple[int, dict]],
    *,
    kind: str = "grad",
    mode: str = "adaptive_is",
    noise_levels: list[float] | None = None,
) -> None:
    """One static plot, one curve per checkpoint.

    ESM/DSM ``kind``-variance ratio across the diffusion τ-axis, with
    ``mode`` fixed (so every curve uses the same k-sampling). Curves are
    coloured by training step on a warm sequential colormap — earlier
    checkpoints sit at the cool end of the palette, the final checkpoint
    at the hot end. A colorbar maps colour back to step number.

    ``noise_levels`` (optional): the schedule's ``σ̃_τ`` values indexed
    by ``k``, e.g. ``model.transition.sigma_tilde.tolist()``. When given
    the x axis becomes noise level on a log scale; otherwise it stays as
    the raw ``τ``-bin index ``k``.
    """
    step_curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    for step, data in per_step_data:
        kvals = (
            data.get("metrics", {}).get("ratio_per_tau", {}).get(kind, {}).get(mode, {})
        )
        if not kvals:
            continue
        items = sorted(kvals.items(), key=lambda kv: int(kv[0]))
        ks = np.array([int(k) for k, _ in items], dtype=int)
        ys = np.array([float(v) for _, v in items], dtype=float)
        if noise_levels is not None:
            try:
                xs = np.array([float(noise_levels[k]) for k in ks], dtype=float)
            except (IndexError, TypeError):
                log.warning(
                    "noise_levels missing entry for k in %s — falling back to k axis",
                    ks.tolist(),
                )
                xs = ks.astype(float)
        else:
            xs = ks.astype(float)
        step_curves.append((step, xs, ys))

    if not step_curves:
        log.warning(
            "No ratio_per_tau[%s][%s] data — skipping trajectory plot",
            kind,
            mode,
        )
        return

    use_noise = noise_levels is not None

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Truncate the YlOrRd colormap to the warm half so the early
    # (lightest) checkpoint isn't near-white on a white background.
    base = plt.colormaps["YlOrRd"]
    cmap = LinearSegmentedColormap.from_list(
        "ylorrd_warm",
        base(np.linspace(0.25, 1.0, 256)),
    )
    steps = [s for s, _, _ in step_curves]
    vmin = min(steps) if len(set(steps)) > 1 else min(steps) - 1
    vmax = max(steps)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for step, xs, ys in step_curves:
        ax.plot(
            xs,
            ys,
            color=cmap(norm(step)),
            linewidth=1.4,
            alpha=0.85,
        )

    ax.axhline(
        1.0,
        color="grey",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label="parity (ESM = DSM)",
    )
    if use_noise:
        ax.set_xlabel(r"noise level $\tilde{\sigma}_\tau$")
        ax.set_xscale("log")
    else:
        ax.set_xlabel(r"$\tau$-bin index $k$")
    ax.set_ylabel(f"ESM / DSM {kind} variance ratio")
    ax.set_yscale("log")
    axis_desc = "noise level" if use_noise else r"$\tau$"
    ax.set_title(
        f"ESM vs DSM {kind}-variance ratio per {axis_desc}, "
        f"across training steps\n(k-sampling mode: {mode})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("training step")
    # Tick at each available step so a reader can map colour → step.
    cbar.set_ticks(steps)
    cbar.ax.tick_params(labelsize=8)

    fig.text(
        0.01,
        0.01,
        "Lines run from light (early training) to dark (late). "
        "< 1 → ESM has lower variance; > 1 → DSM has lower variance.",
        fontsize=7,
        style="italic",
        color="dimgrey",
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))

    axis_slug = "noise" if use_noise else "tau"
    out_path = os.path.join(
        run_dir, f"ratio_trajectory_{kind}_{mode}_{axis_slug}.png"
    )
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    log.info("Saved trajectory plot %s (%d curves)", out_path, len(step_curves))


def _compile_gif(
    plot_name: str,
    step_frames: list[tuple[int, str]],
    out_path: str,
    *,
    frame_ms: int = 600,
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


def _worker_probe_step(cfg_yaml: str, step: int, ckpt_path: str, sub_dir: str) -> tuple[int, str]:
    """Worker entry point (top-level so ``spawn`` can pickle it).

    Runs in a fresh process: re-registers experiments, rebuilds the
    ``Experiment`` from the parent's serialized config, and probes one
    checkpoint into ``sub_dir``. Each worker owns its own CUDA context,
    so multiple workers can share the single GPU with only launch-latency
    contention — which is exactly what dominates this probe's wall time.
    """
    import os as _os
    import logging as _logging
    import torch as _torch
    from omegaconf import OmegaConf as _OmegaConf
    from hydra_zen import instantiate as _instantiate

    from ddssm.experiment.registry import register_experiments as _reg

    _reg()
    cfg = _OmegaConf.create(cfg_yaml)
    experiment = _instantiate(cfg.experiment)
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    _os.makedirs(sub_dir, exist_ok=True)
    _logging.basicConfig(level=_logging.INFO, force=True)
    experiment.variance_probe(
        device=device, run_dir=sub_dir, checkpoint_path=ckpt_path,
    )
    return step, sub_dir


def _probe_per_step(
    experiment, device: torch.device, run_dir: str,
    *, cfg_yaml: str | None = None, workers: int = 1,
) -> dict:
    """Probe every ckpt_step*.pth in ``<run_dir>/checkpoints/``.

    When ``workers > 1`` and ``cfg_yaml`` is provided, checkpoints are
    probed concurrently in ``spawn``-context subprocesses — each rebuilds
    the experiment from the serialized parent config. Workers share the
    GPU (small model → GPU is launch-latency bound, so ~2-3× speedup is
    typical up to ~4 workers before contention eats the win).
    """
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
        "Per-step probe: %d checkpoints (steps %s), workers=%d",
        len(step_ckpts), [s for s, _ in step_ckpts], workers,
    )
    per_step_dir = os.path.join(run_dir, "probe")
    os.makedirs(per_step_dir, exist_ok=True)

    step_outputs: list[tuple[int, str]] = []
    if workers > 1 and cfg_yaml is not None:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as ex:
            fut_to_step = {}
            for step, ckpt_path in step_ckpts:
                sub_dir = os.path.join(per_step_dir, f"step_{step:06d}")
                fut = ex.submit(
                    _worker_probe_step, cfg_yaml, step, ckpt_path, sub_dir,
                )
                fut_to_step[fut] = step
            done_count = 0
            for fut in as_completed(fut_to_step):
                step_done, sub_dir_done = fut.result()
                done_count += 1
                log.info(
                    "[%d/%d] worker done: step %d",
                    done_count, len(step_ckpts), step_done,
                )
                step_outputs.append((step_done, sub_dir_done))
        step_outputs.sort()
    else:
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
        frames = [
            (step, os.path.join(sub_dir, f"{plot_name}.png"))
            for step, sub_dir in step_outputs
        ]
        gif_path = os.path.join(run_dir, f"{plot_name}.gif")
        _compile_gif(plot_name, frames, gif_path)

    # Persist the aggregated trajectory data so the plots can be
    # re-rendered / re-styled without rerunning the probe. One entry per
    # step; ``metrics`` mirrors each per-step ``variance_summary.json``.
    trajectory_path = os.path.join(run_dir, "trajectory_data.json")
    with open(trajectory_path, "w") as f:
        json.dump(
            [
                {"step": step, **data}
                for step, data in per_step_data
            ],
            f, indent=2, default=float,
        )
    log.info(
        "Saved trajectory data → %s (%d steps)",
        trajectory_path, len(per_step_data),
    )

    # Grab the schedule's noise levels σ̃_τ from the parent experiment's
    # transition so the trajectory plot can put actual noise scale on the
    # x-axis (rather than raw τ-bin index k). Buffers are populated at
    # transition construction time from the VP schedule params — no
    # checkpoint load needed. Save to disk so post-hoc re-renders can
    # reuse it without rebuilding the experiment.
    noise_levels: list[float] | None = None
    try:
        sigma_tilde = experiment.model.module.transition.sigma_tilde
        noise_levels = [float(x) for x in sigma_tilde.detach().cpu().tolist()]
        with open(os.path.join(run_dir, "noise_levels.json"), "w") as f:
            json.dump(noise_levels, f, indent=2)
    except AttributeError:
        log.warning(
            "Transition has no sigma_tilde buffer — trajectory plot will "
            "use raw τ-bin index k as x-axis."
        )

    # Static trajectory plot — all checkpoints overlaid, coloured by step.
    # Only adaptive_is is rendered because that's what these probes actually
    # sample (uniform/lsgm_is cells are not in the ProbeSpec, so their
    # ratio_per_tau slots are empty in the summary JSONs).
    log.info("Rendering ratio trajectory plot")
    _render_ratio_trajectory(
        run_dir, per_step_data, kind="grad", mode="adaptive_is",
        noise_levels=noise_levels,
    )

    return {
        "per_step_dir": per_step_dir,
        "steps": [s for s, _ in step_outputs],
        "gifs": {name: os.path.join(run_dir, f"{name}.gif") for name in _PLOT_NAMES},
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
    experiment.model_config_yaml = OmegaConf.to_yaml(
        cfg.experiment.model,
        resolve=True,
    )

    # Per-step sweep mode — auto-trains if no step checkpoints exist
    # yet, then probes every ckpt_step*.pth. When ``+parallel_workers=N``
    # is set (N>1), checkpoints are probed concurrently in spawn workers,
    # each rebuilding the experiment from the serialised parent config.
    if bool(cfg.get("per_step", False)):
        workers = int(cfg.get("parallel_workers", 1))
        cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True) if workers > 1 else None
        return _probe_per_step(
            experiment, device, run_dir,
            cfg_yaml=cfg_yaml, workers=workers,
        )

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
