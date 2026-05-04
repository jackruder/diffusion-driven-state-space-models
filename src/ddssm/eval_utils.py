"""Evaluation helpers and visualisation utilities for DDSSM models."""

import math

import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
import ast

from .config import (
    DDSSMConfig,
    deep_merge,
    apply_dot_overrides,
    load_config_from_files,
)


def visualize_results(
    model,
    loader,
    device,
    T_split,
    save_path="verification_plot.png",
    sample_indices=None,
    font_size: int = 18,
    tick_font_size: int = 12,
    show_title: bool = False,
    time_start_at_zero: bool = False,
):
    """Plot reconstruction and forecast for a batch of samples.

    For 1-D data, draws observed vs reconstructed time series and overlays
    forecast sample paths and the forecast mean beyond ``T_split``.  For 2-D
    spatial data, plots X-vs-Y trajectories with sample-path forecasts.

    Args:
        model: Trained ``DDSSM_base`` model (put in eval mode internally).
        loader: ``DataLoader`` whose first batch (or ``sample_indices``) is used.
        device: Device for inference tensors.
        T_split: Index separating the context window from the forecast horizon.
        save_path: File path for the saved figure (PNG).
        sample_indices: Optional list of dataset indices to plot; if ``None``,
            the first batch from ``loader`` is used (up to 8 samples).
        font_size: Base font size for labels and legends.
        tick_font_size: Font size for axis tick labels.
        show_title: Whether to add a per-subplot title with the sample index.
        time_start_at_zero: If ``True``, the time axis starts at 0; otherwise at 1.
    """
    model.eval()

    # Grab specific samples if requested, otherwise take the first batch
    if sample_indices is not None:
        from torch.utils.data.dataloader import default_collate

        items = [loader.dataset[i] for i in sample_indices]
        batch = default_collate(items)
        B_plot = len(sample_indices)
    else:
        batch = next(iter(loader))
        B_plot = min(8, batch["observed_data"].shape[0])

    observed = batch["observed_data"].to(device)  # (B, D, T)
    timepoints = batch["timepoints"].to(device)  # (B, T)
    B_plot = min(B_plot, observed.shape[0])

    # Robust mask handling
    if "observed_mask" in batch:
        mask = batch["observed_mask"].to(device)
    elif "mask" in batch:
        mask = batch["mask"].to(device)
    else:
        # Default to all ones if no mask provided
        mask = torch.ones_like(observed, device=device)

    # 1. Reconstruction
    with torch.no_grad():
        # Get reconstruction (posterior samples)
        _loss, _rate, _distortion, _metrics, stats = model(
            observed, mask, timepoints, train=False
        )
        zs = stats["zs"]  # (B, S, d, T)

        # Use a single sample path, never average samples
        z_sample = zs[:, 0, :, :]  # (B, d, T)

        from ddssm.net_utils import time_embedding

        time_embed = time_embedding(timepoints, model.emb_time_dim, device=device)

        # Decode step-by-step
        recons = []
        for t in range(observed.shape[-1]):
            t_idx = torch.full((observed.shape[0],), t, device=device, dtype=torch.long)
            z_hist = z_sample[..., : t + 1]
            if z_hist.shape[-1] > model.j:
                z_hist = z_hist[..., -model.j :]

            mu_x, _ = model.decoder(z_hist, time_embed, t_idx)
            recons.append(mu_x)

        recons = torch.stack(recons, dim=-1)  # (B, D, T)

    #  Forecast (Generative)
    # Split into past (context) and future (pred)
    x_hist = observed[..., :T_split].contiguous()
    x_mask = mask[..., :T_split].contiguous()
    t_past = timepoints[:, :T_split].contiguous()
    t_fut = timepoints[:, T_split:].contiguous()

    with torch.no_grad():
        forecast_out = model.forecast(
            x_hist=x_hist,
            x_mask=x_mask,
            past_time=t_past,
            future_time=t_fut,
            num_samples=10,
        )
    pred_mean = forecast_out["pred_mean"]  # (B, D, L2)
    pred_samples = forecast_out["pred_samples"]  # (B, S, D, L2)

    # Plotting
    # Tweak figure size based on 1D vs 2D
    D_data = observed.shape[1]
    is_2d = D_data >= 2

    # Configure 2-column grid
    n_cols = 1
    n_rows = B_plot

    plt.rcParams.update({"font.size": font_size, "axes.titlesize": font_size})

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 6 * n_rows if is_2d else 4 * n_rows),
        sharex=False,
        squeeze=False,
    )
    # Flatten grid to 1D array for easy iteration
    axes = axes.flatten()

    observed = observed.cpu().numpy()
    recons = recons.cpu().numpy()
    pred_mean = pred_mean.cpu().numpy()
    pred_samples = pred_samples.cpu().numpy()

    for i in range(B_plot):
        ax = axes[i]

        # Use the actual index numbers for the subplot titles if provided
        plot_title = f"Sample {sample_indices[i] if sample_indices else i}"

        if not is_2d:
            T_total = observed.shape[-1]
            x_obs = (
                np.arange(T_total) if time_start_at_zero else np.arange(1, T_total + 1)
            )

            ax.plot(x_obs, observed[i, 0, :], "k-", label="Observed Data", alpha=0.6)
            ax.plot(x_obs, recons[i, 0, :], "b--", label="Reconstruction", alpha=0.8)

            # Forecast starts from the last observed point (T_split - 1)
            last_obs_idx = T_split - 1
            x_last = x_obs[last_obs_idx]
            x_fut = np.concatenate([
                [x_last],
                x_obs[T_split:],
            ])  # same axis, guaranteed alignment

            # Prepend last observed value so forecast lines connect visually
            last_val = observed[i, 0, last_obs_idx]

            for s in range(pred_samples.shape[1]):
                y_s = np.concatenate([[last_val], pred_samples[i, s, 0, :]])
                ax.plot(x_fut, y_s, color="red", alpha=0.15, linewidth=1)

            y_mean = np.concatenate([[last_val], pred_mean[i, 0, :]])
            ax.plot(x_fut, y_mean, "r-", label="Forecast Mean", linewidth=2)

            ax.axvline(x=x_last, color="gray", linestyle=":", label="Context Split")
            ax.set_ylabel("Value", fontsize=font_size + 3)
            ax.tick_params(axis="both", which="major", labelsize=tick_font_size + 4)

            if i == 0:
                ax.legend(fontsize=font_size, loc="upper left")
            if show_title:
                ax.set_title(plot_title, fontsize=font_size)

        else:
            # --- 2D SPATIAL PLOT ---
            # Draw the central obstacle box for context [-0.6, 0.6]
            import matplotlib.patches as patches

            rect = patches.Rectangle(
                (-0.6, -0.6),
                1.2,
                1.2,
                linewidth=1,
                edgecolor="black",
                facecolor="gray",
                alpha=0.3,
                label="Obstacle",
            )
            ax.add_patch(rect)

            # Ground truth and recon (X vs Y)
            ax.plot(
                observed[i, 0, :],
                observed[i, 1, :],
                "k-",
                label="Ground Truth",
                alpha=0.6,
                marker=".",
                markersize=3,
            )
            ax.plot(
                recons[i, 0, :],
                recons[i, 1, :],
                "b--",
                label="Reconstruction",
                alpha=0.7,
            )

            # Context Split Marker
            ax.plot(
                observed[i, 0, T_split - 1],
                observed[i, 1, T_split - 1],
                "go",
                label="Context End",
                markersize=8,
            )

            # individual forecast trajectories in light red
            for s in range(pred_samples.shape[1]):
                xs = np.concatenate([
                    [observed[i, 0, T_split - 1]],
                    pred_samples[i, s, 0, :],
                ])
                ys = np.concatenate([
                    [observed[i, 1, T_split - 1]],
                    pred_samples[i, s, 1, :],
                ])
                ax.plot(xs, ys, color="red", alpha=0.15, linewidth=1)

            # mean Forecast in solid red
            xs_mean = np.concatenate([
                [observed[i, 0, T_split - 1]],
                pred_mean[i, 0, :],
            ])
            ys_mean = np.concatenate([
                [observed[i, 1, T_split - 1]],
                pred_mean[i, 1, :],
            ])
            ax.plot(xs_mean, ys_mean, "r-", label="Forecast Mean", linewidth=2)

            ax.set_aspect("equal", "box")
            ax.set_xlim([-2.2, 2.5])
            ax.set_ylim([-2.0, 2.0])

            if i == 0:
                # Remove duplicate rect labels
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc="upper left")
            ax.set_title(f"{plot_title} (2D Spatial Path)")

    # Hide unused subplots if B_plot is less than grid size
    for k in range(B_plot, len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.rcParams.update(plt.rcParamsDefault)
