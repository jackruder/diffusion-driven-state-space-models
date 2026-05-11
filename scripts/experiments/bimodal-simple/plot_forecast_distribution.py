"""Plot forecast sample distributions for a trained bimodal DDSSM checkpoint.

Usage::

    python scripts/experiments/bimodal-simple/plot_forecast_distribution.py \\
        --config configs/base.yaml \\
        --ckpt checkpoints/bimodal_run/ckpt_latest.pth \\
        --out_path plots/forecast_dist.png \\
        [--sample_indices 0 1 2]

Loads a checkpoint, draws forecast sample paths, and overlays them against the
analytic bimodal density for visual inspection.
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from ddssm.ddssm import DDSSM_base
from ddssm.data.synthetic import SyntheticDataset
from ddssm.eval_utils import load_config_from_files


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--set", nargs="+", default=None)
    p.add_argument("--resume", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--seq_len", type=int, default=48)
    p.add_argument("--split", type=int, default=32)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--forecast_samples", type=int, default=512)
    p.add_argument("--series_idx", type=int, default=0)
    p.add_argument("--dim_idx", type=int, default=0)
    p.add_argument("--t_future_idx", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/plots/forecast_dist.png")

    p.add_argument("--dataset_seed", type=int, default=123)
    p.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_files(args.config, args.set)

    ds = SyntheticDataset(
        mode=args.mode,
        split=args.dataset_split,
        N_per_split=args.num_samples,
        T=args.seq_len,
        D=config.data_dim,
        dataset_seed=args.dataset_seed,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = DDSSM_base(config, device).to(device)
    load_checkpoint(model, args.resume, device)
    model.eval()

    batch = next(iter(dl))
    observed = batch["observed_data"].to(device)  # (B,D,T)
    timepoints = batch["timepoints"].to(device)  # (B,T)
    mask = batch.get("observation_mask", torch.ones_like(observed)).to(device)

    x_hist = observed[..., : args.split]
    x_mask = mask[..., : args.split]
    past_time = timepoints[:, : args.split]
    future_time = timepoints[:, args.split :]
    y_future = observed[..., args.split :]  # (B,D,L2)

    with torch.no_grad():
        out = model.forecast(
            x_hist=x_hist,
            x_mask=x_mask,
            past_time=past_time,
            future_time=future_time,
            num_samples=args.forecast_samples,
        )

    pred_samples = out["pred_samples"]  # (B,S,D,L2)
    pred_mean = out["pred_mean"]  # (B,D,L2)

    b = args.series_idx
    d = args.dim_idx
    t = args.t_future_idx

    vals = pred_samples[b, :, d, t].detach().cpu().numpy()
    y = float(y_future[b, d, t].detach().cpu().item())
    mu = float(pred_mean[b, d, t].detach().cpu().item())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=50, density=True, alpha=0.6, label="forecast samples")
    plt.axvline(y, color="red", linestyle="--", linewidth=2, label=f"truth={y:.3f}")
    plt.axvline(mu, color="black", linestyle="-", linewidth=2, label=f"mean={mu:.3f}")
    plt.title(f"{args.mode} | series={b} dim={d} t+{t}")
    plt.xlabel("value")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
