# filepath: evaluate_models.py
import os
import time
import csv
import json
import math
import argparse
import random
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from dssd.dssd import DSSD_base
from dssd.config import DSSDConfig, load_config_from_files, apply_dot_overrides

from dssd.data.synthetic import SyntheticDataset
from dssd.eval_utils import (
    visualize_results,
)


def set_eval_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_split(split: int, seq_len: int) -> int:
    if split < 1 or split >= seq_len:
        raise ValueError(
            f"Invalid split={split}. Must satisfy 1 <= split < seq_len ({seq_len})."
        )
    return split


def prepare_forecast_batch(batch: dict, split: int, device: torch.device):
    observed = batch["observed_data"].to(device)  # (B,D,T)
    timepoints = batch["timepoints"].to(device)  # (B,T)

    if "observed_mask" in batch:
        mask = batch["observed_mask"].to(device)
    elif "mask" in batch:
        mask = batch["mask"].to(device)
    else:
        mask = torch.ones_like(observed, device=device)

    x_hist = observed[..., :split].contiguous()
    x_mask = mask[..., :split].contiguous()
    past_time = timepoints[:, :split].contiguous()
    future_time = timepoints[:, split:].contiguous()
    y_future = observed[..., split:].contiguous()

    return {
        "x_hist": x_hist,
        "x_mask": x_mask,
        "past_time": past_time,
        "future_time": future_time,
        "y_future": y_future,
    }


def mae_metrics(pred_mean: torch.Tensor, y_future: torch.Tensor):
    # pred_mean, y_future: (B,D,L2)
    abs_err = (pred_mean - y_future).abs()  # (B,D,L2)
    mae_global = abs_err.mean()  # scalar
    mae_per_t = abs_err.mean(dim=(0, 1))  # (L2,)
    return mae_global, mae_per_t


def crps_sum_metrics(pred_samples: torch.Tensor, y_future: torch.Tensor):
    """
    CRPS-sum as in CSDI: CRPS for the distribution of the sum of all features.
    pred_samples: (B,S,D,L2)
    y_future:     (B,D,L2)
    Returns:
        crps_sum_global: scalar
        crps_sum_per_t: (L2,)
    """
    B, S, D, L2 = pred_samples.shape
    quantile_levels = torch.arange(0.05, 1.0, 0.05, device=pred_samples.device)
    n_quantiles = quantile_levels.shape[0]

    # Sum over features for each sample
    pred_sum_samples = pred_samples.sum(dim=2)  # (B,S,L2)
    y_sum = y_future.sum(dim=1)  # (B,L2)

    # Compute empirical quantiles for each α
    quantiles = torch.quantile(
        pred_sum_samples, quantile_levels, dim=1, keepdim=False
    )  # (n_quantiles,B,L2)
    quantiles = quantiles.permute(1, 2, 0)  # (B,L2,n_quantiles)

    y = y_sum.unsqueeze(-1)  # (B,L2,1)
    indicator = (y < quantiles).float()  # (B,L2,n_quantiles)

    quantile_loss = (quantile_levels - indicator) * (
        y_sum.unsqueeze(-1) - quantiles
    )  # (B,L2,n_quantiles)
    crps = 2 * quantile_loss.mean(dim=-1)  # (B,L2)

    crps_sum_global = crps.mean()
    crps_sum_per_t = crps.mean(dim=0)  # (L2,)
    return crps_sum_global, crps_sum_per_t


def energy_score_metrics(pred_samples: torch.Tensor, y_future: torch.Tensor):
    """
    Multivariate Energy Score over D dimensions.
    pred_samples: (B,S,D,L2)
    y_future:     (B,D,L2)
    """
    B, S, D, L2 = pred_samples.shape
    y = y_future.unsqueeze(1)  # (B,1,D,L2)

    # ||X - y|| over D
    term1 = torch.norm(pred_samples - y, p=2, dim=2).mean(dim=1)  # (B,L2)

    if S > 1:
        x_i = pred_samples.unsqueeze(2)  # (B,S,1,D,L2)
        x_j = pred_samples.unsqueeze(1)  # (B,1,S,D,L2)
        pair = torch.norm(x_i - x_j, p=2, dim=3).mean(dim=(1, 2))  # (B,L2)
        term2 = 0.5 * pair
    else:
        term2 = torch.zeros_like(term1)

    es = term1 - term2  # (B,L2)
    es_global = es.mean()
    es_per_t = es.mean(dim=0)  # (L2,)
    return es_global, es_per_t


def init_accumulators(horizon: int):
    z = torch.zeros(horizon, dtype=torch.float64)
    return {
        "sum_mae": 0.0,
        "sum_crps": 0.0,
        "sum_es": 0.0,
        "sum_mae_t": z.clone(),
        "sum_crps_t": z.clone(),
        "sum_es_t": z.clone(),
        "n_series": 0,
        "n_batches": 0,
    }


def update_acc(acc, mae_g, crps_g, es_g, mae_t, crps_t, es_t, batch_size: int):
    acc["sum_mae"] += float(mae_g.item()) * batch_size
    acc["sum_crps"] += float(crps_g.item()) * batch_size
    acc["sum_es"] += float(es_g.item()) * batch_size

    acc["sum_mae_t"] += mae_t.detach().cpu().double() * batch_size
    acc["sum_crps_t"] += crps_t.detach().cpu().double() * batch_size
    acc["sum_es_t"] += es_t.detach().cpu().double() * batch_size

    acc["n_series"] += batch_size
    acc["n_batches"] += 1


def finalize_metrics(acc, meta: dict):
    n = max(acc["n_series"], 1)
    metrics = {
        "meta": meta,
        "mae_mean": acc["sum_mae"] / n,
        "sum_crps": acc["sum_crps"] / n,
        "energy_score_mean": acc["sum_es"] / n,
        "mae_per_t": (acc["sum_mae_t"] / n).tolist(),
        "sum_crps_per_t": (acc["sum_crps_t"] / n).tolist(),
        "energy_score_per_t": (acc["sum_es_t"] / n).tolist(),
        "n_series": acc["n_series"],
        "n_batches": acc["n_batches"],
    }
    return metrics


def save_outputs(out_dir: str, metrics: dict):
    os.makedirs(out_dir, exist_ok=True)

    metrics_json = os.path.join(out_dir, "metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Save] {metrics_json}")

    per_t_csv = os.path.join(out_dir, "metrics_per_t.csv")
    with open(per_t_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_future_idx", "mae", "crps", "energy_score"])
        for t, (m, c, e) in enumerate(
            zip(
                metrics["mae_per_t"],
                metrics["sum_crps_per_t"],
                metrics["energy_score_per_t"],
            )
        ):
            writer.writerow([t, m, c, e])
    print(f"[Save] {per_t_csv}")


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    state = None
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]  # DSSDTrainer payload format
            print("[Checkpoint] Detected trainer payload format (model_state).")
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]  # common alt format
            print("[Checkpoint] Detected model_state_dict format.")
        else:
            # Fallback: raw state_dict-like mapping
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state = ckpt
                print("[Checkpoint] Detected raw state_dict format.")
            else:
                keys_preview = list(ckpt.keys())[:20]
                raise RuntimeError(
                    "Unsupported checkpoint dict format. "
                    f"Expected one of ['model_state', 'model_state_dict'] or raw tensor mapping. "
                    f"Top-level keys (preview): {keys_preview}"
                )
    else:
        raise RuntimeError(
            f"Unsupported checkpoint object type: {type(ckpt)}. Expected dict."
        )

    model.load_state_dict(state, strict=True)
    print(f"[Checkpoint] Loaded: {ckpt_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate trained DSSD models (no training)."
    )
    parser.add_argument("--config", type=str, nargs="+", required=True)
    parser.add_argument("--set", type=str, nargs="+", default=None)
    parser.add_argument("--resume", type=str, required=True, help="Checkpoint path.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "iid",
            "lgssm",
            "nonlinear",
            "nongaussian",
            "student_t",
            "harmonic",
            "harmonic-noisy",
            "bimodal",
            "bimodal-block",
            "robot-basis-pursuit",
        ],
    )
    parser.add_argument("--work_dir", type=str, default="runs/eval")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--split", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--num_eval_sequences", type=int, default=1000)
    parser.add_argument("--dataset_seed", type=int, default=123)
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which pre-generated split to evaluate on.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--forecast_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_plot", action="store_true")
    args = parser.parse_args()

    set_eval_seed(args.seed)
    split = validate_split(args.split, args.seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    config = load_config_from_files(args.config, args.set)

    dataset = SyntheticDataset(
        mode=args.mode,
        split=args.dataset_split,
        N_per_split=1024,
        T=args.seq_len,
        D=config.data_dim,
        dataset_seed=args.dataset_seed,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        generator=g,
    )

    model = DSSD_base(config, device).to(device)
    load_checkpoint(model, args.resume, device)
    model.eval()

    horizon = args.seq_len - split
    acc = init_accumulators(horizon=horizon)

    with torch.no_grad():
        for batch in loader:
            fb = prepare_forecast_batch(batch, split=split, device=device)
            out = model.forecast(
                x_hist=fb["x_hist"],
                x_mask=fb["x_mask"],
                past_time=fb["past_time"],
                future_time=fb["future_time"],
                num_samples=args.forecast_samples,
            )
            pred_mean = out["pred_mean"]  # (B,D,L2)
            pred_samples = out["pred_samples"]  # (B,S,D,L2)
            y_future = fb["y_future"]  # (B,D,L2)

            mae_g, mae_t = mae_metrics(pred_mean, y_future)
            crps_g, crps_t = crps_sum_metrics(pred_samples, y_future)
            es_g, es_t = energy_score_metrics(pred_samples, y_future)

            update_acc(
                acc,
                mae_g=mae_g,
                crps_g=crps_g,
                es_g=es_g,
                mae_t=mae_t,
                crps_t=crps_t,
                es_t=es_t,
                batch_size=y_future.shape[0],
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag if args.tag else os.path.splitext(os.path.basename(args.resume))[0]
    tag_dir = os.path.join(args.work_dir, args.mode, tag)
    out_dir = os.path.join(tag_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # Create/update latest symlink for easy references
    latest_symlink = os.path.join(tag_dir, "latest")
    if os.path.lexists(latest_symlink):
        os.remove(latest_symlink)
    try:
        os.symlink(timestamp, latest_symlink, target_is_directory=True)
    except OSError as e:
        print(f"[Warning] Could not create 'latest' symlink: {e}")

    meta = {
        "mode": args.mode,
        "split": split,
        "seq_len": args.seq_len,
        "horizon": horizon,
        "forecast_samples": args.forecast_samples,
        "seed": args.seed,
        "checkpoint_path": args.resume,
        "config_paths": args.config,
        "device": str(device),
    }
    metrics = finalize_metrics(acc, meta=meta)
    save_outputs(out_dir, metrics)

    if args.save_plot:
        plot_path = os.path.join(out_dir, "eval_plot.png")
        visualize_results(model, loader, device, split, save_path=plot_path)
        print(f"[Save] {plot_path}")

    print(
        f"[Eval Done] MAE={metrics['mae_mean']:.6f} "
        f"CRPS={metrics['sum_crps']:.6f} "
        f"ES={metrics['energy_score_mean']:.6f}"
    )


if __name__ == "__main__":
    main()
