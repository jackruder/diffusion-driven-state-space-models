import os
import csv
from datetime import datetime
from types import SimpleNamespace
import argparse

import torch
import torch._dynamo

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ddssm.ddssm import DDSSM_base
from ddssm.train import DDSSMTrainer
from ddssm.config import (
    DDSSMConfig,
    deep_merge,
    apply_dot_overrides,
    load_config_from_files,
)
from ddssm.data.synthetic import SyntheticDataset
from ddssm.eval_utils import (
    visualize_results,
)


def plot_metrics(csv_path, save_path, keys=None):
    """Plots specific metrics from CSV log.

    Args:
        csv_path: Path to the CSV log file.
        save_path: Path to save the plot image.
        keys: List of column names to plot (e.g., ['loss/total', 'optim/lambda']).
              Defaults to ['loss/total'].
    """
    if keys is None:
        keys = ["loss/total"]

    if not os.path.exists(csv_path):
        print(f"[Warning] No CSV log found at {csv_path}, skipping metric plot.")
        return

    data = {k: [] for k in keys}
    steps = []

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            # Validate keys exist in CSV header
            if reader.fieldnames:
                valid_keys = [k for k in keys if k in reader.fieldnames]
                if len(valid_keys) != len(keys):
                    print(f"[Warning] Some keys not found in CSV. Found: {valid_keys}")
                keys = valid_keys

            for row in reader:
                if "step" not in row:
                    continue
                try:
                    s = int(row["step"])
                    current_values = {}
                    for k in keys:
                        current_values[k] = float(row[k])

                    # Only append if all keys were parsed successfully for this row
                    steps.append(s)
                    for k in keys:
                        data[k].append(current_values[k])
                except ValueError:
                    continue
    except Exception as e:
        print(f"[Error] Failed to read metrics: {e}")
        return

    if not steps:
        print("[Warning] No valid data found in CSV log.")
        return

    plt.figure(figsize=(10, 6))
    for k in keys:
        plt.plot(steps, data[k], label=k)

    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.title(f"Training Metrics: {', '.join(keys)}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved metric plot to {save_path}")


def run_verification(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Verification: {args.mode} on {device} ---")
    print(f"--- Training mode: {args.training_mode} ---")

    # Auto-versioning with timestamps FIRST
    mode_dir = os.path.join(args.work_dir, args.mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(mode_dir, timestamp)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)

    # Create a symlink for referencing in the org file
    latest_symlink = os.path.join(mode_dir, "latest")
    if os.path.lexists(latest_symlink):
        os.remove(latest_symlink)
    try:
        os.symlink(timestamp, latest_symlink, target_is_directory=True)
    except OSError as e:
        print(f"[Warning] Could not create 'latest' symlink: {e}")

    # Load Config SECOND
    if args.config:
        config_paths = args.config if isinstance(args.config, list) else [args.config]
        config = load_config_from_files(config_paths, args.set)
        config.checkpoint_dir = ckpt_dir
    else:
        print("Error: --config is required to specify the model configuration.")
        return

    # Apply overrides to default configs if no YAML was provided
    if not args.config and args.set:
        config_dict = (
            config.model_dump() if hasattr(config, "model_dump") else config.dict()
        )
        config_dict = apply_dot_overrides(config_dict, args.set)
        config = DDSSMConfig.model_validate(config_dict)

    # Instantiate the dataset THIRD, dynamically using config.data_dim
    dataset = SyntheticDataset(
        mode=args.mode,
        split="train",
        N_per_split=1024,
        T=args.seq_len,
        D=config.data_dim,
        dataset_seed=args.dataset_seed,
    )
    loader = DataLoader(dataset, batch_size=config.hyperparams.batch_size, shuffle=True)

    model = DDSSM_base(config, device)

    csv_log_path = os.path.join(run_dir, "metrics.csv")

    trainer = DDSSMTrainer(
        model,
        device=device,
        tensorboard_dir=os.path.join(run_dir, "logs"),
        csv_log_path=csv_log_path,
        quiet=args.quiet,
    )

    config_snapshot_path = os.path.join(run_dir, "config.yaml")
    print(f"[Config] Saving effective configuration to {config_snapshot_path}")
    # We use the trainer's helper or dump via Pydantic directly
    if hasattr(config, "model_dump"):
        with open(config_snapshot_path, "w") as f:
            yaml.safe_dump(config.model_dump(), f)
    elif hasattr(config, "dict"):
        with open(config_snapshot_path, "w") as f:
            yaml.safe_dump(config.dict(), f)

    # Optional resume only for the final joint stage; staged runs here start fresh
    resume_path = args.resume
    if resume_path is None:
        potential_latest = os.path.join(
            ckpt_dir, f"ckpt_{args.training_mode}_latest.pth"
        )
        if os.path.exists(potential_latest):
            print(
                f"[Info] Found existing checkpoint, resuming from: {potential_latest}"
            )
            resume_path = potential_latest

    def _viz(stage_key: str):
        os.makedirs(run_dir, exist_ok=True)
        img_path = os.path.join(run_dir, f"verify_{args.mode}_{stage_key}.png")
        print(f"[Stage {stage_key}] Generating visualization -> {img_path}")
        visualize_results(trainer.model, loader, device, args.split, save_path=img_path)

    # ==================== Training Mode Dispatch ====================

    if args.training_mode == "joint":
        # ---------------- Joint training only ----------------
        print("[Joint] Full joint training from scratch")
        trainer._set_trainable(
            SimpleNamespace(encoder=True, decoder=True, z_init=True, transition=True)
        )

        try:
            trainer.fit(
                train_loader=loader,
                total_steps=args.steps,
                validate_every=0,
                log_every=10,
                checkpoint_every=50,
                compute_recon=True,
                compute_trans=True,
                resume_from=resume_path,
            )
        except KeyboardInterrupt:
            print("\n[Info] Training interrupted by user.")

        _viz("joint")

    elif args.training_mode == "recon_only":
        # ---------------- Reconstruction only ----------------
        print("[Recon Only] Training encoder/decoder only")
        trainer._set_trainable(
            SimpleNamespace(encoder=True, decoder=True, z_init=True, transition=False)
        )

        try:
            trainer.fit(
                train_loader=loader,
                total_steps=args.steps,
                validate_every=0,
                log_every=10,
                checkpoint_every=50,
                compute_recon=True,
                compute_trans=False,
                resume_from=resume_path,
            )
        except KeyboardInterrupt:
            print("\n[Info] Training interrupted by user.")

        _viz("recon_only")

    elif args.training_mode == "trans_only":
        # ---------------- Transition only ----------------
        print("[Trans Only] Training transition model only")
        trainer._set_trainable(
            SimpleNamespace(encoder=False, decoder=False, z_init=False, transition=True)
        )

        try:
            trainer.fit(
                train_loader=loader,
                total_steps=args.steps,
                validate_every=0,
                log_every=10,
                compute_recon=False,
                compute_trans=True,
                checkpoint_every=50,
                resume_from=resume_path,
            )
        except KeyboardInterrupt:
            print("\n[Info] Training interrupted by user.")

        _viz("trans_only")

    print(f"--- Finished {args.mode} ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DDSSM verification on synthetic data."
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",  # Allow multiple config files!
        default=None,
        help="Path(s) to YAML config files. Usage: --config base.yaml override.yaml",
    )

    parser.add_argument(
        "--set",
        type=str,
        nargs="+",
        default=None,
        help="Override config values using dot notation (e.g., --set hyperparams.batch_size=32 encoder.hidden_dim=64)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="iid",
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
        help="Synthetic data mode.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="diffusion",
        choices=["gaussian", "diffusion"],
        help="Type of transition model to use (gaussian vs diffusion config).",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="runs/verify",
        help="Base directory for logs and checkpoints.",
    )
    parser.add_argument("--steps", type=int, default=500, help="Total training steps.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of synthetic sequences to generate.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to specific checkpoint to resume from. If not set, checks for 'ckpt_latest.pth'.",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="joint",
        choices=["joint", "joint", "recon_only", "trans_only"],
        help="Training strategy: three_stage (recon -> trans -> joint), joint (joint only), recon_only, or trans_only.",
    )

    parser.add_argument(  # if passed, sets args.quiet to True supressing console loggingg
        "--quiet",
        action="store_true",
        help="If set, suppress console logging.",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=16,
        help="Timestep of context split.",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=32,
        help="Total sequence length.",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=123,
        help="Random seed for synthetic dataset generation.",
    )
    args = parser.parse_args()
    run_verification(args)
