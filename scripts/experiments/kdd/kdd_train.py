"""Train a DDSSM model on the KDD Cup 2018 air-quality dataset.

Usage::

    python scripts/experiments/kdd/kdd_train.py \\
        --config configs/base.yaml configs/kdd.yaml \\
        [--override hyperparams.batch_size=32 transition.type=diffusion]

The script loads data from the KDD TSF files, initialises a ``DDSSMTrainer``
from the merged YAML configs, and runs multi-stage training via
``StageOrchestrator``.  Checkpoints and logs are written under the run directory.
"""

import os
from pathlib import Path
import argparse
from datetime import datetime

import yaml
import torch
from ddssm.ddssm import DDSSM_base
import torch._dynamo
import torch._inductor.config as inductor_config

from ddssm.train import DDSSMTrainer
from ddssm.config import DDSSMConfig, apply_dot_overrides, load_config_from_files
from ddssm.data.dataload import parse_batch, build_loaders_for_expt


def _update_latest_symlink(work_dir: str, run_dir: str) -> None:
    work = Path(work_dir)
    latest = work / "latest"
    target = Path(run_dir).resolve()
    tmp = work / ".latest_tmp"

    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target, target_is_directory=True)
    tmp.replace(latest)


def main(args):
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 64
    torch._dynamo.config.recompile_limit = 64
    inductor_config.pad_outputs = False
    inductor_config.pad_dynamic_shapes = False
    inductor_config.force_shape_pad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        _dummy = torch.randn(8, 8, device=device) @ torch.randn(8, 8, device=device)
        del _dummy
        torch.cuda.synchronize()

    print(f"--- Running KDD Training on {device} ---")
    #  Load Preprocessed Data
    print(f"Loading data from {args.data_path}...")
    data_payload = torch.load(args.data_path, weights_only=False)
    series_list = data_payload["series_list"]
    covariates_list = data_payload["covariates_list"]
    static_covariates = data_payload["static_covariates"]

    # Setup Config
    config = load_config_from_files(args.config, args.set)
    config_dict = (
        config.model_dump() if hasattr(config, "model_dump") else config.dict()
    )
    config_dict = apply_dot_overrides(config_dict, args.set)

    # Inject dataset constants strictly
    config_dict["data_dim"] = len(series_list)
    config_dict["covariate_dim"] = covariates_list[0].shape[0]
    config_dict["num_classes_per_static"] = [
        int(static_covariates[:, 0].max() + 1),
        int(static_covariates[:, 1].max() + 1),
        int(static_covariates[:, 2].max() + 1),
    ]

    config = DDSSMConfig.model_validate(config_dict)

    L1, L2 = args.split, args.seq_len - args.split

    train_loader, _, _, (means, stds) = build_loaders_for_expt(
        series_list=series_list,
        L1=L1,
        L2=L2,
        val_windows=24,  # February (28 days - 5 days + 1)
        test_windows=27,  # March (31 days - 5 days + 1)
        eval_step_size=24,  # Daily evaluation
        batch_size=config.hyperparams.batch_size,
        num_train_batches_per_epoch=None,  # use all remaining data for training
        covariates_list=covariates_list,
        static_covariates=static_covariates,
        backend="torch",
        normalize=True,
        device=device,
    )
    # 3. Setup Trainer
    run_dir = os.path.join(args.work_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    _update_latest_symlink(args.work_dir, run_dir)
    config.checkpoint_dir = os.path.join(run_dir, "checkpoints")

    model = DDSSM_base(config, device)
    trainer = DDSSMTrainer(
        model,
        device=device,
        tensorboard_dir=os.path.join(run_dir, "logs"),
        csv_log_path=os.path.join(run_dir, "metrics.csv"),
        quiet=args.quiet,
    )

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(
            config.model_dump() if hasattr(config, "model_dump") else config.dict(), f
        )

    # 4. Train
    print("[Joint] Full training phase.")
    trainer.fit(
        train_loader=train_loader,
        val_loader=None,
        total_steps=args.steps,
        validate_every=0,
        log_every=10,
        amp=True,
        checkpoint_every=50,
        batch_transform=parse_batch,
        compute_recon=True,
        compute_trans=True,
        profile_steps=args.profile_steps,
    )
    print(f"--- Finished ! Metrics in {run_dir} ---")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path", type=str, required=True, help="Path to kdd_processed.pt"
    )
    p.add_argument("--config", type=str, nargs="+", required=True)
    p.add_argument("--set", type=str, nargs="+", default=None)
    p.add_argument("--work_dir", type=str, default="runs/kdd_train")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seq_len", type=int, default=120)  # L1+L2
    p.add_argument("--split", type=int, default=72)  # L1
    p.add_argument(
        "--profile_steps",
        type=int,
        default=10,
        help="Number of steps to profile with torch.profiler (default: 10). Set to 0 to disable profiling.",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    main(args)
