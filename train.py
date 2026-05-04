"""Hydra entry point for DDSSM training with optional SLURM/Optuna sweep support.

Single run (local)::

    python train.py dataset=kdd steps=5000

    # Override a model hyperparameter
    python train.py dataset=kdd hp.vae_lr=3e-4

SLURM hyperparameter sweep (phase 1)::

    python train.py --multirun \\
        +sweep=kdd_p1 \\
        hydra/launcher=submitit_slurm \\
        hydra/sweeper=optuna \\
        "++hydra.sweeper.storage=sqlite:///runs/optuna/kdd_p1.db" \\
        "++hydra.launcher.partition=gpu"

The function returns the validation CRPS-sum when ``do_eval=true``, which the
Optuna sweeper uses as the minimization objective.  For plain training runs
(``do_eval=false``) the return value is ``None`` and no evaluation is performed.
"""

import csv
import json
import math
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ddssm.config import DDSSMConfig, load_config_from_files
from ddssm.data.dataload import build_loaders_for_expt, parse_batch
from ddssm.ddssm import DDSSM_base
from ddssm.train import DDSSMTrainer


# ---------------------------------------------------------------------------
# Helpers (inlined from scripts/experiments/kdd/ for subprocess-free operation)
# ---------------------------------------------------------------------------


def _read_csv_column(csv_path: Path, col: str) -> list[float]:
    values: list[float] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(col, "")
            if not raw:
                continue
            try:
                values.append(float(raw))
            except (ValueError, TypeError):
                continue
    return values


def _find_train_csv(run_dir: Path) -> Path | None:
    expected = run_dir / "csv_logs" / "train_metrics.csv"
    if expected.is_file():
        return expected
    candidates = sorted(run_dir.rglob("train_metrics.csv"))
    return candidates[-1] if candidates else None


def _pick_recon_column(csv_path: Path) -> str | None:
    with open(csv_path) as f:
        headers = csv.DictReader(f).fieldnames or []
    for candidate in ("loss/distortion/rec", "loss/total"):
        if candidate in headers:
            return candidate
    for h in headers:
        if "recon" in h.lower() or "distortion" in h.lower():
            return h
    return None


def _check_recon_divergence(
    run_dir: Path,
    spike_factor: float = 5.0,
    tail_fraction: float = 0.2,
    min_rows: int = 10,
) -> tuple[bool, str]:
    """Return (diverged, reason) by inspecting the training CSV."""
    csv_path = _find_train_csv(run_dir)
    if csv_path is None:
        return False, "no train_metrics.csv found"

    col = _pick_recon_column(csv_path)
    if col is None:
        return False, "no recon/distortion column in CSV"

    values = _read_csv_column(csv_path, col)
    if len(values) < min_rows:
        return False, f"only {len(values)} rows (<{min_rows}), skipping check"

    if any(not math.isfinite(v) for v in values):
        return True, f"non-finite {col} detected"

    n = len(values)
    half = n // 2
    tail_start = max(int(n * (1.0 - tail_fraction)), half)
    first_half_sorted = sorted(values[:half])
    median_first = first_half_sorted[len(first_half_sorted) // 2]
    mean_tail = sum(values[tail_start:]) / len(values[tail_start:])

    if median_first > 0 and mean_tail > spike_factor * median_first:
        return True, (
            f"{col} spike: tail mean={mean_tail:.4f} > "
            f"{spike_factor}x first-half median={median_first:.4f}"
        )

    return False, "ok"


def _mae_metrics(
    pred_mean: torch.Tensor, y_future: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    abs_err = (pred_mean - y_future).abs()
    return abs_err.mean(), abs_err.mean(dim=(0, 1))


def _crps_sum_metrics(
    pred_samples: torch.Tensor, y_future: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    levels = torch.arange(0.05, 1.0, 0.05, device=pred_samples.device)
    pred_sum = pred_samples.sum(dim=2)  # (B,S,L2)
    y_sum = y_future.sum(dim=1)  # (B,L2)
    qs = torch.quantile(pred_sum, levels, dim=1).permute(1, 2, 0)  # (B,L2,Q)
    indicator = (y_sum.unsqueeze(-1) < qs).float()
    crps = 2 * ((levels - indicator) * (y_sum.unsqueeze(-1) - qs)).mean(dim=-1)
    return crps.mean(), crps.mean(dim=0)


def _collect_overrides(cfg: DictConfig) -> list[str]:
    """Translate ``cfg.hp.*`` keys into dot-notation DDSSMConfig overrides."""
    hp_map: dict[str, str] = {
        "batch_size": "hyperparams.batch_size",
        "lambda_schedule": "hyperparams.lambda_schedule",
        "lambda_warmup_steps": "hyperparams.lambda_warmup_steps",
        "lambda_end": "hyperparams.lambda_end",
        "enc_lr": "hyperparams.enc_lr",
        "dec_lr": "hyperparams.dec_lr",
        "zinit_lr": "hyperparams.zinit_lr",
        "trans_lr": "hyperparams.trans_lr",
        "weight_decay": "hyperparams.weight_decay",
        "S": "hyperparams.S",
        "S_k": "transition.schedule.S_k",
        "k_chunk": "transition.schedule.k_chunk",
    }
    hp: dict = OmegaConf.to_container(cfg.hp, resolve=True) if cfg.get("hp") else {}  # type: ignore[assignment]
    overrides: list[str] = []

    for key, path in hp_map.items():
        val = hp.get(key)
        if val is not None:
            overrides.append(f"{path}={val}")

    # vae_lr shorthand → enc_lr, dec_lr, zinit_lr (individual keys take precedence)
    vae_lr = hp.get("vae_lr")
    if vae_lr is not None:
        already_set = {o.split("=")[0] for o in overrides}
        for lr_path in (
            "hyperparams.enc_lr",
            "hyperparams.dec_lr",
            "hyperparams.zinit_lr",
        ):
            if lr_path not in already_set:
                overrides.append(f"{lr_path}={vae_lr}")

    return overrides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float | None:
    """Train (and optionally evaluate) a DDSSM model.

    Returns the validation CRPS-sum when ``cfg.do_eval=true`` so that the
    Optuna sweeper can use it as the minimization objective.
    """
    import yaml  # local import keeps top-level clean

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 64
    torch._dynamo.config.recompile_limit = 64
    inductor_config.pad_outputs = False
    inductor_config.pad_dynamic_shapes = False
    inductor_config.force_shape_pad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        _d = torch.randn(8, 8, device=device) @ torch.randn(8, 8, device=device)
        del _d
        torch.cuda.synchronize()

    print(f"--- DDSSM Training | device={device} ---")

    # Resolve model-config and data paths against the *original* working
    # directory (Hydra changes cwd to the job output dir).
    data_path = to_absolute_path(cfg.dataset.data_path)
    model_configs = [to_absolute_path(p) for p in cfg.model_configs]

    # --- Load and merge model architecture configs, then apply hp overrides ---
    overrides = _collect_overrides(cfg)
    config = load_config_from_files(model_configs, overrides)

    # --- Load preprocessed data ---
    print(f"Loading data from {data_path}...")
    data_payload = torch.load(data_path, weights_only=False)
    series_list = data_payload["series_list"]
    covariates_list = data_payload["covariates_list"]
    static_covariates = data_payload["static_covariates"]

    # Inject dataset-derived constants into the config
    config_dict = config.model_dump()
    config_dict["data_dim"] = len(series_list)
    config_dict["covariate_dim"] = covariates_list[0].shape[0]
    config_dict["num_classes_per_static"] = [
        int(static_covariates[:, 0].max() + 1),
        int(static_covariates[:, 1].max() + 1),
        int(static_covariates[:, 2].max() + 1),
    ]
    config = DDSSMConfig.model_validate(config_dict)

    L1, L2 = cfg.split, cfg.seq_len - cfg.split

    train_loader, val_loader, _, _ = build_loaders_for_expt(
        series_list=series_list,
        L1=L1,
        L2=L2,
        val_windows=24,
        test_windows=27,
        eval_step_size=24,
        batch_size=config.hyperparams.batch_size,
        num_train_batches_per_epoch=None,
        covariates_list=covariates_list,
        static_covariates=static_covariates,
        backend="torch",
        normalize=True,
        device=device,
    )

    # Hydra sets cwd to the job output directory, which becomes the run dir.
    run_dir = Path(os.getcwd())
    config.checkpoint_dir = str(run_dir / "checkpoints")

    # Persist the resolved model config alongside the Hydra-generated config.
    with open(run_dir / "model_config.yaml", "w") as f:
        yaml.safe_dump(config.model_dump(), f)

    model = DDSSM_base(config, device)
    trainer = DDSSMTrainer(
        model,
        device=device,
        tensorboard_dir=str(run_dir / "tb_logs"),
        csv_log_path=str(run_dir / "csv_logs" / "train_metrics.csv"),
        quiet=cfg.quiet,
    )

    print(f"[Training] steps={cfg.steps}")
    trainer.fit(
        train_loader=train_loader,
        val_loader=None,
        total_steps=cfg.steps,
        validate_every=0,
        log_every=10,
        amp=True,
        checkpoint_every=50,
        batch_transform=parse_batch,
        compute_recon=True,
        compute_trans=True,
        profile_steps=cfg.profile_steps,
    )
    print(f"--- Training done. Outputs in {run_dir} ---")

    if not cfg.do_eval:
        return None

    # --- Divergence pruning (Optuna-aware) ---
    try:
        import optuna

        diverged, reason = _check_recon_divergence(
            run_dir,
            spike_factor=cfg.prune_spike_factor,
            tail_fraction=cfg.prune_tail_fraction,
        )
        if diverged:
            print(f"=== PRUNED (divergence): {reason} ===")
            raise optuna.TrialPruned(reason)
    except ImportError:
        pass

    # --- Evaluate ---
    ckpt = run_dir / "checkpoints" / "ckpt_latest.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    torch.manual_seed(cfg.eval_seed)
    model.load_state_dict(
        torch.load(ckpt, map_location=device)["model_state"], strict=True
    )
    model.eval()

    acc_mae: list[float] = []
    acc_mae_per_t: list[list[float]] = []
    acc_crps: list[float] = []
    acc_crps_per_t: list[list[float]] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = parse_batch(batch, device)
            x_hist = batch["observed_data"][..., :L1]
            x_mask = batch["observation_mask"][..., :L1]
            past_time = batch["timepoints"][:, :L1]
            future_time = batch["timepoints"][:, L1:]
            y_future = batch["observed_data"][..., L1:]
            static_cov = batch.get("static_covariates")
            covariates = batch.get("covariates")

            out = model.forecast(
                x_hist=x_hist,
                x_mask=x_mask,
                past_time=past_time,
                future_time=future_time,
                past_covariates=covariates[..., :L1] if covariates is not None else None,
                future_covariates=covariates[..., L1:] if covariates is not None else None,
                static_covariates=static_cov,
                num_samples=cfg.forecast_samples,
            )
            pred_samples = out["pred_samples"]
            pred_mean = pred_samples.mean(dim=1)

            mae_g, mae_t = _mae_metrics(pred_mean, y_future)
            crps_g, crps_t = _crps_sum_metrics(pred_samples, y_future)

            acc_mae.append(float(mae_g))
            acc_mae_per_t.append(mae_t.cpu().numpy().tolist())
            acc_crps.append(float(crps_g))
            acc_crps_per_t.append(crps_t.cpu().numpy().tolist())

    metrics = {
        "mae": float(np.mean(acc_mae)),
        "mae_per_t": np.mean(np.stack(acc_mae_per_t, axis=0), axis=0).tolist(),
        "crps_sum": float(np.mean(acc_crps)),
        "crps_sum_per_t": np.mean(
            np.stack(acc_crps_per_t, axis=0), axis=0
        ).tolist(),
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[Eval] crps_sum={metrics['crps_sum']:.6f}  mae={metrics['mae']:.6f}"
    )
    return metrics["crps_sum"]


if __name__ == "__main__":
    main()
