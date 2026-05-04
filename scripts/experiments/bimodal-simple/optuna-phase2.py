import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import optuna
import datetime
from optuna.trial import FrozenTrial

import numpy as np
import torch


def run_cmd(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def optuna_log_callback(study: optuna.Study, trial: FrozenTrial):
    best = study.best_trial
    print(
        f"[trial={trial.number}] state={trial.state.name} value={trial.value} "
        f"best_trial={best.number} best_value={best.value}",
        flush=True,
    )


def find_latest_metrics_json(eval_tag_dir: Path) -> Path:
    candidates = sorted([p for p in eval_tag_dir.glob("*/metrics.json") if p.is_file()])
    if not candidates:
        raise FileNotFoundError(f"No metrics.json found under {eval_tag_dir}")
    return candidates[-1]


def objective_factory(args):
    def objective(trial: optuna.Trial) -> float:
        print(f"\n=== Trial {trial.number} start ===", flush=True)
        trial_dir = (
            Path(args.work_root)
            / "phase1"
            / args.study_name
            / f"trial_{trial.number:04d}"
        )
        train_dir = trial_dir / "train"
        eval_dir = trial_dir / "eval"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        lambda_schedule = trial.suggest_categorical(
            "hyperparams.lambda_schedule", ["cosine"]
        )
        lambda_warmup_steps = trial.suggest_int(
            "hyperparams.lambda_warmup_steps", 400, 1200
        )
        lambda_end = trial.suggest_float("hyperparams.lambda_end", 0.8, 1.2)
        vae_lr = trial.suggest_float("hyperparams.vae_lr", 3e-4, 1.5e-3, log=True)
        trans_lr = trial.suggest_float("hyperparams.trans_lr", 2e-4, 3e-3, log=True)
        S = 1
        batch_size = 256
        overrides = [
            f"hyperparams.lambda_schedule={lambda_schedule}",
            f"hyperparams.lambda_warmup_steps={lambda_warmup_steps}",
            f"hyperparams.lambda_end={lambda_end}",
            f"hyperparams.enc_lr={vae_lr}",
            f"hyperparams.dec_lr={vae_lr}",
            f"hyperparams.trans_lr={trans_lr}",
            f"hyperparams.zinit_lr={vae_lr}",
            f"hyperparams.S={S}",
            f"hyperparams.batch_size={batch_size}",
        ]

        # This will only affect runs using diffusion override.
        is_diffusion = any("diff" in str(c).lower() for c in args.config)
        if is_diffusion:
            S_k = trial.suggest_int("transition.schedule.S_k", 3, 8)
            overrides.append(f"transition.schedule.S_k={S_k}")
            overrides.append(f"transition.schedule.k_chunk={S_k}")

        train_cmd = [
            "uv",
            "run",
            "python",
            "scripts/experiments/verifications.py",
            "--mode",
            args.mode,
            "--config",
            *args.config,
            "--work_dir",
            str(train_dir),
            "--training_mode",
            "joint",
            "--steps",
            str(args.train_steps),
            "--seq_len",
            str(args.seq_len),
            "--split",
            str(args.split),
            "--set",
            *overrides,
            "--dataset_seed",
            str(args.dataset_seed),
        ]
        if not args.quiet:
            pass
        else:
            train_cmd.append("--quiet")

        run_cmd(train_cmd)

        ckpt = train_dir / args.mode / "latest" / "checkpoints" / "ckpt_latest.pth"
        if not ckpt.exists():
            candidates = sorted(
                (train_dir / args.mode).glob("*/checkpoints/ckpt_latest.pth")
            )
            if not candidates:
                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
            ckpt = candidates[-1]
            print(f"[Checkpoint] Fallback checkpoint: {ckpt}")

        eval_cmd = [
            "uv",
            "run",
            "python",
            "scripts/experiments/evaluate_models.py",
            "--mode",
            args.mode,
            "--config",
            *args.config,
            "--resume",
            str(ckpt),
            "--work_dir",
            str(eval_dir),
            "--tag",
            "joint",
            "--seq_len",
            str(args.seq_len),
            "--split",
            str(args.split),
            "--num_eval_sequences",
            str(args.num_eval_sequences),
            "--batch_size",
            str(args.eval_batch_size),
            "--forecast_samples",
            str(args.forecast_samples),
            "--seed",
            str(args.eval_seed),
            "--dataset_seed",
            str(args.dataset_seed),
            "--dataset_split",
            "val",
        ]
        run_cmd(eval_cmd)

        jsd_json = eval_dir / "jsd_metrics.json"
        jsd_cmd = [
            "uv",
            "run",
            "python",
            "scripts/experiments/bimodal-simple/eval_single_jsd.py",
            "--config",
            *args.config,
            "--resume",
            str(ckpt),
            "--seq_len",
            str(args.seq_len),
            "--split",
            str(args.seq_len - 1),
            "--n_series",
            str(args.num_eval_sequences),
            "--batch_size",
            str(args.eval_batch_size),
            "--forecast_samples",
            str(args.forecast_samples),
            "--out_json",
            str(jsd_json),
        ]
        run_cmd(jsd_cmd)

        metrics_path = find_latest_metrics_json(eval_dir / args.mode / "joint")
        with open(metrics_path, "r") as f:
            m = json.load(f)

        with open(jsd_json, "r") as f:
            jsd_data = json.load(f)

        for k in ("sum_crps", "energy_score_mean", "mae_mean"):
            v = m.get(k, None)
            if v is not None and (not math.isfinite(v)):
                raise ValueError(f"Non-finite metric {k}: {v}")

        obj = float(jsd_data["jsd_centered_mean"])

        trial.set_user_attr("metrics_path", str(metrics_path))
        trial.set_user_attr("checkpoint", str(ckpt))
        trial.set_user_attr("jsd_centered_mean", obj)
        trial.set_user_attr("mae_mean", float(m.get("mae_mean", float("nan"))))
        trial.set_user_attr("sum_crps", float(m.get("sum_crps", float("nan"))))
        trial.set_user_attr(
            "energy_score_mean", float(m.get("energy_score_mean", float("nan")))
        )

        print(
            f"=== Trial {trial.number} done: objective={obj:.6f} "
            f"(mae={m.get('mae_mean', float('nan')):.6f}, "
            f"crps={m.get('sum_crps', float('nan')):.6f}, "
            f"es={m.get('energy_score_mean', float('nan')):.6f}) ===",
            flush=True,
        )
        return obj

    return objective


def main():
    # set high precision for matmul 32
    torch.set_float32_matmul_precision("high")
    p = argparse.ArgumentParser()
    p.add_argument("--study_name", type=str, default="ddssm_phase1")
    p.add_argument(
        "--storage", type=str, default="sqlite:///runs/optuna/phase1/study.db"
    )
    p.add_argument("--work_root", type=str, default="runs/optuna")
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--mode", type=str, default="harmonic")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--eval_seed", type=int, default=123)
    p.add_argument("--dataset_seed", type=int, default=123)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--split", type=int, default=16)
    p.add_argument("--train_steps", type=int, default=1000)
    p.add_argument("--num_eval_sequences", type=int, default=1000)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--forecast_samples", type=int, default=20)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of concurrent Optuna trials in this process.",
    )
    p.add_argument("--n_startup_trials", type=int, default=10)
    p.add_argument("--objective", choices=["1d", "2d"], default="1d")
    p.add_argument("--tune_zinit_lr", action="store_true")
    p.add_argument("--quiet", action="store_true", help="Silence inner training logs.")
    args = p.parse_args()

    if not (1 <= args.split < args.seq_len):
        raise ValueError(
            f"Require 1 <= split < seq_len, got split={args.split}, seq_len={args.seq_len}"
        )

    Path("runs/optuna/phase1").mkdir(parents=True, exist_ok=True)

    # sampler = optuna.samplers.TPESampler(
    #     seed=args.seed, n_startup_trials=args.n_startup_trials
    # )

    sampler = optuna.samplers.TPESampler(n_startup_trials=args.n_startup_trials)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(
        objective_factory(args),
        n_trials=args.n_trials,
        callbacks=[optuna_log_callback],
        show_progress_bar=True,
        n_jobs=args.n_jobs,
    )

    out_dir = Path(args.work_root) / "phase1" / args.study_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
    }
    with open(out_dir / "best.json", "w") as f:
        json.dump(best, f, indent=2)

    rows = []
    for t in study.trials:
        row = {
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            **t.params,
            **{f"attr_{k}": v for k, v in t.user_attrs.items()},
        }
        rows.append(row)
    with open(out_dir / "trials.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("[Done] Best:", best)


if __name__ == "__main__":
    main()
