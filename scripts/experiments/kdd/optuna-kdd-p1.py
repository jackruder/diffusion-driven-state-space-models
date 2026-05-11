"""Optuna hyperparameter search (phase 1) for the KDD Cup 2018 experiment.

Phase 1 trains only the encoder/decoder/init-prior (``recon_only`` mode) and
tunes hyperparameters that affect the reconstruction objective.

Usage::

    python scripts/experiments/kdd/optuna-kdd-p1.py \\
        --storage sqlite:///runs/optuna/kdd/phase1.db \\
        --study_name ddssm_phase1 \\
        --n_trials 50 \\
        --config configs/base.yaml configs/kdd.yaml

Requires Optuna and (optionally) optuna-dashboard for live visualisation.
"""

import csv
import json
import math
from pathlib import Path
import argparse
import subprocess

import optuna
from optuna.trial import FrozenTrial


def _ensure_optuna_storage_dir(storage_url: str) -> None:
    # Handles local sqlite URLs like: sqlite:///runs/optuna/phase1/study.db
    prefix = "sqlite:///"
    if not storage_url.startswith(prefix):
        return
    db_path = Path(storage_url[len(prefix) :])
    db_path.parent.mkdir(parents=True, exist_ok=True)


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
    direct = eval_tag_dir / "metrics.json"
    if direct.is_file():
        return direct

    candidates = [p for p in eval_tag_dir.rglob("metrics.json") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No metrics.json found under {eval_tag_dir}")

    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _read_csv_column(csv_path: Path, col: str) -> list[float]:
    """Read a single numeric column from a CSV, skipping blank/unparseable rows."""
    values: list[float] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(col, "")
            if raw == "" or raw is None:
                continue
            try:
                values.append(float(raw))
            except (ValueError, TypeError):
                continue
    return values


def _find_train_csv(train_dir: Path) -> Path | None:
    """Locate the train_metrics.csv written by CSVLogger.

    Expected at: <train_dir>/latest/csv_logs/train_metrics.csv
    Falls back to recursive search.
    """
    expected = train_dir / "latest" / "csv_logs" / "train_metrics.csv"
    if expected.is_file():
        return expected
    candidates = sorted(train_dir.rglob("train_metrics.csv"))
    return candidates[-1] if candidates else None


def _pick_recon_column(csv_path: Path) -> str | None:
    """Return the first column name that looks like reconstruction loss."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
    # Prefer the exact distortion/rec key logged by DDSSM forward()
    for candidate in ("loss/distortion/rec", "loss/total"):
        if candidate in headers:
            return candidate
    # Fuzzy fallback
    for h in headers:
        if "recon" in h.lower() or "distortion" in h.lower():
            return h
    return None


def check_recon_divergence(
    train_dir: Path,
    spike_factor: float = 5.0,
    tail_fraction: float = 0.2,
    min_rows: int = 10,
) -> tuple[bool, str]:
    """Parse the training CSV for reconstruction-error spikes.

    Returns (diverged, reason).

    Checks
    ------
    1. Any NaN / Inf in the recon column.
    2. Mean of the final *tail_fraction* of values exceeds *spike_factor* x
       the median of the first half -- a clear blow-up.
    """
    csv_path = _find_train_csv(train_dir)
    if csv_path is None:
        return False, "no train_metrics.csv found"

    col = _pick_recon_column(csv_path)
    if col is None:
        return False, "no recon/distortion column in CSV"

    values = _read_csv_column(csv_path, col)
    if len(values) < min_rows:
        return False, f"only {len(values)} rows (<{min_rows}), skipping check"

    # Check 1: non-finite values
    if any(not math.isfinite(v) for v in values):
        return True, f"non-finite {col} detected"

    # Check 2: tail-vs-first-half spike
    n = len(values)
    half = n // 2
    tail_start = max(int(n * (1.0 - tail_fraction)), half)  # never overlap

    first_half_sorted = sorted(values[:half])
    median_first = first_half_sorted[len(first_half_sorted) // 2]

    tail = values[tail_start:]
    mean_tail = sum(tail) / len(tail)

    if median_first > 0 and mean_tail > spike_factor * median_first:
        return True, (
            f"{col} spike: tail mean={mean_tail:.4f} > "
            f"{spike_factor}x first-half median={median_first:.4f}"
        )

    return False, "ok"


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

        # lambda_schedule = trial.suggest_categorical(
        #     "hyperparams.lambda_schedule", ["linear", "cosine"]
        # )
        lambda_warmup_steps = trial.suggest_int(
            "hyperparams.lambda_warmup_steps", 1, int(0.75 * args.train_steps)
        )
        lambda_end = trial.suggest_float("hyperparams.lambda_end", 0.7, 2.0, log=True)
        vae_lr = trial.suggest_float("vae_lr", 1e-5, 1e-3, log=True)
        trans_lr = trial.suggest_float("trans_lr", 1e-5, 3e-4, log=True)

        weight_decay = trial.suggest_float(
            "hyperparams.weight_decay", 1e-5, 1e-2, log=True
        )
        batch_size = trial.suggest_categorical("hyperparams.batch_size", [32])

        # Fix S and S_k to 1 and 4 for fast tuning sweeps. Can increase during final training.
        S = 1
        S_k = 4

        overrides = [
            "hyperparams.lambda_schedule=cosine",
            f"hyperparams.lambda_warmup_steps={lambda_warmup_steps}",
            f"hyperparams.lambda_end={lambda_end}",
            f"hyperparams.enc_lr={vae_lr}",
            f"hyperparams.dec_lr={vae_lr}",
            f"hyperparams.trans_lr={trans_lr}",
            f"hyperparams.zinit_lr={vae_lr}",
            f"hyperparams.weight_decay={weight_decay}",
            f"hyperparams.S={S}",
            f"hyperparams.batch_size={batch_size}",
            f"transition.schedule.S_k={S_k}",
            f"transition.schedule.k_chunk={S_k}",
        ]

        train_cmd = [
            "uv",
            "run",
            "python",
            "scripts/experiments/kdd/kdd_train.py",
            "--data_path",
            args.data_path,
            "--config",
            *args.config,
            "--work_dir",
            str(train_dir),
            "--steps",
            str(args.train_steps),
            "--profile_steps",
            str(args.profile_steps),
            "--set",
            *overrides,
        ]
        if args.quiet:
            train_cmd.append("--quiet")
        run_cmd(train_cmd)

        # --- Prune diverged trials before expensive evaluation ---
        diverged, reason = check_recon_divergence(
            train_dir,
            spike_factor=args.prune_spike_factor,
            tail_fraction=args.prune_tail_fraction,
        )
        if diverged:
            print(
                f"=== Trial {trial.number} PRUNED (recon divergence): {reason} ===",
                flush=True,
            )
            trial.set_user_attr("prune_reason", reason)
            raise optuna.TrialPruned(reason)

        ckpt = train_dir / "latest" / "checkpoints" / "ckpt_latest.pth"
        if not ckpt.exists():
            candidates = sorted(
                (train_dir / args.mode).glob("*/checkpoints/ckpt_latest.pth")
            )
            if not candidates:
                raise FileNotFoundError(f"missing checkpoint: {ckpt}")
            ckpt = candidates[-1]
            print(f"[checkpoint] fallback checkpoint: {ckpt}")

        eval_cmd = [
            "uv",
            "run",
            "python",
            "scripts/experiments/kdd/evaluate_kdd.py",
            "--data_path",
            args.data_path,
            "--config",
            *args.config,
            "--resume",
            str(ckpt),
            "--work_dir",
            str(eval_dir),
            "--split",
            str(args.split),
            "--seq_len",
            str(args.seq_len),
            "--batch_size",
            str(batch_size),
            "--forecast_samples",
            str(args.forecast_samples),
            "--seed",
            str(args.eval_seed),
        ]
        run_cmd(eval_cmd)

        metrics_path = find_latest_metrics_json(eval_dir)
        with open(metrics_path, "r") as f:
            m = json.load(f)

        for k in ("crps_sum", "mae"):
            v = m.get(k, None)
            if v is not None and (not math.isfinite(v)):
                raise ValueError(f"Non-finite metric {k}: {v}")

        obj = float(m["crps_sum"])  # optimize CRPS-sum
        trial.set_user_attr("metrics_path", str(metrics_path))
        trial.set_user_attr("checkpoint", str(ckpt))
        trial.set_user_attr("mae", float(m.get("mae", float("nan"))))
        trial.set_user_attr("crps_sum", float(m.get("crps_sum", float("nan"))))

        print(
            f"=== Trial {trial.number} done: objective={obj:.6f} "
            f"(mae={m.get('mae', float('nan')):.6f}, "
            f"crps_sum={m.get('crps_sum', float('nan')):.6f}) ===",
            flush=True,
        )

        return obj

    return objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study_name", type=str, default="ddssm_phase1")
    p.add_argument(
        "--storage", type=str, default="sqlite:///runs/optuna/phase1/study.db"
    )
    p.add_argument("--work_root", type=str, default="runs/optuna")
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument(
        "--data_path",
        type=str,
        default="data/kdd_processed.pt",
        help="Path to processed KDD tensor payload.",
    )
    p.add_argument("--mode", type=str, default="harmonic")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--eval_seed", type=int, default=123)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--split", type=int, default=16)
    p.add_argument("--train_steps", type=int, default=1000)
    p.add_argument("--forecast_samples", type=int, default=20)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_startup_trials", type=int, default=10)
    p.add_argument("--quiet", action="store_true", help="Silence inner training logs.")
    p.add_argument(
        "--profile_steps",
        type=int,
        default=0,
        help="Profile first N optimizer steps in training (0 disables).",
    )
    p.add_argument(
        "--prune_spike_factor",
        type=float,
        default=5.0,
        help="Prune if tail recon loss exceeds this multiple of first-half median.",
    )
    p.add_argument(
        "--prune_tail_fraction",
        type=float,
        default=0.2,
        help="Fraction of training tail to average for spike detection.",
    )
    p.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of concurrent Optuna trials in this process.",
    )
    args = p.parse_args()

    if not (1 <= args.split < args.seq_len):
        raise ValueError(
            f"Require 1 <= split < seq_len, got split={args.split}, seq_len={args.seq_len}"
        )

    _ensure_optuna_storage_dir(args.storage)

    sampler = optuna.samplers.TPESampler(
        seed=args.seed, n_startup_trials=args.n_startup_trials
    )
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
        catch=(Exception,),
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
