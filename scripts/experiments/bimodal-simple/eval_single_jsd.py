import argparse
import csv
import os
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from dssd.dssd import DSSD_base
from dssd.data.synthetic import SyntheticDataset
from dssd.config import load_config_from_files

EPS = 1e-12


def normal_pdf(x: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def jsd_discrete(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


def histogram_mass(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(vals, bins=edges, density=False)
    h = h.astype(np.float64)
    return np.ones_like(h) / h.size if h.sum() <= 0 else h / h.sum()


def analytic_truth_mass_per_example(
    centers: np.ndarray,
    x_prev: float,
    a: float = 0.9,
    step_size: float = 4.0,
    sigma: float = 0.2,
) -> np.ndarray:
    shift = (0.9 - a) * x_prev
    q_pdf = 0.5 * normal_pdf(centers, shift - step_size, sigma) + 0.5 * normal_pdf(
        centers, shift + step_size, sigma
    )
    q_pdf = np.clip(q_pdf, EPS, None)
    return q_pdf / q_pdf.sum()


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def pick_two_median_indices(jsd_arr: np.ndarray, sample_idx_arr: np.ndarray):
    order = np.argsort(jsd_arr)
    n = len(jsd_arr)
    if n == 0:
        return None, None

    if n % 2 == 0:
        r1, r2 = n // 2 - 1, n // 2
    else:
        # deterministic pair around median
        r1 = n // 2
        r2 = min(n // 2 + 1, n - 1)

    i1 = int(sample_idx_arr[order[r1]])
    i2 = int(sample_idx_arr[order[r2]])
    return i1, i2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=None)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--baseline", type=str, default="model", choices=["model", "locf"])
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--out_npz", type=str, default="")
    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--seq_len", type=int, default=48)
    p.add_argument("--split", type=int, default=47)
    p.add_argument("--n_series", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--forecast_samples", type=int, default=1024)
    p.add_argument("--dataset_seed", type=int, default=123)
    p.add_argument("--center_coef", type=float, default=0.9)
    p.add_argument(
        "--dataset_split", type=str, default="val", choices=["train", "val", "test"]
    )
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--no_print", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.baseline == "model":
        if not args.config or not args.resume:
            raise ValueError("--config and --resume are required when --baseline=model")
        cfg = load_config_from_files(args.config, None)
        data_dim = cfg.data_dim
    else:
        data_dim = 1

    ds = SyntheticDataset(
        mode="bimodal",
        split=args.dataset_split,
        N_per_split=args.n_series,
        T=args.seq_len,
        D=data_dim,
        dataset_seed=args.dataset_seed,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = None
    if args.baseline == "model":
        model = DSSD_base(cfg, device).to(device)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(
            ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)), strict=True
        )
        model.eval()

    edges = np.linspace(-10.0, 10.0, 301)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows = []
    sample_idx = 0
    all_sample_idx, all_x_prev, all_xhat_samples, all_model_mass, all_truth_mass = (
        [],
        [],
        [],
        [],
        [],
    )

    with torch.no_grad():
        for batch in dl:
            obs = batch["observed_data"].to(device)
            tp = batch["timepoints"].to(device)
            B = obs.shape[0]

            if args.baseline == "model":
                mask = torch.ones_like(obs).to(device)
                out = model.forecast(
                    x_hist=obs[..., : args.split],
                    x_mask=mask[..., : args.split],
                    past_time=tp[:, : args.split],
                    future_time=tp[:, args.split :],
                    num_samples=args.forecast_samples,
                )
                pred = out["pred_samples"]  # (B,S,D,L2)
                if pred.shape[-1] != 1:
                    raise ValueError(
                        f"Expected one-step horizon (L2=1), got {pred.shape[-1]}"
                    )
                xhat_batch = pred[:, :, 0, 0].detach().cpu().numpy()  # (B,S)
            else:
                x_prev_np = obs[:, 0, args.split - 1].detach().cpu().numpy()  # (B,)
                xhat_batch = np.repeat(
                    x_prev_np[:, None], repeats=args.forecast_samples, axis=1
                ).astype(np.float32)

            for b in range(B):
                vals = xhat_batch[b]
                x_prev = float(obs[b, 0, args.split - 1].detach().cpu().numpy())

                ctr_vals = vals - args.center_coef * x_prev
                p_mass = histogram_mass(ctr_vals, edges)
                q_mass = analytic_truth_mass_per_example(
                    centers, x_prev, a=args.center_coef
                )
                jsd = float(jsd_discrete(p_mass, q_mass))

                rows.append({"sample_idx": sample_idx, "x_prev": x_prev, "jsd": jsd})
                all_sample_idx.append(sample_idx)
                all_x_prev.append(x_prev)
                all_xhat_samples.append(vals.astype(np.float32, copy=True))
                all_model_mass.append(p_mass.astype(np.float32, copy=True))
                all_truth_mass.append(q_mass.astype(np.float32, copy=True))
                sample_idx += 1

    jsd_arr = np.array([r["jsd"] for r in rows], dtype=np.float64)
    idx_arr = np.array([r["sample_idx"] for r in rows], dtype=np.int64)

    n = int(len(jsd_arr))
    mean = float(jsd_arr.mean()) if n > 0 else float("nan")
    std = float(jsd_arr.std()) if n > 0 else float("nan")
    sem = float(std / math.sqrt(n)) if n > 0 else float("nan")
    median = float(np.median(jsd_arr)) if n > 0 else float("nan")

    median_idx_1, median_idx_2 = pick_two_median_indices(jsd_arr, idx_arr)
    best_idx = int(idx_arr[np.argmin(jsd_arr)]) if n > 0 else None
    worst_idx = int(idx_arr[np.argmax(jsd_arr)]) if n > 0 else None

    if args.out_csv:
        ensure_parent(args.out_csv)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_idx", "x_prev", "jsd"])
            for r in rows:
                w.writerow([r["sample_idx"], r["x_prev"], r["jsd"]])

    if args.out_npz:
        ensure_parent(args.out_npz)
        np.savez_compressed(
            args.out_npz,
            sample_idx=np.asarray(all_sample_idx, dtype=np.int64),
            x_prev=np.asarray(all_x_prev, dtype=np.float32),
            xhat_samples=np.stack(all_xhat_samples, axis=0),  # (N,S)
            edges=edges.astype(np.float32),
            centers=centers.astype(np.float32),
            model_mass=np.stack(all_model_mass, axis=0),  # (N,B)
            truth_mass=np.stack(all_truth_mass, axis=0),  # (N,B)
        )

    summary = {
        "dataset_split": args.dataset_split,
        "dataset_seed": args.dataset_seed,
        "n": n,
        "jsd_centered_mean": mean,
        "jsd_centered_std": std,
        "jsd_centered_sem": sem,
        "jsd_centered_median": median,
        "median_sample_idx_1": median_idx_1,
        "median_sample_idx_2": median_idx_2,
        "best_sample_idx": best_idx,
        "worst_sample_idx": worst_idx,
        "out_csv": args.out_csv if args.out_csv else None,
        "center_coef": args.center_coef,
    }

    ensure_parent(args.out_json)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_print:
        if args.out_csv:
            print(f"[Saved] {args.out_csv}")
        print(f"[Saved] {args.out_json}")
    print(
        f"[JSD centered] mean={mean:.6f} std={std:.6f} sem={sem:.6f} "
        f"median={median:.6f} median_idxs=({median_idx_1}, {median_idx_2})"
    )


if __name__ == "__main__":
    main()
