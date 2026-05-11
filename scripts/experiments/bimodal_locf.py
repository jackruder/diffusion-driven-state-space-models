"""LOCF (last-observation-carried-forward) baseline for bimodal JSD analysis.

Drives the synthetic bimodal data loader without any trained model and writes
an NPZ in the same schema as the ``bimodal_jsd`` eval metric. The NPZ can be
fed straight into ``scripts/experiments/plot_bimodal_compare.py`` alongside
NPZs from real model runs (Gaussian / Diffusion).

Usage::

    python scripts/experiments/bimodal_locf.py \\
        --out runs/locf/bimodal_jsd.npz \\
        --n_per_split 256 --T 32 --T_split 31 --num_samples 1024

The defaults mirror the legacy ``bimodal_jsd.py --baseline=locf`` invocation:
T=32, T_split=31 → strict one-step horizon.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ddssm.data.synthetic import SyntheticDataset
from ddssm.eval.metrics import (
    _bimodal_truth_mass,
    _hist_mass,
    _jsd_discrete,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Output NPZ path.")
    p.add_argument("--summary_json", default=None,
                   help="Optional path to write a summary JSON next to the NPZ.")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n_per_split", type=int, default=1024)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--T_split", type=int, default=31,
                   help="Past/future boundary index (default 31 → strict one-step).")
    p.add_argument("--D", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=1024,
                   help="Forecast samples per series. LOCF is degenerate, so all "
                        "samples are equal; this just controls histogram shape.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--dataset_seed", type=int, default=123)
    # Histogram + DGP knobs (must match those in the eval metric for fair compare)
    p.add_argument("--edges_min", type=float, default=-10.0)
    p.add_argument("--edges_max", type=float, default=10.0)
    p.add_argument("--n_bins", type=int, default=300)
    p.add_argument("--step_size", type=float, default=4.0)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--a", type=float, default=0.9)
    p.add_argument("--center_coef", type=float, default=0.9)
    args = p.parse_args()

    if args.T_split >= args.T or args.T_split < 1:
        raise SystemExit(f"--T_split must be in [1, T-1]; got {args.T_split} for T={args.T}")

    ds = SyntheticDataset(
        mode="bimodal", split=args.split, N_per_split=args.n_per_split,
        T=args.T, D=args.D, dataset_seed=args.dataset_seed,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    edges = np.linspace(args.edges_min, args.edges_max, args.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    jsds: list[float] = []
    x_prevs: list[float] = []
    sample_buf: list[np.ndarray] = []
    model_mass_buf: list[np.ndarray] = []
    truth_mass_buf: list[np.ndarray] = []

    for batch in dl:
        obs = batch["observed_data"]  # (B, D, T)
        B = obs.shape[0]
        x_prev = obs[:, 0, args.T_split - 1].numpy()  # (B,)
        # LOCF: every "sample" is just x_prev repeated
        xhat = np.repeat(x_prev[:, None], args.num_samples, axis=1).astype(np.float32)

        for b in range(B):
            ctr = xhat[b] - args.center_coef * x_prev[b]
            mm = _hist_mass(ctr, edges)
            tm = _bimodal_truth_mass(centers, float(x_prev[b]),
                                     a=args.a, step_size=args.step_size,
                                     sigma=args.sigma, center_coef=args.center_coef)
            jsds.append(_jsd_discrete(mm, tm))
            x_prevs.append(float(x_prev[b]))
            sample_buf.append(xhat[b].astype(np.float32, copy=True))
            model_mass_buf.append(mm.astype(np.float32, copy=True))
            truth_mass_buf.append(tm.astype(np.float32, copy=True))

    jsd_arr = np.asarray(jsds, dtype=np.float64)
    n = int(jsd_arr.size)
    if n == 0:
        raise SystemExit("No samples produced — check --n_per_split / --batch_size.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        sample_idx=np.arange(n, dtype=np.int64),
        x_prev=np.asarray(x_prevs, dtype=np.float32),
        xhat_samples=np.stack(sample_buf, axis=0),
        edges=edges.astype(np.float32),
        centers=centers.astype(np.float32),
        model_mass=np.stack(model_mass_buf, axis=0),
        truth_mass=np.stack(truth_mass_buf, axis=0),
        center_coef=np.float32(args.center_coef),
        step_size=np.float32(args.step_size),
        sigma=np.float32(args.sigma),
        a=np.float32(args.a),
    )

    summary = {
        "baseline": "locf",
        "n": n,
        "bimodal_jsd_mean": float(jsd_arr.mean()),
        "bimodal_jsd_std": float(jsd_arr.std()),
        "bimodal_jsd_sem": float(jsd_arr.std() / math.sqrt(n)),
        "bimodal_jsd_median": float(np.median(jsd_arr)),
    }
    if args.summary_json:
        import json
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"[LOCF] n={n} JSD mean={summary['bimodal_jsd_mean']:.6f} "
          f"sem={summary['bimodal_jsd_sem']:.6f} median={summary['bimodal_jsd_median']:.6f}")
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
