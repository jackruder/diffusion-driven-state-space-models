"""Sanity-check the MV dataset mixes enough modes within T=32.

The ``nonlinear-bimodal-lift-mv`` latent has 2^d = 16 attractors
(d=4, per-dim independent Rademacher signs). The grilling decision
held ``T_max = 32`` constant — but with 32 timesteps we need to see
*evidence* that trajectories visit multiple attractors per sequence,
not get stuck near one mode. If they don't, the ``per_t`` tracking
mode has no signal to distinguish it from the ``fixed`` mode.

This script:

1. Generates ``N`` trajectories from the MV dataset (default N=64).
2. Per trajectory, assigns each t to its nearest attractor in
   {-1, +1}^4 (16 modes total) using the sign of ``z[:, t]``.
3. Counts distinct attractors visited per trajectory.
4. Builds a per-t state-distribution heatmap (which attractor is
   dominant at each t, averaged across trajectories).
5. Writes a 2-panel PNG: (a) histogram of "distinct attractors visited
   per trajectory" — peak near 1 ⇒ T is too short; peak near 16 ⇒
   good mixing; (b) the per-t × attractor heatmap.

Eyeball the plot before committing to T=32 for the ablation.

Run::

    python -m experiments.init_centering.verify_mv_mixing \\
        --n-trajectories 128 --out runs/mv_mixing.png
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np

from ddssm.data.synthetic import (
    NLBL_MV_OBS_D,
    NLBL_MV_LATENT_D,
    SyntheticDataset,
)


def _sign_attractor_indices(z: np.ndarray) -> np.ndarray:
    """Convert sign(z) ∈ {-1,+1}^(N, d, T) to attractor-index in [0, 2^d)."""
    signs = (z > 0).astype(np.int64)  # (N, d, T) ∈ {0, 1}
    d = signs.shape[1]
    powers = (1 << np.arange(d, dtype=np.int64))[None, :, None]  # (1, d, 1)
    return (signs * powers).sum(axis=1)  # (N, T) ∈ [0, 2^d)


def main(argv: list[str] | None = None) -> int:
    """Sample MV trajectories, measure attractor mixing, and write the plot.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv`` when ``None``.

    Returns:
        Process exit code (``0`` on success).
    """
    p = argparse.ArgumentParser(
        prog="python -m experiments.init_centering.verify_mv_mixing",
    )
    p.add_argument(
        "--n-trajectories", type=int, default=64,
        help="Number of MV trajectories to sample (default 64).",
    )
    p.add_argument(
        "--t-max", type=int, default=32,
        help="Sequence length (default 32, matches the ablation).",
    )
    p.add_argument(
        "--out", default="runs/mv_mixing.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Dataset generation seed (controls split + matrices).",
    )
    args = p.parse_args(argv)

    n_attractors = 1 << NLBL_MV_LATENT_D  # 16

    ds = SyntheticDataset(
        mode="nonlinear-bimodal-lift-mv",
        split="train",
        N_per_split=max(args.n_trajectories, 16),
        T=args.t_max,
        D=NLBL_MV_OBS_D,
        dataset_seed=args.seed,
        expose_gt_latents=True,
    )
    assert ds.gt_latents is not None, "GT latents must be exposed for this check"
    z = ds.gt_latents.numpy()[: args.n_trajectories]  # (N, d, T)

    attractor_ids = _sign_attractor_indices(z)  # (N, T)
    distinct_per_traj = np.array(
        [len(set(attractor_ids[n].tolist())) for n in range(args.n_trajectories)]
    )
    per_t_per_attractor = np.zeros((args.t_max, n_attractors), dtype=np.int64)
    for n in range(args.n_trajectories):
        for t in range(args.t_max):
            per_t_per_attractor[t, attractor_ids[n, t]] += 1
    per_t_per_attractor = per_t_per_attractor.astype(np.float64) / args.n_trajectories

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_hist, ax_heat) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )

    # Panel (a): distinct-attractors histogram.
    bins = np.arange(0.5, n_attractors + 1.5)
    ax_hist.hist(distinct_per_traj, bins=bins, color="#1f77b4", alpha=0.85)
    ax_hist.set_xlabel("distinct attractors visited per trajectory")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(
        f"Mixing: median={int(np.median(distinct_per_traj))} of {n_attractors} attractors"
    )
    ax_hist.set_xticks(range(1, n_attractors + 1))
    ax_hist.grid(True, axis="y", linestyle=":", linewidth=0.5)

    # Panel (b): per-t × attractor heatmap.
    im = ax_heat.imshow(
        per_t_per_attractor.T, aspect="auto", origin="lower",
        cmap="viridis", vmin=0.0, vmax=per_t_per_attractor.max(),
    )
    ax_heat.set_xlabel("latent timestep t")
    ax_heat.set_ylabel("attractor index (bit-pattern of sign(z))")
    ax_heat.set_title(
        f"Per-t attractor distribution (avg across {args.n_trajectories} trajectories)"
    )
    fig.colorbar(im, ax=ax_heat, label="fraction of trajectories")

    fig.suptitle(
        "MV mixing check — nonlinear-bimodal-lift-mv at T="
        f"{args.t_max}",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"Wrote mixing-check plot to {args.out}")
    print(f"  trajectories sampled: {args.n_trajectories}")
    print(f"  distinct attractors visited per traj: "
          f"min={distinct_per_traj.min()} "
          f"median={int(np.median(distinct_per_traj))} "
          f"max={distinct_per_traj.max()}")
    print(f"  out of {n_attractors} possible attractors.")
    print(
        "Healthy mixing: median ≥ ~6 (a third of the modes visited per "
        "trajectory). Pathological: median = 1 (trajectories stuck in "
        "one attractor; T=32 too short for the chosen δ/σ_z)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
