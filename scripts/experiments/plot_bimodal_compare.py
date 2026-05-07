"""Multi-run JSD/density comparison plot for bimodal experiments.

Reads NPZ files produced by the ``bimodal_jsd`` eval metric (with
``npz_path=...``) or by ``bimodal_locf.py``, and produces:

* ``--out_jsd``: errorbar plot of mean JSD ± 1 SEM, one row per run.
* ``--out_density``: average forecast density across examples vs the
  analytic-truth density, one curve per run.

Usage::

    python scripts/experiments/plot_bimodal_compare.py \\
        --runs gauss=runs/bimodal/gauss/bimodal_jsd.npz \\
               diff=runs/bimodal/diff/bimodal_jsd.npz \\
               locf=runs/locf/bimodal_jsd.npz \\
        --out_jsd  runs/bimodal/compare/jsd.png \\
        --out_density runs/bimodal/compare/density.png \\
        --center_coefs 0.9 1.0

If ``--center_coefs`` lists multiple values, one pair of plots is produced
per coefficient with a ``_aX.png`` suffix appended to each output path.
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from ddssm.eval.metrics import _bimodal_truth_mass, _hist_mass, _jsd_discrete


def _suffix_path(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _recompute_for_coef(npz, center_coef: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Recompute model+truth mean masses and JSD summary at a different center_coef.

    The NPZ stores raw forecast samples and x_prev so we can re-bin under a
    different centring without re-running the model.
    """
    xhat_samples = npz["xhat_samples"]
    x_prev = npz["x_prev"]
    edges = npz["edges"]
    centers = npz["centers"]
    a = float(npz["a"]) if "a" in npz.files else 0.9
    step_size = float(npz["step_size"]) if "step_size" in npz.files else 4.0
    sigma = float(npz["sigma"]) if "sigma" in npz.files else 0.2

    model_masses, truth_masses, jsds = [], [], []
    for i in range(len(x_prev)):
        ctr = xhat_samples[i] - center_coef * x_prev[i]
        mm = _hist_mass(ctr, edges)
        tm = _bimodal_truth_mass(centers, float(x_prev[i]),
                                 a=a, step_size=step_size, sigma=sigma,
                                 center_coef=center_coef)
        model_masses.append(mm)
        truth_masses.append(tm)
        jsds.append(_jsd_discrete(mm, tm))
    n = len(jsds)
    return (
        np.mean(model_masses, axis=0),
        np.mean(truth_masses, axis=0),
        float(np.mean(jsds)),
        float(np.std(jsds) / math.sqrt(n)) if n > 0 else 0.0,
    )


def _plot_jsd(out_path: str, labels: list[str], means: list[float], sems: list[float]) -> None:
    plt.figure(figsize=(7.2, 0.6 + 0.5 * len(labels)))
    y = np.arange(len(labels), dtype=float)
    plt.errorbar(means, y, xerr=sems, fmt="o", capsize=5, color="black")
    plt.yticks(y, labels, fontsize=14)
    plt.xticks(fontsize=11)
    plt.xlabel("Jensen–Shannon divergence (mean ± 1 SEM)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_density(out_path: str, centers: np.ndarray, edges: np.ndarray,
                  run_masses: dict[str, np.ndarray], truth_mass: np.ndarray) -> None:
    bw = float(edges[1] - edges[0])
    plt.figure(figsize=(8.2, 5.0))
    for label, mm in run_masses.items():
        plt.plot(centers, mm / bw, linestyle="-", label=f"{label} (avg forecast)")
    plt.plot(centers, truth_mass / bw, color="tab:green", linewidth=2,
             label="DGP truth (analytic avg)")
    plt.xlabel("Δ = xₜ − a·xₜ₋₁")
    plt.ylabel("Average density across examples")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    _ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _parse_runs(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--runs entries must be label=path, got {item!r}")
        label, path = item.split("=", 1)
        out.append((label, path))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more label=path.npz pairs.")
    p.add_argument("--out_jsd", required=True)
    p.add_argument("--out_density", required=True)
    p.add_argument("--center_coefs", type=float, nargs="+", default=[0.9])
    args = p.parse_args()

    runs = _parse_runs(args.runs)
    npzs = {label: np.load(path) for label, path in runs}

    centers = next(iter(npzs.values()))["centers"]
    edges = next(iter(npzs.values()))["edges"]

    multi = len(args.center_coefs) > 1
    for coef in args.center_coefs:
        suffix = f"_a{coef:.1f}".replace(".", "p") if multi else ""
        run_means: dict[str, np.ndarray] = {}
        run_truth_means: list[np.ndarray] = []
        labels, jsd_means, jsd_sems = [], [], []
        for label, npz in npzs.items():
            mm, tm, jsd_mean, jsd_sem = _recompute_for_coef(npz, coef)
            run_means[label] = mm
            run_truth_means.append(tm)
            labels.append(label)
            jsd_means.append(jsd_mean)
            jsd_sems.append(jsd_sem)

        truth_mean = np.mean(run_truth_means, axis=0)
        out_jsd = _suffix_path(args.out_jsd, suffix)
        out_density = _suffix_path(args.out_density, suffix)
        _plot_jsd(out_jsd, labels, jsd_means, jsd_sems)
        _plot_density(out_density, centers, edges, run_means, truth_mean)
        print(f"[Saved a={coef}] {out_jsd}")
        print(f"[Saved a={coef}] {out_density}")


if __name__ == "__main__":
    main()
