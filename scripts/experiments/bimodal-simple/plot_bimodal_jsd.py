import argparse
import math
import os
import json

import matplotlib.pyplot as plt
import numpy as np


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


def load_npz(path: str):
    return np.load(path)


def compute_metrics_for_coef(npz_data, center_coef: float):
    xhat_samples = npz_data["xhat_samples"]
    x_prev = npz_data["x_prev"]
    edges = npz_data["edges"]
    centers = npz_data["centers"]

    N = len(x_prev)
    model_masses = []
    truth_masses = []
    jsds = []

    for i in range(N):
        ctr_vals = xhat_samples[i] - center_coef * x_prev[i]
        p_mass = histogram_mass(ctr_vals, edges)
        q_mass = analytic_truth_mass_per_example(centers, x_prev[i], a=center_coef)

        model_masses.append(p_mass)
        truth_masses.append(q_mass)
        jsds.append(float(jsd_discrete(p_mass, q_mass)))

    mean_model_mass = np.mean(model_masses, axis=0)
    mean_truth_mass = np.mean(truth_masses, axis=0)
    mean_jsd = float(np.mean(jsds))
    sem_jsd = float(np.std(jsds) / math.sqrt(N)) if N > 0 else 0.0

    return mean_model_mass, mean_truth_mass, mean_jsd, sem_jsd


def suffix_path(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def save_jsd_plot(
    out_path: str,
    g_jsd_mean: float,
    g_jsd_sem: float,
    d_jsd_mean: float,
    d_jsd_sem: float,
    n_jsd_mean: float,
    n_jsd_sem: float,
    center_coef: float,
):
    plt.figure(figsize=(7.2, 4.0))
    y_labels = ["DKF", "DDDSSM", "LOCF"]
    y_pos = np.array([0, 1, 2], dtype=float)
    means = np.array([g_jsd_mean, d_jsd_mean, n_jsd_mean], dtype=float)
    sems = np.array([g_jsd_sem, d_jsd_sem, n_jsd_sem], dtype=float)

    plt.errorbar(means, y_pos, xerr=sems, fmt="o", capsize=5, color="black")
    plt.xticks(fontsize=12)
    plt.yticks(y_pos, y_labels, fontsize=18)
    # plt.xlabel("Jensen–Shannon Divergence", fontsize=18)

    coef_str = f"{center_coef:.3g}"
    coef_disp = "" if coef_str == "1" else coef_str

    # plt.title(
    #     f"Bimodal one-step forecast quality: Δ = xₜ − {coef_disp}xₜ₋₁ (mean ± 1 SEM)"
    # )
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_density_plot(
    out_path: str,
    centers: np.ndarray,
    edges: np.ndarray,
    g_mean_mass: np.ndarray,
    d_mean_mass: np.ndarray,
    truth_mean_mass: np.ndarray,
    center_coef: float,
):
    bw = edges[1] - edges[0]
    g_den = g_mean_mass / bw
    d_den = d_mean_mass / bw
    t_den = truth_mean_mass / bw

    plt.figure(figsize=(8.2, 5.0))
    plt.plot(
        centers, g_den, color="tab:blue", linestyle="-", label="Gaussian (avg forecast)"
    )
    plt.plot(
        centers,
        d_den,
        color="tab:orange",
        linestyle="-",
        label="Diffusion (avg forecast)",
    )
    plt.plot(
        centers, t_den, color="tab:green", linewidth=2, label="DGP truth (analytic avg)"
    )

    coef_str = f"{center_coef:.3g}"
    coef_disp = "" if coef_str == "1" else coef_str

    # plt.xlabel(f"Δ = xₜ − {coef_disp}xₜ₋₁", fontsize=18)
    plt.ylabel("Average density across examples", fontsize=18)
    # plt.title("Bimodal one-step distribution match")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    # Accept JSONs for arg-compatibility but we won't read them
    p.add_argument("--gauss_summary_json", type=str, required=False, default="")
    p.add_argument("--diff_summary_json", type=str, required=False, default="")
    p.add_argument("--naive_summary_json", type=str, required=False, default="")

    p.add_argument("--gauss_npz", type=str, required=True)
    p.add_argument("--diff_npz", type=str, required=True)
    p.add_argument("--naive_npz", type=str, required=True)

    p.add_argument("--out_jsd", type=str, required=True)
    p.add_argument("--out_density", type=str, required=True)

    p.add_argument("--center_coefs", type=float, nargs="+", default=[0.9, 1.0])
    args = p.parse_args()

    g_npz = load_npz(args.gauss_npz)
    d_npz = load_npz(args.diff_npz)
    n_npz = load_npz(args.naive_npz)

    centers = g_npz["centers"]
    edges = g_npz["edges"]

    for coef in args.center_coefs:
        suffix = f"_a{coef:.1f}".replace(".", "p")

        g_mm, g_tm, g_jsd, g_sem = compute_metrics_for_coef(g_npz, coef)
        d_mm, d_tm, d_jsd, d_sem = compute_metrics_for_coef(d_npz, coef)
        n_mm, n_tm, n_jsd, n_sem = compute_metrics_for_coef(n_npz, coef)

        truth_mean_mass = (g_tm + d_tm + n_tm) / 3.0

        out_jsd_path = suffix_path(args.out_jsd, suffix)
        save_jsd_plot(
            out_jsd_path,
            g_jsd,
            g_sem,
            d_jsd,
            d_sem,
            n_jsd,
            n_sem,
            center_coef=coef,
        )

        out_density_path = suffix_path(args.out_density, suffix)
        save_density_plot(
            out_density_path,
            centers,
            edges,
            g_mm,
            d_mm,
            truth_mean_mass,
            center_coef=coef,
        )

        print(f"[Saved a={coef}] {out_jsd_path}")
        print(f"[Saved a={coef}] {out_density_path}")


if __name__ == "__main__":
    main()
