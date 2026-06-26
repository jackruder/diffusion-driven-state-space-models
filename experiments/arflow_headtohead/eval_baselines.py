"""Forecast-metric sanity check: a trained head-to-head cell vs trivial baselines.

Answers "is the cell underfitting?" by scoring the best trial's checkpoint
against two parameter-free baselines on the held-out test split, using the
SAME proper scores the family reports (``crps_sum`` / energy score), plus a
Gaussian predictive NLL on the channel-sum series (the 8-channel sum is
near-Gaussian by CLT, so a moment-matched NLL is fair and multimodal-robust).

Baselines (estimated from each batch's OWN history window — no future peeking):
  * locf  : last-observation-carried-forward with random-walk noise
            (y_{L1+h} = x_{L1-1} + cumsum_h eps, eps ~ N(0, sigma_diff^2)).
  * marg  : i.i.d. Gaussian marginal N(mu_d, sigma_d^2) per channel.

The model and both baselines are scored through the identical metric code, so
the numbers are directly comparable. A trained model that does not clearly beat
both baselines on CRPS-sum is underfitting / not learning the dynamics.

Run::

    .venv/bin/python experiments/arflow_headtohead/eval_baselines.py \
        --experiment arflow_h2h__gaussian_j1 \
        --checkpoint runs/sweep_gaussian_j1/13/checkpoints/ckpt_stage_2_latest.pth \
        --num-samples 100 --t-split 24
"""

import torch  # preload first so numpy's C-extensions resolve on NixOS
import os
import sys
import math
import argparse

# Make ``import experiments`` resolve when run as a bare script (mirrors registry.py).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import numpy as np
from hydra import compose, initialize_config_dir
from hydra_zen import instantiate

from ddssm.experiment.registry import register_experiments
from ddssm.training.checkpoint import load_into_model
from ddssm.eval.eval_metrics import mae_metrics, rmse_metrics, crps_sum_metrics


def gauss_sum_nll(pred_samples: torch.Tensor, y_future: torch.Tensor) -> torch.Tensor:
    """Gaussian predictive NLL on the channel-sum series, per (B, L2)."""
    ps = pred_samples.sum(dim=2)            # (B, S, L2)
    ys = y_future.sum(dim=1)                # (B, L2)
    mu = ps.mean(dim=1)                     # (B, L2)
    var = ps.var(dim=1, unbiased=True).clamp_min(1e-8)
    return 0.5 * ((ys - mu) ** 2 / var + torch.log(2 * math.pi * var))


def energy_score(pred_samples: torch.Tensor, y_future: torch.Tensor) -> torch.Tensor:
    """Energy score per batch element (mirrors eval.metrics.eval_energy_score)."""
    B, S, D, L2 = pred_samples.shape
    s_flat = pred_samples.reshape(B, S, -1)
    y_flat = y_future.reshape(B, -1).unsqueeze(1)
    term1 = torch.norm(s_flat - y_flat, dim=-1).mean(dim=1)        # (B,)
    diff = s_flat.unsqueeze(2) - s_flat.unsqueeze(1)
    pair = torch.norm(diff, dim=-1)
    if S > 1:
        term2 = pair.sum(dim=(1, 2)) / (S * (S - 1))
    else:
        term2 = torch.zeros_like(term1)
    return term1 - 0.5 * term2


def _baseline_samples(x_hist: torch.Tensor, L2: int, S: int, kind: str):
    """Return (pred_samples (B,S,D,L2), pred_mean (B,D,L2)) for a baseline."""
    B, D, L1 = x_hist.shape
    dev = x_hist.device
    if kind == "locf":
        x_last = x_hist[..., -1]                              # (B, D)
        diffs = x_hist[..., 1:] - x_hist[..., :-1]            # (B, D, L1-1)
        sigma_d = diffs.std(dim=(0, 2)).clamp_min(1e-6)       # (D,)
        eps = torch.randn(B, S, D, L2, device=dev) * sigma_d.view(1, 1, D, 1)
        samples = x_last.view(B, 1, D, 1) + eps.cumsum(dim=-1)
        mean = x_last.unsqueeze(-1).expand(B, D, L2)
        return samples, mean
    if kind == "marg":
        mu_d = x_hist.mean(dim=(0, 2))                        # (D,)
        sd_d = x_hist.std(dim=(0, 2)).clamp_min(1e-6)         # (D,)
        samples = mu_d.view(1, 1, D, 1) + torch.randn(
            B, S, D, L2, device=dev
        ) * sd_d.view(1, 1, D, 1)
        mean = mu_d.view(1, D, 1).expand(B, D, L2)
        return samples, mean
    raise ValueError(kind)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--t-split", type=int, default=None, help="L1; default 3/4 of T")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    register_experiments()
    conf_dir = os.path.join(_REPO, "src", "ddssm", "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        cfg = compose(config_name="config",
                      overrides=[f"experiment={args.experiment}"])
    exp = instantiate(cfg.experiment)
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(
        _REPO, args.checkpoint)
    model = exp.model.to(device)
    # strict=False: the checkpoint carries training-only ``baseline_anchor.*``
    # (centering-handoff anchor for R_sigma_p/R_mu_p), absent from a fresh
    # eval model and unused by forecast.
    load_into_model(model, ckpt, device=device, load_ema=True, strict=False)
    model.eval()

    loader = exp.data.loader("test")
    transform = exp.data.batch_transform
    S = int(args.num_samples)

    # First batch tells us T; pick the split.
    first = next(iter(loader))
    T = first["observed_data"].shape[-1]
    L1 = args.t_split if args.t_split is not None else max(1, (T * 3) // 4)
    L2 = T - L1
    print(f"experiment={args.experiment}  ckpt={os.path.basename(ckpt)}")
    print(f"T={T}  L1(history)={L1}  L2(horizon)={L2}  S={S}  device={device}\n")

    forecasters = ["model", "locf", "marg"]
    acc = {f: {"crps": [], "energy": [], "nll": [], "mae": [], "rmse": []}
           for f in forecasters}

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            if transform is not None:
                batch = transform(batch, device)
            else:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}

            x_hist = batch["observed_data"][..., :L1]
            x_mask = batch["observation_mask"][..., :L1]
            past_t = batch["timepoints"][:, :L1]
            fut_t = batch["timepoints"][:, L1:]
            y_fut = batch["observed_data"][..., L1:]
            cov = batch.get("covariates")
            pcov = cov[..., :L1] if cov is not None else None
            fcov = cov[..., L1:] if cov is not None else None
            scov = batch.get("static_covariates")

            for f in forecasters:
                if f == "model":
                    out = model.forecast(
                        x_hist=x_hist, x_mask=x_mask, past_time=past_t,
                        future_time=fut_t, past_covariates=pcov,
                        future_covariates=fcov, static_covariates=scov,
                        num_samples=S,
                    )
                    ps, pm = out["pred_samples"], out["pred_mean"]
                else:
                    ps, pm = _baseline_samples(x_hist, L2, S, f)
                acc[f]["crps"].append(float(crps_sum_metrics(ps, y_fut)[0]))
                acc[f]["energy"].append(float(energy_score(ps, y_fut).mean()))
                acc[f]["nll"].append(float(gauss_sum_nll(ps, y_fut).mean()))
                acc[f]["mae"].append(float(mae_metrics(pm, y_fut)[0]))
                acc[f]["rmse"].append(float(rmse_metrics(pm, y_fut)[0]))
            print(f"  batch {bi} done", flush=True)

    hdr = f"{'forecaster':>12} {'CRPS_sum':>10} {'energy':>10} {'NLL_sum':>10} {'MAE':>10} {'RMSE':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for f in forecasters:
        m = {k: float(np.mean(v)) for k, v in acc[f].items()}
        print(f"{f:>12} {m['crps']:>10.4f} {m['energy']:>10.4f} "
              f"{m['nll']:>10.4f} {m['mae']:>10.4f} {m['rmse']:>10.4f}")
    print("\n(lower is better for every column; CRPS_sum is ND-normalized)")


if __name__ == "__main__":
    main()
