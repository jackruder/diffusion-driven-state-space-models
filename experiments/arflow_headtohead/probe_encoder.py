"""Probe an arflow encoder checkpoint: is it stuck in the noise-floor collapse?

The arflow latent is z_t = cumsum_s(g_s + σ_s·η_s). If σ stays ≈1 (its zero-init
value), the accumulated noise (std ≈ √T) swamps the data-dependent signal g, so
recon can never learn — the per-step SNR = Var[g] / E[σ²] stays ≪ 1. This prints
σ, |g|, that SNR, and the latent-std growth over time to confirm/refute it.

Run::

    .venv/bin/python experiments/arflow_headtohead/probe_encoder.py \
        arflow_h2h__arflow_j1 runs/sweep_arflow_j1_lw/1/checkpoints/ckpt_stage_2_latest.pth
"""

import torch
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from hydra import compose, initialize_config_dir
from hydra_zen import instantiate

from ddssm.experiment.registry import register_experiments
from ddssm.training.checkpoint import load_into_model
from ddssm.model.encoder import _shift_right_time
from ddssm.nn.net_utils import time_embedding


def main() -> None:
    exp_name, ckpt = sys.argv[1], sys.argv[2]
    ckpt = ckpt if os.path.isabs(ckpt) else os.path.join(_REPO, ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    register_experiments()
    with initialize_config_dir(
        config_dir=os.path.join(_REPO, "src", "ddssm", "conf"), version_base="1.3"
    ):
        cfg = compose(config_name="config", overrides=[f"experiment={exp_name}"])
    exp = instantiate(cfg.experiment)
    # ``exp.model`` is now a ModelAdapter; the raw ``DDSSM_base`` module lives
    # under ``.module`` and is what ``load_into_model`` + the encoder probe act on.
    model = exp.model.module.to(device)
    load_into_model(model, ckpt, device=device, load_ema=True, strict=False)
    model.eval()

    loader = exp.data.loader("test")
    transform = exp.data.batch_transform
    batch = next(iter(loader))
    if transform is not None:
        batch = transform(batch, device)
    x = batch["observed_data"]
    tp = batch["timepoints"]
    mask = batch["observation_mask"]
    te = time_embedding(tp, model.emb_time_dim, device=device)

    with torch.no_grad():
        zs, _logq, stats = model.encoder.sample_paths(
            observed_data=x, time_embed=te, S=1,
            cond_mask=mask if getattr(model.encoder, "use_mask", False) else None,
        )
        mus, logvars = stats["mus"], stats["logvars"]
        sigma = (0.5 * logvars).exp()
        sigma2 = logvars.exp()
        g = mus - _shift_right_time(zs)            # the data-dependent residual

        var_g = g.var(dim=(0, 1)).mean().item()    # signal power (var across batch)
        mean_s2 = sigma2.mean().item()             # noise power
        T = zs.shape[-1]
        z0 = zs[..., 0].std().item()
        zT = zs[..., -1].std().item()

    print(f"exp={exp_name}")
    print(f"  mean sigma            = {sigma.mean().item():.4f}  (zero-init value = 1.0)")
    print(f"  mean sigma^2 (noise)  = {mean_s2:.4f}")
    print(f"  mean |g|              = {g.abs().mean().item():.4f}")
    print(f"  Var[g] (signal)       = {var_g:.6f}")
    print(f"  per-step SNR = Var[g]/E[sigma^2] = {var_g / max(mean_s2, 1e-12):.6f}")
    print(f"  latent std: t=0 {z0:.3f} -> t={T-1} {zT:.3f}  (ratio {zT / max(z0, 1e-9):.2f})")
    print(f"  sqrt(T) noise-accumulation reference = {T ** 0.5:.2f}")


if __name__ == "__main__":
    main()
