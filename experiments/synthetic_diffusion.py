"""LGSSM smoke test + Diffusion transition (1000 steps)."""

from __future__ import annotations

from ddssm.builders import (
    DiffTransition, Eval, Hparams, Plot, Schedule, Synthetic, Training, Unet, Viz,
)

from experiments._make import make_experiment, run


exp = make_experiment(
    data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
    data=Synthetic(mode="lgssm", T=64, N_per_split=512, batch_size=32),
    hparams=Hparams(
        S=1, batch_size=32, grad_accum_steps=1,
        lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=300,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
    ),
    training=Training(steps=1000, log_every=25, amp=False),
    transition=DiffTransition(unet=Unet(), schedule=Schedule()),
    eval=Eval(metrics=["loss_tail", "recon_mse"], split="val"),
    viz=Viz(
        plots=[
            Plot(name="metrics_csv", save_filename="train_loss.png",
                 kwargs={"keys": ["loss/total"], "log_y": True}),
        ],
        split="val", num_samples=10, T_split=32,
    ),
)


if __name__ == "__main__":
    run(exp, run_dir="runs/synthetic_diffusion")
