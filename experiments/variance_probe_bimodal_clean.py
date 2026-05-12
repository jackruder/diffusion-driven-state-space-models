"""Variance probe: bimodal synthetic + Diffusion-V2 transition."""

from __future__ import annotations

from ddssm.builders import (
    DiffV2Transition, Hparams, Objective, Probe, ScheduleV2, Synthetic,
    Training, Unet,
)

from experiments._make import make_experiment, run


exp = make_experiment(
    data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
    checkpoint_dir="${oc.env:PWD,.}/runs/variance_probe/variance_probe_bimodal_clean/checkpoints",
    data=Synthetic(mode="bimodal", T=64, D=1, N_per_split=256, batch_size=32),
    hparams=Hparams(
        S=1, batch_size=32, grad_accum_steps=1,
        lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=50,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
    ),
    training=Training(steps=300, log_every=20, checkpoint_every=100, amp=False),
    transition=DiffV2Transition(unet=Unet(), schedule=ScheduleV2()),
    objective=Objective(metric="loss/total", split="train", tail_frac=0.1),
    variance=Probe(),
)


if __name__ == "__main__":
    run(exp, run_dir="runs/variance_probe_bimodal_clean")
