"""Bimodal synthetic + Gaussian transition (D=1, j=1, S=4)."""

from __future__ import annotations

from ddssm.builders import (
    Eval, GaussTransition, Hparams, Plot, Synthetic, Training, Viz,
)

from experiments._make import make_experiment, run


exp = make_experiment(
    data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
    data=Synthetic(mode="bimodal", T=64, N_per_split=1024, batch_size=32),
    hparams=Hparams(
        S=4, batch_size=32, grad_accum_steps=1,
        lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=200,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
    ),
    training=Training(steps=1000, log_every=25, checkpoint_every=200, amp=False),
    transition=GaussTransition(),
    eval=Eval(metrics=["energy_score", "crps_sum"], split="val",
              num_samples=64, T_split=32),
    viz=Viz(
        plots=[
            Plot(name="forecast_1d", save_filename="forecast.png",
                 kwargs={"n_show": 4}),
            Plot(name="metrics_csv", save_filename="train_loss.png",
                 kwargs={"keys": ["loss/total"], "log_y": True}),
        ],
        split="val", num_samples=64, T_split=32,
    ),
)


if __name__ == "__main__":
    run(exp, run_dir="runs/bimodal_gauss")
