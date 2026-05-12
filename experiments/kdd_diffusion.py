"""KDD Cup 2018 PM2.5 + Diffusion transition."""

from __future__ import annotations

from ddssm.builders import (
    DiffTransition, Eval, Hparams, KDD, Plot, Schedule, Training, Unet, Viz,
)

from experiments._make import make_experiment, run


exp = make_experiment(
    data_dim=6, latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
    data=KDD(batch_size=64, eval_step_size=24),
    hparams=Hparams(
        S=1, batch_size=64, grad_accum_steps=1,
        lambda_schedule="linear", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=500,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
    ),
    training=Training(steps=8000, log_every=50, checkpoint_every=500, amp=True),
    transition=DiffTransition(unet=Unet(), schedule=Schedule()),
    eval=Eval(metrics=["mae", "crps_sum"], split="test", num_samples=32),
    viz=Viz(
        plots=[
            Plot(name="forecast_1d", save_filename="forecast.png",
                 kwargs={"n_show": 4}),
            Plot(name="metrics_csv", save_filename="train_loss.png",
                 kwargs={"keys": ["loss/total"], "log_y": True}),
        ],
        split="test", num_samples=32,
    ),
)


if __name__ == "__main__":
    run(exp, run_dir="runs/kdd_diffusion")
