"""Bimodal synthetic + Diffusion transition (D=1, j=1, S=4)."""

from ddssm.builders import Eval, Hparams, Plot, Training, Viz

from experiments._make import experiment, run
from experiments._models import SmallDiff
from experiments._datasets import Bimodal
from experiments._registry import experiment_store


exp = experiment(
    data=Bimodal,
    model=SmallDiff,
    hparams=Hparams(
        S=4, batch_size=32, grad_accum_steps=1,
        lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=400,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
    ),
    training=Training(steps=2000, log_every=25, checkpoint_every=500, amp=False),
    eval=Eval(metrics=["energy_score", "crps_sum"], split="val", num_samples=64, T_split=32),
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
experiment_store(exp, name="bimodal_diffusion")


if __name__ == "__main__":
    run(exp, run_dir="runs/bimodal_diffusion")
