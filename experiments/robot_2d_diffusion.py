"""Robot navigation 2D + Diffusion transition (D=2, j=2)."""

from ddssm.builders import Eval, Hparams, Plot, Training, Viz

from experiments._make import experiment, run
from experiments._models import Robot2DDiff
from experiments._datasets import Robot2D
from experiments._registry import experiment_store


exp = experiment(
    data=Robot2D,
    model=Robot2DDiff,
    hparams=Hparams(
        S=1, batch_size=32, grad_accum_steps=1,
        lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=800,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
    ),
    training=Training(steps=4000, log_every=50, checkpoint_every=500, amp=False),
    eval=Eval(metrics=["energy_score", "crps_sum"], split="val", num_samples=32, T_split=32),
    viz=Viz(
        plots=[
            Plot(name="forecast_2d_spatial", save_filename="forecast_2d.png",
                 kwargs={"n_show": 4}),
            Plot(name="metrics_csv", save_filename="train_loss.png",
                 kwargs={"keys": ["loss/total"], "log_y": True}),
        ],
        split="val", num_samples=32, T_split=32,
    ),
)
experiment_store(exp, name="robot_2d_diffusion")


if __name__ == "__main__":
    run(exp, run_dir="runs/robot_2d_diffusion")
