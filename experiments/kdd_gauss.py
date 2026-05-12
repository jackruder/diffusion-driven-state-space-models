"""KDD Cup 2018 PM2.5 + Gaussian transition (D=6, j=1, covariates=3)."""

from ddssm.builders import Eval, Hparams, Plot, Training, Viz

from experiments._make import experiment, run
from experiments._models import KDDGauss
from experiments._datasets import KDDData
from experiments._registry import experiment_store


exp = experiment(
    data=KDDData,
    model=KDDGauss,
    hparams=Hparams(
        S=1, batch_size=128, grad_accum_steps=1,
        lambda_schedule="linear", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=500,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
    ),
    training=Training(steps=5000, log_every=50, checkpoint_every=500, amp=True),
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
experiment_store(exp, name="kdd_gauss")


if __name__ == "__main__":
    run(exp, run_dir="runs/kdd_gauss")
