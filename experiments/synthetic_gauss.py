"""LGSSM smoke test + Gaussian transition (500 steps)."""

from ddssm.builders import Eval, Hparams, Plot, Training, Viz

from experiments._make import experiment, run
from experiments._models import SmallGauss
from experiments._datasets import LGSSM
from experiments._registry import experiment_store


exp = experiment(
    data=LGSSM,
    model=SmallGauss,
    hparams=Hparams(
        S=1, batch_size=32, grad_accum_steps=1,
        lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=200,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
    ),
    training=Training(steps=500, log_every=25, amp=False),
    eval=Eval(metrics=["loss_tail", "recon_mse"], split="val"),
    viz=Viz(
        plots=[Plot(name="metrics_csv", save_filename="train_loss.png",
                    kwargs={"keys": ["loss/total"], "log_y": True})],
        split="val", num_samples=10, T_split=32,
    ),
)
experiment_store(exp, name="synthetic_gauss")


if __name__ == "__main__":
    run(exp, run_dir="runs/synthetic_gauss")
