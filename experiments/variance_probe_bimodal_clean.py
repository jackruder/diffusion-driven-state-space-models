"""Variance probe: bimodal synthetic + Diffusion-V2 transition."""

from ddssm.builders import Eval, Hparams, Plot, Training, Viz, Objective, Probe

from experiments._make import experiment, run
from experiments._models import ProbeSmall
from experiments._datasets import ProbeBimodal
from experiments._registry import experiment_store


exp = experiment(
    data=ProbeBimodal,
    model=ProbeSmall,
    hparams=Hparams(
        S=1, batch_size=32, grad_accum_steps=1,
        lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=50,
        enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
    ),
    training=Training(steps=300, log_every=20, checkpoint_every=100, amp=False),
    eval=None,
    viz=None,
    objective=Objective(metric="loss/total", split="train", tail_frac=0.1),
    variance=Probe(),
)
experiment_store(exp, name="variance_probe_bimodal_clean")


if __name__ == "__main__":
    run(exp, run_dir="runs/variance_probe_bimodal_clean")
