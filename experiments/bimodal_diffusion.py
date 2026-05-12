"""Bimodal synthetic + Diffusion transition (D=1, j=1, S=4)."""

from ddssm.builders import DiffTransition, Schedule, Unet

from experiments._bases import bimodal_base
from experiments._make import run


exp = bimodal_base(
    DiffTransition(unet=Unet(), schedule=Schedule()),
    steps=2000, checkpoint_every=500, lambda_warmup_steps=400,
)


if __name__ == "__main__":
    run(exp, run_dir="runs/bimodal_diffusion")
