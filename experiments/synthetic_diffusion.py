"""LGSSM smoke test + Diffusion transition (1000 steps)."""

from ddssm.builders import DiffTransition, Schedule, Unet

from experiments._bases import synthetic_base
from experiments._make import run


exp = synthetic_base(
    DiffTransition(unet=Unet(), schedule=Schedule()),
    steps=1000, lambda_warmup_steps=300,
)


if __name__ == "__main__":
    run(exp, run_dir="runs/synthetic_diffusion")
