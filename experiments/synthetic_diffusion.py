"""LGSSM smoke test + Diffusion transition (1000 steps)."""

from ddssm.builders import DiffTransition, Schedule, Unet

from experiments._bases import synthetic_base
from experiments._make import run
from experiments._registry import experiment_store


exp = synthetic_base(
    DiffTransition(unet=Unet(), schedule=Schedule()),
    steps=1000, lambda_warmup_steps=300,
)


experiment_store(exp, name="synthetic_diffusion")


if __name__ == "__main__":
    run(exp, run_dir="runs/synthetic_diffusion")
