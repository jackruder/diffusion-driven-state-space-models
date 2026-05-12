"""KDD Cup 2018 PM2.5 + Diffusion transition."""

from ddssm.builders import DiffTransition, Schedule, Unet

from experiments._bases import kdd_base
from experiments._make import run
from experiments._registry import experiment_store


exp = kdd_base(
    DiffTransition(unet=Unet(), schedule=Schedule()),
    steps=8000, batch_size=64,
)


experiment_store(exp, name="kdd_diffusion")


if __name__ == "__main__":
    run(exp, run_dir="runs/kdd_diffusion")
