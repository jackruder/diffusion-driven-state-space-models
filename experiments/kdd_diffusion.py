"""KDD Cup 2018 PM2.5 + Diffusion transition."""

from ddssm.builders import DiffTransition, Schedule, Unet

from experiments._bases import kdd_base
from experiments._make import run


exp = kdd_base(
    DiffTransition(unet=Unet(), schedule=Schedule()),
    steps=8000, batch_size=64,
)


if __name__ == "__main__":
    run(exp, run_dir="runs/kdd_diffusion")
