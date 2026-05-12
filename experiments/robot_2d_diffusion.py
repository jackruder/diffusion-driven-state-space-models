"""Robot navigation 2D + Diffusion transition (D=2, j=2)."""

from ddssm.builders import DiffTransition, Schedule, Unet

from experiments._bases import robot_2d_base
from experiments._make import run


exp = robot_2d_base(
    DiffTransition(unet=Unet(), schedule=Schedule()),
    steps=4000, lambda_warmup_steps=800,
)


if __name__ == "__main__":
    run(exp, run_dir="runs/robot_2d_diffusion")
