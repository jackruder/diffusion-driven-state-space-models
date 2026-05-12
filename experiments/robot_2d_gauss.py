"""Robot navigation 2D + Gaussian transition (D=2, j=2)."""

from ddssm.builders import GaussTransition

from experiments._bases import robot_2d_base
from experiments._make import run
from experiments._registry import experiment_store


exp = robot_2d_base(GaussTransition())


experiment_store(exp, name="robot_2d_gauss")


if __name__ == "__main__":
    run(exp, run_dir="runs/robot_2d_gauss")
