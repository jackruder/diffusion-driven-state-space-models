"""LGSSM smoke test + Gaussian transition (500 steps)."""

from ddssm.builders import GaussTransition

from experiments._bases import synthetic_base
from experiments._make import run


exp = synthetic_base(GaussTransition())


if __name__ == "__main__":
    run(exp, run_dir="runs/synthetic_gauss")
