"""LGSSM smoke test + Gaussian transition (500 steps)."""

from ddssm.builders import GaussTransition

from experiments._bases import synthetic_base
from experiments._make import run
from experiments._registry import experiment_store


exp = synthetic_base(GaussTransition())


experiment_store(exp, name="synthetic_gauss")


if __name__ == "__main__":
    run(exp, run_dir="runs/synthetic_gauss")
