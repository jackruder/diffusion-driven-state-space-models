"""Bimodal synthetic + Gaussian transition (D=1, j=1, S=4)."""

from ddssm.builders import GaussTransition

from experiments._bases import bimodal_base
from experiments._make import run
from experiments._registry import experiment_store


exp = bimodal_base(GaussTransition())


experiment_store(exp, name="bimodal_gauss")


if __name__ == "__main__":
    run(exp, run_dir="runs/bimodal_gauss")
