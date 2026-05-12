"""Harmonic synthetic + Gaussian transition (D=1, j=1)."""

from ddssm.builders import GaussTransition

from experiments._bases import harmonic_base
from experiments._make import run


exp = harmonic_base(GaussTransition())


if __name__ == "__main__":
    run(exp, run_dir="runs/harmonic_gauss")
