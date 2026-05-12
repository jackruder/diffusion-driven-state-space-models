"""KDD Cup 2018 PM2.5 + Gaussian transition (D=6, j=1, covariates=3)."""

from ddssm.builders import GaussTransition

from experiments._bases import kdd_base
from experiments._make import run


exp = kdd_base(GaussTransition())


if __name__ == "__main__":
    run(exp, run_dir="runs/kdd_gauss")
