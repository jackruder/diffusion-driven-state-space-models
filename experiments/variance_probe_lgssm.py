"""Variance probe: LGSSM synthetic + Diffusion-V2 transition."""

from experiments._bases import variance_probe_base
from experiments._make import run
from experiments._registry import experiment_store


exp = variance_probe_base(name="variance_probe_lgssm", mode="lgssm")


experiment_store(exp, name="variance_probe_lgssm")


if __name__ == "__main__":
    run(exp, run_dir="runs/variance_probe_lgssm")
