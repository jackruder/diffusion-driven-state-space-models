"""Variance probe: bimodal-noisy synthetic + Diffusion-V2 transition."""

from experiments._bases import variance_probe_base
from experiments._make import run


exp = variance_probe_base(
    name="variance_probe_bimodal_noisy", mode="bimodal-noisy",
)


if __name__ == "__main__":
    run(exp, run_dir="runs/variance_probe_bimodal_noisy")
