"""Variance probe: nonlinear-bimodal-lift synthetic + Diffusion-V2."""

from experiments._bases import variance_probe_base
from experiments._make import run
from experiments._registry import experiment_store


exp = variance_probe_base(
    name="variance_probe_nonlinear_bimodal_lift",
    mode="nonlinear-bimodal-lift",
    data_dim=4, latent_dim=8,
)


experiment_store(exp, name="variance_probe_nonlinear_bimodal_lift")


if __name__ == "__main__":
    run(exp, run_dir="runs/variance_probe_nonlinear_bimodal_lift")
