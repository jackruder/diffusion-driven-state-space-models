"""Math claim: parametric baselines (Linear, MLP) actually learn during stage 1.

For Zero / Persistence baselines, μ_p has no trainable parameters of its
own beyond the σ_p head.  For Linear / MLP, μ_p IS parametric, and
stage-1 training should move its parameters off initialisation
toward whatever μ_p ≈ E[μ_t | z_{t-1}] gives best ELBO.

These tests are weaker than "recover the true A, b" — that's
impossible because the encoder is free to pick any latent
representation — but they verify the gradient path *exists* and
drives the baseline somewhere non-trivial.
"""

from __future__ import annotations

import torch
import pytest

from .conftest import run_stage, make_vhp_model, make_smooth_sine_data

pytestmark = pytest.mark.slow


def _baseline_param_norm(model) -> float:
    params = [p.detach().reshape(-1) for p in model.baseline.parameters()]
    if not params:
        return 0.0
    return float(torch.cat(params).norm().item())


@pytest.mark.parametrize("baseline_form", ["linear", "mlp"])
def test_parametric_baseline_param_norm_moves_in_stage1(baseline_form: str) -> None:
    """Parametric baseline parameters move during stage 1."""
    torch.manual_seed(0)
    model = make_vhp_model(
        baseline_form=baseline_form,
        lambda_sigma_p=1e-2,
    )

    pre_params = [p.detach().clone() for p in model.baseline.parameters()]
    pre_norm = _baseline_param_norm(model)

    run_stage(
        model=model,
        stage="stage_1",
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=8,
            T=8,
            seed=torch.randint(0, 10_000, (1,)).item(),
        ),
        n_steps=80,
        lr=3e-3,
    )
    post_params = [p.detach().clone() for p in model.baseline.parameters()]

    # Some parameter must have changed.
    diffs = [
        (post - pre).abs().max().item() for pre, post in zip(pre_params, post_params)
    ]
    assert any(d > 1e-4 for d in diffs), (
        f"baseline={baseline_form}: no parameter moved (max abs diff per tensor: {diffs})"
    )

    # And the L2 norm of the parameter set has shifted (sanity).
    post_norm = _baseline_param_norm(model)
    assert abs(post_norm - pre_norm) > 1e-5, (
        f"baseline={baseline_form}: param L2 norm unchanged (pre={pre_norm:.4f})"
    )


@pytest.mark.parametrize("baseline_form", ["zero", "persistence"])
def test_parameter_free_baseline_mean_params_dont_exist(baseline_form: str) -> None:
    """Zero / Persistence have no μ_p parameters (only σ_p head)."""
    model = make_vhp_model(baseline_form=baseline_form)
    # The σ_p head still has parameters, but the μ_p head must not.
    # Check by inspecting the baseline's named children: Zero and
    # Persistence should expose only ``sigma_head`` parameters.
    mu_p_param_count = 0
    for name, p in model.baseline.named_parameters():
        if "sigma" not in name.lower():
            mu_p_param_count += p.numel()
    assert mu_p_param_count == 0, (
        f"{baseline_form}: expected μ_p to be parameter-free; got "
        f"{mu_p_param_count} non-σ parameters"
    )
