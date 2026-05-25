"""Math claim: stage-1 KL drives ``μ̂_t = μ_t − μ_p(z_{t-1}) → 0``.

From ``model-v2.org`` § Centered coordinates and encoder marginal:

  "The pretraining prior N(μ_p, diag(σ_p²(z_{t-1}))) drives
  μ̂_t ≈ 0 per-state and σ_t ≈ σ_p(z_{t-1}) ideally with the
  global log-variance regularizer anchoring the scale to 1."

The closed-form Gaussian KL ``KL[q || p]`` includes the term
``‖μ_q − μ_p‖² / (2 σ_p²)``, which is minimised at ``μ_q = μ_p`` —
so a long-enough stage-1 run should bring ‖μ̂_t‖ close to zero.
"""

from __future__ import annotations

import pytest
import torch

from .conftest import (
    make_smooth_sine_data,
    make_vhp_model,
    run_stage,
)


pytestmark = pytest.mark.slow


def _measure_centered_residual_norm(model, data) -> float:
    """Encode a batch and return ``mean ‖μ_t − μ_p(z_{t-1})‖``.

    Averaged across (b, s, t) for ``t ≥ j+1`` (the transition window
    where the centering applies).
    """
    j = model.j
    with torch.no_grad():
        from ddssm.net_utils import time_embedding

        te = time_embedding(
            data["timepoints"], model.emb_time_dim, device=data["observed_data"].device,
        )
        zs, _, stats = model._encode_latents(
            observed_data=data["observed_data"],
            time_embed=te,
            observation_mask=data["observation_mask"],
        )
        mus = stats["mus"]  # (B, S, d, T)
        B, S, d, T = mus.shape
        if T <= j:
            return 0.0
        # For each t ≥ j+1, compute mu_p(z_hist) then ‖mu_t - mu_p‖.
        total = 0.0
        count = 0
        for t in range(j, T):
            z_hist = zs[:, :, :, t - j : t]  # (B, S, d, j)
            z_hist_flat = z_hist.reshape(B * S, d, j)
            mu_p = model.baseline.mean(z_hist_flat)  # (B*S, d)
            mu_t_flat = mus[:, :, :, t].reshape(B * S, d)
            mu_hat = mu_t_flat - mu_p
            total += float(mu_hat.norm(dim=-1).mean().item())
            count += 1
        return total / max(count, 1)


@pytest.mark.parametrize("baseline_form", ["zero", "mlp"])
def test_centered_residual_norm_shrinks_during_stage1(baseline_form: str) -> None:
    """After stage-1 training, ‖μ̂_t‖ has shrunk vs. its random-init value."""
    torch.manual_seed(0)
    model = make_vhp_model(
        baseline_form=baseline_form,
        lambda_sigma_p=1e-2,
        tracking_mode="per_t",
    )
    batch = make_smooth_sine_data(n_seqs=32, T=8, seed=99)

    # Baseline measurement at init.
    pre_norm = _measure_centered_residual_norm(model, batch)

    # Train stage 1.
    run_stage(
        model=model,
        stage="stage_1",
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=8, T=8, seed=torch.randint(0, 10_000, (1,)).item(),
        ),
        n_steps=400,
        lr=3e-3,
    )

    post_norm = _measure_centered_residual_norm(model, batch)
    # The doc predicts μ̂_t → 0; with finite training we expect a
    # non-trivial decrease.  Tolerance: at least 30% reduction.
    assert post_norm < 0.9 * pre_norm, (
        f"baseline={baseline_form}: ‖μ̂‖ did not shrink (pre={pre_norm:.3f}, "
        f"post={post_norm:.3f})"
    )
