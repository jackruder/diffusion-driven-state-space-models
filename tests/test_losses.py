"""Tests for the loss-object abstraction (ADR-0004)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Reuse the canonical VHP fixture from the existing stage-1 test file.
sys.path.insert(0, str(Path(__file__).parent))
from test_dssd_stage1 import _make_batch, _make_vhp_model

from ddssm.model.losses import Loss, FullELBO, LossComponents


def test_full_elbo_assembles_recon_plus_lambda_rate_plus_anchors() -> None:
    """FullELBO sums recon + λ·(init_kl + trans_kl) + anchor regularizers.

    The centering regularizers are NOT multiplied by ``rate_lambda``: they
    carry their own ``lambda_sigma_p`` / ``lambda_mu_p`` weights and must
    stay live even when ``λ=0`` (recon-only warmup), otherwise σ_p is
    free to collapse precisely when it's most fragile.
    """
    components = LossComponents(
        recon=torch.tensor(1.0),
        init_kl_phith=torch.tensor(2.0),
        init_kl_psi=torch.zeros(()),
        trans_kl_phith=torch.tensor(3.0),
        trans_kl_psi=torch.zeros(()),
        r_sigma_p=torch.tensor(4.0),
        r_mu_p=torch.tensor(5.0),
    )
    loss = FullELBO(
        rate_lambda=lambda step: 0.5,
        lambda_sigma_p=0.1,
        lambda_mu_p=0.01,
    )
    expected = 1.0 + 0.5 * (2.0 + 3.0) + 0.1 * 4.0 + 0.01 * 5.0
    assert torch.allclose(loss(components, step=0), torch.tensor(expected))


def test_full_elbo_keeps_anchors_live_at_lambda_zero() -> None:
    """At λ=0 (recon-only warmup) the σ_p anchor must still contribute.

    Regression for the warmup-anchor bug: previously the regularizers
    were inside the ``λ * (...)`` bracket, so they vanished when ``λ=0``
    — exactly when σ_p collapse is most likely. Both anchors must now
    be additive on top of ``recon`` regardless of the rate ramp.
    """
    components = LossComponents(
        recon=torch.tensor(1.0),
        init_kl_phith=torch.tensor(2.0),
        init_kl_psi=torch.zeros(()),
        trans_kl_phith=torch.tensor(3.0),
        trans_kl_psi=torch.zeros(()),
        r_sigma_p=torch.tensor(7.0),
        r_mu_p=torch.tensor(0.0),
    )
    # λ=0 (warmup), σ_p anchor weight non-zero.
    loss_sigma = FullELBO(
        rate_lambda=lambda step: 0.0,
        lambda_sigma_p=1.0,
        lambda_mu_p=0.0,
    )
    out = loss_sigma(components, step=0)
    # recon (1.0) + λ·rate (0.0) + λ_σp · r_sigma_p (1·7).
    assert torch.allclose(out, torch.tensor(8.0))
    assert out.item() != components.recon.item(), "σ_p anchor must survive λ=0 warmup"

    # Same for μ_p anchor with r_sigma_p zero.
    components_mu = LossComponents(
        recon=torch.tensor(1.0),
        init_kl_phith=torch.tensor(2.0),
        init_kl_psi=torch.zeros(()),
        trans_kl_phith=torch.tensor(3.0),
        trans_kl_psi=torch.zeros(()),
        r_sigma_p=torch.tensor(0.0),
        r_mu_p=torch.tensor(11.0),
    )
    loss_mu = FullELBO(
        rate_lambda=lambda step: 0.0,
        lambda_sigma_p=0.0,
        lambda_mu_p=1.0,
    )
    out_mu = loss_mu(components_mu, step=0)
    assert torch.allclose(out_mu, torch.tensor(12.0))
    assert out_mu.item() != components_mu.recon.item(), (
        "μ_p anchor must survive λ=0 warmup"
    )


def test_forward_returns_loss_components() -> None:
    """DDSSM_base.forward() returns (LossComponents, metrics, stats).

    Each of the five component fields is a finite scalar tensor; the
    existing metrics dict survives unchanged for logging continuity.
    """
    torch.manual_seed(0)
    model = _make_vhp_model(lambda_sigma_p=0.01)
    batch = _make_batch(B=2, T=5)
    torch.manual_seed(42)
    result = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    assert len(result) == 3, "forward() must return a 3-tuple post-ADR-0004"
    components, metrics, _stats = result
    assert isinstance(components, LossComponents)
    for field in ("recon", "init_kl", "trans_kl", "r_sigma_p", "r_mu_p"):
        t = getattr(components, field)
        assert t.ndim == 0, f"{field} must be a scalar tensor"
        assert torch.isfinite(t), f"{field} must be finite"
    assert "loss/distortion/rec" in metrics  # logging continuity


def test_full_elbo_reproduces_pre_refactor_loss() -> None:
    """ADR-0004 lockdown: FullELBO numerically reproduces the
    pre-refactor trainer's `loss = distortion + λ_rate * rate`.

    Captured pre-refactor against the VHP stage-1 fixture with
    seed=0 (model init), seed=42 (forward), B=2, T=5,
    lambda_sigma_p=0.01.

    Note: the warmup-anchor fix (regularizers pulled out of the
    `λ * (...)` bracket) is numerically equivalent to the old formula
    at ``λ=1.0``. The value was recaptured after commit a00f7a3 (#2:
    encoder logvars are now clamped in-place in ``enc_stats`` so the
    downstream init/KL terms see the clamped values), which legitimately
    shifts the stage-1 ELBO by ~0.25%; then recaptured again when the
    baseline σ_p head moved from raw ``nn.Linear`` to :class:`LogvarHead`
    (init logvar anchored at 0 → σ_p² = I), which shifts the stage-1
    ELBO by ~1.1% on this fixture; then recaptured again when
    ``LinearBaseline`` / ``MLPBaseline``'s ``mu_head`` adopted the
    ``GaussianHead.mu_head`` convention (xavier-uniform weight + zero
    bias) so μ_p(0)=0 — shifting the loss by ~0.7% on the MLP-baseline
    fixture.
    """
    LAMBDA_SIGMA_P = 0.01
    LAMBDA_MU_P = 0.0
    EXPECTED_LOSS = (
        20.757253646850586  # recaptured: custom TransformerBlock (RMSNorm + SwiGLU)
    )

    torch.manual_seed(0)
    model = _make_vhp_model(lambda_sigma_p=LAMBDA_SIGMA_P)
    # GaussianEncoder now defaults to mu_mode="additive" (the persistence frame).
    # This ADR-0004 guard locks the FullELBO *formula* against a fixed pre-refactor
    # probe, which used the free encoder; pin it back so the reference stays valid.
    model.encoder.mu_mode = "free"
    batch = _make_batch(B=2, T=5)
    torch.manual_seed(42)
    components, _metrics, _stats = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    loss_obj = FullELBO(
        rate_lambda=lambda step: 1.0,
        lambda_sigma_p=LAMBDA_SIGMA_P,
        lambda_mu_p=LAMBDA_MU_P,
    )
    loss = loss_obj(components, step=0)
    assert abs(loss.item() - EXPECTED_LOSS) < 1e-5, (
        f"Got {loss.item()}, expected {EXPECTED_LOSS}"
    )


def test_loss_selection_actually_selects() -> None:
    """A non-FullELBO Loss subclass produces a different scalar on identical
    components — proves the abstraction's selection point isn't a no-op.
    """

    class ReconOnly(Loss):
        """Toy loss object — only the recon term contributes."""

        def __call__(self, components: LossComponents, step: int) -> torch.Tensor:
            return components.recon

    components = LossComponents(
        recon=torch.tensor(1.0),
        init_kl_phith=torch.tensor(2.0),
        init_kl_psi=torch.zeros(()),
        trans_kl_phith=torch.tensor(3.0),
        trans_kl_psi=torch.zeros(()),
        r_sigma_p=torch.tensor(0.0),
        r_mu_p=torch.tensor(0.0),
    )
    full = FullELBO(rate_lambda=lambda step: 1.0)
    recon_only = ReconOnly()
    assert full(components, 0).item() != recon_only(components, 0).item()
    assert recon_only(components, 0).item() == 1.0
