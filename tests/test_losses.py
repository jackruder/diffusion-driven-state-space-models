"""Tests for the loss-object abstraction (ADR-0004)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Reuse the canonical VHP fixture from the existing stage-1 test file.
sys.path.insert(0, str(Path(__file__).parent))
from test_dssd_stage1 import _make_batch, _make_vhp_model  # noqa: E402

from ddssm.losses import FullELBO, Loss, LossComponents  # noqa: E402


def test_full_elbo_reproduces_trainer_formula() -> None:
    """FullELBO weights LossComponents the way the trainer used to.

    Old trainer: ``loss = distortion + λ_rate(step) * rate``, where
    ``rate = init_kl + trans_kl + r_sigma_p_weighted + r_mu_p_weighted``
    and the regularizer weights were applied inside the model. Under
    ADR-0004 the components are unweighted; FullELBO holds both
    ``lambda_sigma_p`` and ``lambda_mu_p`` and applies them inside the
    same rate sum so the resulting scalar is numerically identical.
    """
    components = LossComponents(
        recon=torch.tensor(1.0),
        init_kl=torch.tensor(2.0),
        trans_kl=torch.tensor(3.0),
        r_sigma_p=torch.tensor(4.0),
        r_mu_p=torch.tensor(5.0),
    )
    loss = FullELBO(
        rate_lambda=lambda step: 0.5,
        lambda_sigma_p=0.1,
        lambda_mu_p=0.01,
    )
    expected = 1.0 + 0.5 * (2.0 + 3.0 + 0.1 * 4.0 + 0.01 * 5.0)
    assert torch.allclose(loss(components, step=0), torch.tensor(expected))


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
    """
    LAMBDA_SIGMA_P = 0.01
    LAMBDA_MU_P = 0.0
    EXPECTED_LOSS = 22.7022514343  # captured pre-refactor

    torch.manual_seed(0)
    model = _make_vhp_model(lambda_sigma_p=LAMBDA_SIGMA_P)
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

        def __call__(
            self, components: LossComponents, step: int
        ) -> torch.Tensor:
            return components.recon

    components = LossComponents(
        recon=torch.tensor(1.0),
        init_kl=torch.tensor(2.0),
        trans_kl=torch.tensor(3.0),
        r_sigma_p=torch.tensor(0.0),
        r_mu_p=torch.tensor(0.0),
    )
    full = FullELBO(rate_lambda=lambda step: 1.0)
    recon_only = ReconOnly()
    assert full(components, 0).item() != recon_only(components, 0).item()
    assert recon_only(components, 0).item() == 1.0
