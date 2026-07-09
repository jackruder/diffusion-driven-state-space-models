"""Tests for the loss-object abstraction (ADR-0004).

Post-refactor: r_sigma_p / r_mu_p regularizers were removed, so
LossComponents no longer carries r_ fields and FullELBO no longer takes
lambda_sigma_p / lambda_mu_p. Only the split-loss + rate-lambda tests
remain.
"""

from __future__ import annotations

import torch

from ddssm.model.losses import Loss, FullELBO, LossComponents, SplitLoss


def _components(**overrides) -> LossComponents:
    values = dict(
        recon=torch.tensor(1.0),
        init_kl_phith=torch.tensor(2.0),
        init_kl_psi=torch.tensor(3.0),
        trans_kl_phith=torch.tensor(4.0),
        trans_kl_psi=torch.tensor(5.0),
    )
    values.update(overrides)
    return LossComponents(**values)


def test_full_elbo_assembles_recon_plus_lambda_rate() -> None:
    """FullELBO sums recon + λ·(init_kl + trans_kl) (no r_ regularizers)."""
    comps = _components()
    loss = FullELBO(rate_lambda=lambda step: 0.5)
    expected = 1.0 + 0.5 * (2.0 + 4.0)
    assert torch.allclose(loss(comps, step=0), torch.tensor(expected))


def test_full_elbo_at_lambda_zero_is_recon_only() -> None:
    """At λ=0 (recon-only warmup) FullELBO reduces to recon."""
    comps = _components()
    loss = FullELBO(rate_lambda=lambda step: 0.0)
    out = loss(comps, step=0)
    assert torch.allclose(out, comps.recon)


def test_full_elbo_split_returns_splitloss() -> None:
    """``use_split_loss=True`` returns a ``SplitLoss`` pair."""
    comps = _components()
    single = FullELBO(rate_lambda=lambda _s: 0.5)
    out = single(comps, 1)
    assert isinstance(out, torch.Tensor)
    expected_phith = 1.0 + 0.5 * (2.0 + 4.0)
    assert out.item() == expected_phith

    split = FullELBO(rate_lambda=lambda _s: 0.5, use_split_loss=True)
    out_split = split(comps, 1)
    assert isinstance(out_split, SplitLoss)
    assert out_split.phith.item() == expected_phith
    # ψ side is init_kl_psi + trans_kl_psi, NOT gated by rate_lambda.
    assert out_split.psi.item() == 3.0 + 5.0


def test_psi_side_ignores_lambda() -> None:
    """The ψ side is NOT gated by ``rate_lambda``; φθ at λ=0 is KL-free."""
    comps = _components()
    lam0 = FullELBO(rate_lambda=lambda _s: 0.0, use_split_loss=True)(comps, 1)
    lam1 = FullELBO(rate_lambda=lambda _s: 1.0, use_split_loss=True)(comps, 1)
    assert torch.equal(lam0.psi, lam1.psi)
    assert lam0.phith.item() == comps.recon.item()
    assert lam1.phith.item() == 1.0 + 2.0 + 4.0


def test_loss_selection_actually_selects() -> None:
    """A non-FullELBO Loss subclass produces a different scalar on identical
    components — proves the abstraction's selection point isn't a no-op.
    """

    class ReconOnly(Loss):
        """Toy loss object — only the recon term contributes."""

        def __call__(self, components: LossComponents, step: int) -> torch.Tensor:
            return components.recon

    comps = _components()
    full = FullELBO(rate_lambda=lambda step: 1.0)
    recon_only = ReconOnly()
    assert full(comps, 0).item() != recon_only(comps, 0).item()
    assert recon_only(comps, 0).item() == 1.0
