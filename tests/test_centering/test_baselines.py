"""Unit tests for :mod:`ddssm.centering.baselines`."""

from __future__ import annotations

import torch
import pytest

from ddssm.model.centering.baselines import (
    MLPBaseline,
    BaseBaseline,
    ZeroBaseline,
    LinearBaseline,
    PersistenceBaseline,
)

B = 4
D = 3
J_VALUES = [1, 2]


def _all_forms(latent_dim: int, j: int) -> list[BaseBaseline]:
    return [
        ZeroBaseline(latent_dim=latent_dim, j=j),
        PersistenceBaseline(latent_dim=latent_dim, j=j),
        LinearBaseline(latent_dim=latent_dim, j=j),
        MLPBaseline(latent_dim=latent_dim, j=j, hidden_dim=8, n_layers=2),
    ]


@pytest.mark.parametrize("j", J_VALUES)
def test_baselines_mean_shape(j: int) -> None:
    """All four forms return mu with shape (B, d)."""
    z_hist = torch.randn(B, D, j)
    for baseline in _all_forms(latent_dim=D, j=j):
        mu = baseline.mean(z_hist)
        assert mu.shape == (B, D), f"{type(baseline).__name__}: mu shape {mu.shape}"


@pytest.mark.parametrize("j", J_VALUES)
def test_baselines_mean_and_logvar_shape(j: int) -> None:
    """All four forms return (mu, logvar) each of shape (B, d)."""
    z_hist = torch.randn(B, D, j)
    for baseline in _all_forms(latent_dim=D, j=j):
        mu, logvar = baseline.mean_and_logvar(z_hist)
        assert mu.shape == (B, D)
        assert logvar.shape == (B, D)


@pytest.mark.parametrize("j", J_VALUES)
def test_zero_baseline_mean_is_zero(j: int) -> None:
    """ZeroBaseline.mean returns exact zeros (no parameters for μ_p)."""
    z_hist = torch.randn(B, D, j)
    baseline = ZeroBaseline(latent_dim=D, j=j)
    mu = baseline.mean(z_hist)
    assert torch.equal(mu, torch.zeros_like(mu))


@pytest.mark.parametrize("j", J_VALUES)
def test_persistence_baseline_mean_is_last_slot(j: int) -> None:
    """PersistenceBaseline.mean returns z_hist[..., -1] (random-walk prior at j=1; persistence at j>1)."""
    z_hist = torch.randn(B, D, j)
    baseline = PersistenceBaseline(latent_dim=D, j=j)
    mu = baseline.mean(z_hist)
    assert torch.equal(mu, z_hist[..., -1])


def test_linear_baseline_is_affine() -> None:
    """LinearBaseline implements μ_p = A · vec(z) + b."""
    baseline = LinearBaseline(latent_dim=D, j=2)
    z1 = torch.randn(1, D, 2)
    z2 = torch.randn(1, D, 2)
    alpha = 0.7
    z_mix = alpha * z1 + (1 - alpha) * z2

    mu1 = baseline.mean(z1)
    mu2 = baseline.mean(z2)
    mu_mix = baseline.mean(z_mix)
    expected = alpha * mu1 + (1 - alpha) * mu2
    assert torch.allclose(mu_mix, expected, atol=1e-6)


def test_mlp_baseline_shares_backbone() -> None:
    """``mean`` and ``mean_and_logvar`` reuse the same MLP body.

    We exercise the backbone once via ``mean`` and once via
    ``mean_and_logvar`` and check the same hidden tensor is produced
    (modulo the two output heads).
    """
    torch.manual_seed(0)
    baseline = MLPBaseline(latent_dim=D, j=2, hidden_dim=16, n_layers=2)
    z = torch.randn(B, D, 2)

    # Hook the backbone to count forward passes.
    calls = {"n": 0}

    def _counter(_module: torch.nn.Module, _inp: tuple, _out: torch.Tensor) -> None:
        calls["n"] += 1

    h = baseline.backbone.register_forward_hook(_counter)
    try:
        mu_only = baseline.mean(z)
        n_mean_calls = calls["n"]
        mu_full, _logvar = baseline.mean_and_logvar(z)
        n_total_calls = calls["n"]
    finally:
        h.remove()

    # Each access does exactly one backbone pass.
    assert n_mean_calls == 1
    assert n_total_calls == 2
    # μ produced by mean() and by mean_and_logvar() agree exactly
    # (same backbone params, same μ head).
    assert torch.allclose(mu_only, mu_full, atol=1e-6)


@pytest.mark.parametrize("j", J_VALUES)
def test_snapshot_is_disjoint_and_frozen(j: int) -> None:
    """``snapshot()`` returns a frozen deep copy."""
    for baseline in _all_forms(latent_dim=D, j=j):
        snapshot = baseline.snapshot()
        # Frozen.
        for p in snapshot.parameters():
            assert not p.requires_grad
        # Parameter disjoint: editing the live baseline does not move
        # the snapshot.
        for p in baseline.parameters():
            p.data.add_(1.0)
        # Re-compare a forward pass on the same input.
        z_hist = torch.randn(2, D, j)
        live_mu = baseline.mean(z_hist)
        snap_mu = snapshot.mean(z_hist)
        # If the snapshot were sharing parameters, these would match;
        # they should differ for parametric baselines.
        if any(p.numel() > 0 for p in baseline.parameters()):
            # MLPBaseline / LinearBaseline / PersistenceBaseline (has
            # sigma_head params) / ZeroBaseline (has sigma_head params)
            # — at least the σ side has parameters everywhere.
            # mean() may not differ on Zero (μ=0 always) or Persistence
            # (μ=z_hist[..., -1] always) — only the σ part has params.
            # So we test on mean_and_logvar's logvar output instead for
            # those forms.
            _, live_lv = baseline.mean_and_logvar(z_hist)
            _, snap_lv = snapshot.mean_and_logvar(z_hist)
            assert not torch.allclose(live_lv, snap_lv)


@pytest.mark.parametrize("j", J_VALUES)
def test_parametric_baseline_mu_head_bias_is_zero_at_init(j: int) -> None:
    """``LinearBaseline`` / ``MLPBaseline`` zero ``mu_head.bias`` at init.

    Matches the ``GaussianHead.mu_head`` convention (xavier-uniform weight,
    zero bias): μ_p(0) = 0 exactly. Under typical-scale z_hist μ_p is a
    small projection that the model can grow during training. (Going
    further and zeroing ``mu_head.weight`` too — i.e. μ_p ≡ 0 for *any*
    z_hist — landed the score-net log-likelihood in a config where dopri5
    is pathologically slow on the integration tests.)
    """
    linear = LinearBaseline(latent_dim=D, j=j)
    mlp = MLPBaseline(latent_dim=D, j=j, hidden_dim=8, n_layers=2)
    for baseline in (linear, mlp):
        assert torch.allclose(
            baseline.mu_head.bias, torch.zeros_like(baseline.mu_head.bias)
        )
    # LinearBaseline composes only mu_head(z_hist_flat); with bias=0 the
    # output at z_hist=0 is exactly zero. MLPBaseline runs z_hist through a
    # backbone with its own (non-zero) Kaiming biases first, so
    # μ_p(z_hist=0) is generically non-zero even though mu_head.bias=0 —
    # the bias-only contract is the meaningful one for the parametric forms.
    z_zero = torch.zeros(B, D, j)
    assert torch.allclose(linear.mean(z_zero), torch.zeros_like(linear.mean(z_zero)))


@pytest.mark.parametrize("j", J_VALUES)
def test_baselines_init_logvar_is_zero(j: int) -> None:
    """At init every baseline emits ``log σ_p² ≈ 0`` (σ_p² = I).

    Guards the contract that ``LogvarHead`` anchors the initial
    log-variance — important so ``r_sigma_p_loss`` starts at zero and
    so stage-1 KL is not warped by random head init.
    """
    z_hist = torch.randn(B, D, j)
    for baseline in _all_forms(latent_dim=D, j=j):
        _, logvar = baseline.mean_and_logvar(z_hist)
        assert torch.allclose(logvar, torch.zeros_like(logvar), atol=1e-5), (
            f"{type(baseline).__name__}: init logvar {logvar.abs().max().item()=}"
        )


@pytest.mark.parametrize("j", J_VALUES)
def test_baselines_reject_wrong_shape(j: int) -> None:
    """Inputs with the wrong shape are rejected."""
    for baseline in _all_forms(latent_dim=D, j=j):
        with pytest.raises(ValueError):
            baseline.mean(torch.randn(B, D + 1, j))
        with pytest.raises(ValueError):
            baseline.mean(torch.randn(B, D))  # wrong rank
        if j > 1:
            with pytest.raises(ValueError):
                baseline.mean(torch.randn(B, D, j - 1))
