"""Unit tests for :mod:`ddssm.centering.sigma_data`."""

from __future__ import annotations

import torch
import pytest

from ddssm.model.centering.sigma_data import SigmaDataBuffer

T_MAX = 5
D = 3


def test_init_fills_buffer_with_init_value() -> None:
    """Buffer starts at ``init_value`` for every slot."""
    buf = SigmaDataBuffer(T_max=T_MAX, init_value=2.5)
    assert torch.allclose(buf.sigma_data2, torch.full((T_MAX,), 2.5))
    assert torch.equal(buf.ema_step, torch.zeros(T_MAX, dtype=torch.long))
    assert not buf.frozen


def test_init_rejects_bad_inputs() -> None:
    """Constructor validates ``T_max``, ``tracking_mode``, and ``ema_decay``."""
    with pytest.raises(ValueError):
        SigmaDataBuffer(T_max=0)
    with pytest.raises(ValueError):
        SigmaDataBuffer(T_max=T_MAX, tracking_mode="weird")
    with pytest.raises(ValueError):
        SigmaDataBuffer(T_max=T_MAX, ema_decay=1.0)
    with pytest.raises(ValueError):
        SigmaDataBuffer(T_max=T_MAX, ema_decay=-0.1)


def test_read_translates_1_based_to_0_based() -> None:
    """``read(t)`` returns the slot for 1-based ``t``."""
    buf = SigmaDataBuffer(T_max=T_MAX, init_value=1.0)
    # Hand-fill the buffer.
    buf.sigma_data2 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    for t in range(1, T_MAX + 1):
        assert float(buf.read(t).item()) == float(buf.sigma_data2[t - 1].item())


def test_read_out_of_range_raises() -> None:
    """Reading an out-of-range t raises ``IndexError``."""
    buf = SigmaDataBuffer(T_max=T_MAX)
    with pytest.raises(IndexError):
        buf.read(0)
    with pytest.raises(IndexError):
        buf.read(T_MAX + 1)


def test_per_t_update_matches_analytic_estimator() -> None:
    """The per-t update reproduces ``(1/D)(E‖σ²‖ + tr Var[μ̂])``."""
    buf = SigmaDataBuffer(
        T_max=T_MAX, tracking_mode="per_t", ema_decay=0.0, init_value=0.0,
    )
    per_t = 6
    t_idx = torch.tensor([2, 4])  # two timesteps
    mu = torch.randn(per_t * 2, D)
    s2 = torch.rand(per_t * 2, D)

    buf.update(t_idx=t_idx, mu_hat_batch=mu, sigma_t2_batch=s2)

    # With ema_decay=0 the buffer becomes the estimator.
    expected = []
    for k, _t in enumerate(t_idx.tolist()):
        mu_block = mu[k * per_t : (k + 1) * per_t]
        s2_block = s2[k * per_t : (k + 1) * per_t]
        avg_post_var = s2_block.mean(dim=0).sum()
        mu_var = mu_block.var(dim=0, unbiased=True).sum()
        expected.append(((avg_post_var + mu_var) / D).item())
    for k, t in enumerate(t_idx.tolist()):
        assert pytest.approx(float(buf.read(t).item()), rel=1e-5) == expected[k]
    # Unvisited slots untouched.
    for t in (1, 3, 5):
        assert float(buf.read(t).item()) == 0.0


def test_steady_state_is_batch_size_invariant() -> None:
    """EMA steady-state target is unchanged when batch size grows.

    Regression for the ``unbiased=False`` bug: the maximum-likelihood
    variance estimator's expectation scales as ``(per_t − 1)/per_t``, so
    the steady-state EMA value moved with batch size (e.g. ~6% low at
    per_t=16, converging from below as per_t→∞). Using the
    Bessel-corrected estimator removes that dependence: with a true
    distribution held fixed and ``ema_decay=0`` (so the buffer equals
    the per-update estimate), the *expected* per-update estimate is
    identical for any ``per_t > 1``.
    """
    torch.manual_seed(0)
    # Synthesise per-sample (μ̂, σ²) from a known marginal distribution:
    # μ̂ ~ N(0, mu_scale² I), σ² fixed. The true ``σ_data² = (1/D)(‖σ²‖
    # + d · mu_scale²)``. Average the per-update estimator across many
    # independent draws to suppress sampling noise; with ``unbiased=True``
    # the average should converge to the same value for any per_t.
    mu_scale = 0.7
    sigma2_true = torch.tensor([0.4, 0.5, 0.6])
    true_target = (sigma2_true.sum().item() + D * mu_scale ** 2) / D

    n_draws = 4000
    means = {}
    for per_t in (4, 16, 256):
        buf = SigmaDataBuffer(
            T_max=T_MAX, tracking_mode="per_t", ema_decay=0.0, init_value=0.0,
        )
        acc = 0.0
        for _ in range(n_draws):
            mu = torch.randn(per_t, D) * mu_scale
            s2 = sigma2_true.unsqueeze(0).expand(per_t, D).clone()
            buf.update(
                t_idx=torch.tensor([2]), mu_hat_batch=mu, sigma_t2_batch=s2,
            )
            acc += float(buf.read(2).item())
        means[per_t] = acc / n_draws

    # Each per_t setting should match the true target within MC noise
    # (~1/sqrt(n_draws) on the variance estimate).
    for per_t, mean in means.items():
        assert abs(mean - true_target) < 5e-3, (
            f"per_t={per_t} estimator avg {mean:.4f} vs true {true_target:.4f}"
        )


def test_global_ema_updates_all_slots_uniformly() -> None:
    """Under ``"global_ema"`` every slot is updated to the same value."""
    buf = SigmaDataBuffer(
        T_max=T_MAX, tracking_mode="global_ema", ema_decay=0.0, init_value=0.0,
    )
    per_t = 5
    t_idx = torch.tensor([1, 3])
    mu = torch.randn(per_t * 2, D)
    s2 = torch.rand(per_t * 2, D)

    buf.update(t_idx=t_idx, mu_hat_batch=mu, sigma_t2_batch=s2)
    # All slots equal the mean of the per-t estimates.
    assert float((buf.sigma_data2 - buf.sigma_data2[0]).abs().sum().item()) == 0.0


def test_fixed_mode_freezes_after_reset_schedule() -> None:
    """Under ``"fixed"``, updates are no-ops after ``reset_schedule``."""
    buf = SigmaDataBuffer(
        T_max=T_MAX, tracking_mode="fixed", ema_decay=0.0, init_value=0.0,
    )
    per_t = 4
    t_idx = torch.tensor([2])
    mu = torch.randn(per_t, D)
    s2 = torch.rand(per_t, D)

    # Before reset: updates DO take effect (stage-1 accumulation).
    buf.update(t_idx=t_idx, mu_hat_batch=mu, sigma_t2_batch=s2)
    pre_freeze = buf.sigma_data2.clone()
    assert float((pre_freeze[1] - 0.0).abs().item()) > 0.0

    buf.reset_schedule()
    assert buf.frozen
    assert torch.equal(buf.ema_step, torch.zeros(T_MAX, dtype=torch.long))
    # Buffer values persist.
    assert torch.equal(buf.sigma_data2, pre_freeze)

    # Post reset: updates are no-ops.
    buf.update(
        t_idx=t_idx,
        mu_hat_batch=torch.full((per_t, D), 999.0),
        sigma_t2_batch=torch.full((per_t, D), 999.0),
    )
    assert torch.equal(buf.sigma_data2, pre_freeze)


def test_reset_schedule_preserves_values() -> None:
    """``reset_schedule`` zeros ``ema_step`` but never touches ``sigma_data2``."""
    buf = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t", init_value=0.0)
    buf.sigma_data2 = torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5])
    buf.ema_step = torch.tensor([7, 8, 9, 10, 11], dtype=torch.long)

    buf.reset_schedule()
    assert torch.equal(buf.sigma_data2, torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert torch.equal(buf.ema_step, torch.zeros(T_MAX, dtype=torch.long))


def test_update_no_grad_on_buffer() -> None:
    """Buffer updates don't allocate gradient on the live tensors."""
    buf = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    per_t = 4
    t_idx = torch.tensor([1])
    mu = torch.randn(per_t, D, requires_grad=True)
    s2 = torch.rand(per_t, D, requires_grad=True)

    buf.update(t_idx=t_idx, mu_hat_batch=mu, sigma_t2_batch=s2)
    assert buf.sigma_data2.grad_fn is None  # detached


def test_update_rejects_mismatched_inputs() -> None:
    """``update`` rejects shape mismatches."""
    buf = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    with pytest.raises(ValueError):
        buf.update(
            t_idx=torch.tensor([1, 2]),
            mu_hat_batch=torch.zeros(6, D),
            sigma_t2_batch=torch.zeros(6, D + 1),
        )
    with pytest.raises(ValueError):
        # 5 rows for 2 timesteps -> not divisible.
        buf.update(
            t_idx=torch.tensor([1, 2]),
            mu_hat_batch=torch.zeros(5, D),
            sigma_t2_batch=torch.zeros(5, D),
        )
