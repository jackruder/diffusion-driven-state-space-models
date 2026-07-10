"""Unit tests for :mod:`ddssm.centering.sigma_data`."""

from __future__ import annotations

import torch
import pytest

from ddssm.model.centering.sigma_data import SigmaDataBuffer
from tests.fixtures.golden_values import (
    M2_ESTIMATOR_PER_T,
    M2_SEED,
    M2_N_T,
    M2_PER_T,
    M2_D,
    make_m2_inputs,
)

T_MAX = 5
D = 3


def test_init_fills_buffer_with_init_value() -> None:
    """Buffer starts at ``init_value`` for every slot."""
    buf = SigmaDataBuffer(T_max=T_MAX, init_value=2.5)
    assert torch.allclose(buf.sigma_data2, torch.full((T_MAX,), 2.5))
    assert torch.equal(buf.ema_step, torch.zeros(T_MAX, dtype=torch.long))
    assert torch.equal(buf.n_updates, torch.zeros(T_MAX, dtype=torch.long))
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
        T_max=T_MAX,
        tracking_mode="per_t",
        ema_decay=0.0,
        init_value=0.0,
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
    true_target = (sigma2_true.sum().item() + D * mu_scale**2) / D

    n_draws = 4000
    means = {}
    for per_t in (4, 16, 256):
        buf = SigmaDataBuffer(
            T_max=T_MAX,
            tracking_mode="per_t",
            ema_decay=0.0,
            init_value=0.0,
        )
        acc = 0.0
        for _ in range(n_draws):
            mu = torch.randn(per_t, D) * mu_scale
            s2 = sigma2_true.unsqueeze(0).expand(per_t, D).clone()
            buf.update(
                t_idx=torch.tensor([2]),
                mu_hat_batch=mu,
                sigma_t2_batch=s2,
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
        T_max=T_MAX,
        tracking_mode="global_ema",
        ema_decay=0.0,
        init_value=0.0,
    )
    per_t = 5
    t_idx = torch.tensor([1, 3])
    mu = torch.randn(per_t * 2, D)
    s2 = torch.rand(per_t * 2, D)

    buf.update(t_idx=t_idx, mu_hat_batch=mu, sigma_t2_batch=s2)
    # All slots equal the mean of the per-t estimates.
    assert float((buf.sigma_data2 - buf.sigma_data2[0]).abs().sum().item()) == 0.0


def test_fixed_mode_is_frozen_from_construction() -> None:
    """Under ``"fixed"``, ``update`` is a permanent no-op from construction."""
    buf = SigmaDataBuffer(
        T_max=T_MAX,
        tracking_mode="fixed",
        ema_decay=0.0,
        init_value=0.0,
    )
    assert buf.frozen
    per_t = 4
    t_idx = torch.tensor([2])
    mu = torch.full((per_t, D), 999.0)
    s2 = torch.full((per_t, D), 999.0)
    before = buf.sigma_data2.clone()
    with torch.enable_grad():
        buf.update(t_idx=t_idx, mu_hat_batch=mu, sigma_t2_batch=s2)
    assert torch.equal(buf.sigma_data2, before)


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


def test_update_is_noop_under_no_grad() -> None:
    """REGRESSION: σ_data is a TRAINING-only running statistic.

    Eval / inference forwards run under ``torch.no_grad`` while the diffusion /
    baseline transitions update σ_data *inside* the forward. An unguarded update
    let the eval pass drift σ_data toward the eval-data residual and inflate the
    eval ELBO's transition-KL term (obj1) ~2-4x. ``update`` must be a no-op when
    autograd is disabled, and still apply when it is enabled (training).
    """
    buf = SigmaDataBuffer(
        T_max=T_MAX, tracking_mode="per_t", ema_decay=0.9, init_value=1.0
    )
    t_idx = torch.tensor([2, 4])
    mu = torch.randn(12, D)
    s2 = torch.rand(12, D)
    before = buf.sigma_data2.clone()
    step_before = buf.ema_step.clone()

    with torch.no_grad():
        buf.update(t_idx=t_idx, mu_hat_batch=mu, sigma_t2_batch=s2)
    assert torch.equal(buf.sigma_data2, before), "σ_data mutated under no_grad (eval)"
    assert torch.equal(buf.ema_step, step_before)

    # Positive control: with autograd enabled (training) the SAME update applies.
    with torch.enable_grad():
        buf.update(t_idx=t_idx, mu_hat_batch=mu, sigma_t2_batch=s2)
    assert not torch.equal(buf.sigma_data2, before), "σ_data did not update in training"
    assert int(buf.ema_step[1]) == int(step_before[1]) + 1  # t=2 -> slot 1


def _const_batch(v: float, n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs whose per-t estimator ``bar`` equals exactly ``v``.

    ``μ̂`` all-equal → ``tr Var[μ̂] = 0``; ``σ² = v`` everywhere →
    ``avg_post_var = v·d``; so ``bar = (v·d + 0)/d = v``.
    """
    return torch.zeros(n, D), torch.full((n, D), float(v))


def test_warmup_forgets_init_on_first_update() -> None:
    """The first real update fully replaces the uninformative init (α=1).

    With a high steady-state decay (γ=0.99) the *old* plain EMA would land at
    ``0.99·init + 0.01·v`` after one step and take ~hundreds of steps to forget
    ``init``. The warmup makes the buffer equal the batch estimate ``v`` after a
    single update, regardless of ``init_value`` or γ.
    """
    buf = SigmaDataBuffer(
        T_max=T_MAX, tracking_mode="per_t", ema_decay=0.99, init_value=5.0
    )
    mu, s2 = _const_batch(0.2)
    with torch.enable_grad():
        buf.update(t_idx=torch.tensor([2]), mu_hat_batch=mu, sigma_t2_batch=s2)
    assert pytest.approx(float(buf.read(2).item()), rel=1e-5) == 0.2


def test_warmup_is_exact_running_mean_then_crosses_to_ema() -> None:
    """During warmup the buffer is the exact arithmetic mean of estimates seen.

    γ=0.9 → crossover at ``n+1 = 1/(1-γ) = 10``; the three updates below stay
    inside the warmup, so the buffer equals ``mean([1,3,5]) = 3``. Then a fourth
    update made *after* forcing the slot past the crossover moves at the slow EMA
    rate ``1-γ``, confirming the handover to plain EMA.
    """
    buf = SigmaDataBuffer(
        T_max=T_MAX, tracking_mode="per_t", ema_decay=0.9, init_value=0.0
    )
    for v in (1.0, 3.0, 5.0):
        mu, s2 = _const_batch(v)
        with torch.enable_grad():
            buf.update(t_idx=torch.tensor([2]), mu_hat_batch=mu, sigma_t2_batch=s2)
    assert pytest.approx(float(buf.read(2).item()), rel=1e-5) == 3.0

    # Push the slot well past the crossover at the steady value 3.0, then a
    # distinct update should move only by the EMA rate (0.1), not fully replace.
    for _ in range(50):
        mu, s2 = _const_batch(3.0)
        with torch.enable_grad():
            buf.update(t_idx=torch.tensor([2]), mu_hat_batch=mu, sigma_t2_batch=s2)
    assert pytest.approx(float(buf.read(2).item()), rel=1e-4) == 3.0
    mu, s2 = _const_batch(13.0)
    with torch.enable_grad():
        buf.update(t_idx=torch.tensor([2]), mu_hat_batch=mu, sigma_t2_batch=s2)
    assert pytest.approx(float(buf.read(2).item()), rel=1e-4) == 0.9 * 3.0 + 0.1 * 13.0


def test_n_updates_is_persisted_in_state_dict() -> None:
    """``n_updates`` round-trips through ``state_dict`` so resume keeps warmup state."""
    buf = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    assert "n_updates" in buf.state_dict()
    mu, s2 = _const_batch(1.0)
    with torch.enable_grad():
        buf.update(t_idx=torch.tensor([2]), mu_hat_batch=mu, sigma_t2_batch=s2)
    sd = buf.state_dict()
    fresh = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t")
    fresh.load_state_dict(sd)
    assert torch.equal(fresh.n_updates, buf.n_updates)


def test_update_no_grad_gate_covers_global_ema() -> None:
    """The no_grad gate protects the ``global_ema`` branch too."""
    buf = SigmaDataBuffer(
        T_max=T_MAX, tracking_mode="global_ema", ema_decay=0.9, init_value=1.0
    )
    t_idx = torch.tensor([1, 3])
    mu = torch.randn(8, D)
    s2 = torch.rand(8, D)
    before = buf.sigma_data2.clone()
    with torch.no_grad():
        buf.update(t_idx=t_idx, mu_hat_batch=mu, sigma_t2_batch=s2)
    assert torch.equal(buf.sigma_data2, before)
    with torch.enable_grad():
        buf.update(t_idx=t_idx, mu_hat_batch=mu, sigma_t2_batch=s2)
    assert not torch.equal(buf.sigma_data2, before)


# ---------------------------------------------------------------------------
# M2 red tests — sufficient statistics, DDP all-reduce
# ---------------------------------------------------------------------------


def test_suff_stats_shapes_match_convention() -> None:
    """``_suff_stats_per_t`` returns the correct shapes per the M2 spec.

    sum_mu: (n, d), sum_mu2_total: (n,), sum_s2_total: (n,), count: (n,).
    """
    inputs = make_m2_inputs()
    idx = inputs["idx"]
    mu = inputs["mu_hat_batch"]
    s2 = inputs["sigma_t2_batch"]
    n = idx.shape[0]
    d = mu.shape[1]

    stats = SigmaDataBuffer._suff_stats_per_t(idx, mu, s2)
    assert stats["sum_mu"].shape == (n, d), f"sum_mu shape: {stats['sum_mu'].shape}"
    assert stats["sum_mu2_total"].shape == (n,), (
        f"sum_mu2_total shape: {stats['sum_mu2_total'].shape}"
    )
    assert stats["sum_s2_total"].shape == (n,), (
        f"sum_s2_total shape: {stats['sum_s2_total'].shape}"
    )
    assert stats["count"].shape == (n,), f"count shape: {stats['count'].shape}"


def test_suff_stats_are_linear_in_batch() -> None:
    """Sufficient statistics are additive: split a batch in half and the two
    halves' stats sum to the whole batch's stats.

    This is the core DDP property: each rank sees a partial batch and
    all-reduces the sums — the result equals running on the full batch.
    """
    torch.manual_seed(M2_SEED)
    n = M2_N_T
    per_t = M2_PER_T * 2  # use 8 per t so we can split 4+4
    N = n * per_t
    d = M2_D
    idx = torch.tensor([1, 2, 3], dtype=torch.long)
    mu = torch.randn(N, d)
    s2 = torch.exp(torch.randn(N, d) * 0.3)

    # Full batch stats
    stats_full = SigmaDataBuffer._suff_stats_per_t(idx, mu, s2)

    # Split into two halves along the per-t axis
    half = per_t // 2
    # Rows are blocked by t: [0:half, half:per_t] for t=1, etc.
    mu_halves = [
        torch.cat([mu[k * per_t : k * per_t + half] for k in range(n)]),
        torch.cat([mu[k * per_t + half : (k + 1) * per_t] for k in range(n)]),
    ]
    s2_halves = [
        torch.cat([s2[k * per_t : k * per_t + half] for k in range(n)]),
        torch.cat([s2[k * per_t + half : (k + 1) * per_t] for k in range(n)]),
    ]

    stats_a = SigmaDataBuffer._suff_stats_per_t(idx, mu_halves[0], s2_halves[0])
    stats_b = SigmaDataBuffer._suff_stats_per_t(idx, mu_halves[1], s2_halves[1])

    # Linearity: sum_mu, sum_mu2_total, sum_s2_total, count are all additive
    assert torch.allclose(
        stats_a["sum_mu"] + stats_b["sum_mu"], stats_full["sum_mu"], atol=1e-5
    ), "sum_mu is not additive"
    assert torch.allclose(
        stats_a["sum_mu2_total"] + stats_b["sum_mu2_total"],
        stats_full["sum_mu2_total"],
        atol=1e-5,
    ), "sum_mu2_total is not additive"
    assert torch.allclose(
        stats_a["sum_s2_total"] + stats_b["sum_s2_total"],
        stats_full["sum_s2_total"],
        atol=1e-5,
    ), "sum_s2_total is not additive"
    assert torch.equal(
        stats_a["count"] + stats_b["count"], stats_full["count"]
    ), "count is not additive"


def test_estimator_from_suff_stats_matches_original_estimator() -> None:
    """``_estimator_from_suff_stats`` matches ``M2_ESTIMATOR_PER_T`` from
    golden values on the same batch (pre-refactor ground truth).

    The refactored formula ``(sum_mu2_total − (sum_mu)²/count) / (count−1)``
    is algebraically equivalent to the pre-refactor ``var(unbiased=True)``
    but uses a different floating-point evaluation order.  We therefore
    accept agreement within float32 rounding tolerance (atol=1e-5) rather
    than requiring bit-level identity — the difference is at most ~1e-7
    relative on these inputs (verified during implementation).
    """
    inputs = make_m2_inputs()
    idx = inputs["idx"]
    mu = inputs["mu_hat_batch"]
    s2 = inputs["sigma_t2_batch"]
    d = mu.shape[1]

    stats = SigmaDataBuffer._suff_stats_per_t(idx, mu, s2)
    result = SigmaDataBuffer._estimator_from_suff_stats(stats, d)

    expected = torch.tensor(M2_ESTIMATOR_PER_T)
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5), (
        f"Estimator mismatch:\n  got:      {result.tolist()}\n"
        f"  expected: {expected.tolist()}"
    )


def test_all_reduce_called_on_suff_stats_when_dist_initialized(
    monkeypatch,
) -> None:
    """When ``dist.is_available()`` and ``dist.is_initialized()`` are both True,
    ``_update_unchecked`` calls ``all_reduce`` exactly once with ``op=SUM``.
    """
    import ddssm.model.centering.sigma_data as _mod

    called_args = []

    def fake_all_reduce(tensor, op):
        called_args.append(("all_reduce", op))
        # no-op: tensor unchanged (simulates rank-0 = only rank)

    monkeypatch.setattr(_mod, "_dist_available", True)
    monkeypatch.setattr(_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(_mod.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(_mod.dist, "ReduceOp", _mod.dist.ReduceOp)

    buf = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t", ema_decay=0.9)
    inputs = make_m2_inputs()
    with torch.enable_grad():
        buf._update_unchecked(
            inputs["idx"], inputs["mu_hat_batch"], inputs["sigma_t2_batch"]
        )

    assert len(called_args) == 1, (
        f"Expected all_reduce called exactly once, got {len(called_args)}"
    )
    op_used = called_args[0][1]
    assert op_used == _mod.dist.ReduceOp.SUM, (
        f"Expected ReduceOp.SUM, got {op_used}"
    )


@pytest.mark.parametrize(
    "available,initialized",
    [
        (False, False),
        (False, True),
        (True, False),
    ],
)
def test_no_reduce_when_dist_not_available_or_initialized(
    monkeypatch, available, initialized
) -> None:
    """``all_reduce`` is NOT called when dist is unavailable or not initialized."""
    import ddssm.model.centering.sigma_data as _mod

    called = []

    def fake_all_reduce(tensor, op):
        called.append(1)

    monkeypatch.setattr(_mod, "_dist_available", available)
    if available:
        monkeypatch.setattr(_mod.dist, "is_initialized", lambda: initialized)
        monkeypatch.setattr(_mod.dist, "all_reduce", fake_all_reduce)

    buf = SigmaDataBuffer(T_max=T_MAX, tracking_mode="per_t", ema_decay=0.9)
    inputs = make_m2_inputs()
    with torch.enable_grad():
        buf._update_unchecked(
            inputs["idx"], inputs["mu_hat_batch"], inputs["sigma_t2_batch"]
        )

    assert len(called) == 0, (
        f"all_reduce should NOT be called when available={available},"
        f" initialized={initialized}"
    )


def test_ranks_agree_after_reduce_two_process_mock() -> None:
    """Two simulated ranks converge to the same ``bar`` after mock all-reduce.

    We build two buffers with non-overlapping half-batches, compute suff
    stats independently, sum them (mocking the all_reduce), pass to the
    pure estimator on each side, and verify both produce identical results
    — equal to what a single-rank run on the full batch would produce.
    """
    torch.manual_seed(M2_SEED + 10)
    n = M2_N_T
    per_t = M2_PER_T * 2
    N = n * per_t
    d = M2_D
    idx = torch.tensor([1, 2, 3], dtype=torch.long)
    mu = torch.randn(N, d)
    s2 = torch.exp(torch.randn(N, d) * 0.3)

    half = per_t // 2
    mu_a = torch.cat([mu[k * per_t : k * per_t + half] for k in range(n)])
    mu_b = torch.cat([mu[k * per_t + half : (k + 1) * per_t] for k in range(n)])
    s2_a = torch.cat([s2[k * per_t : k * per_t + half] for k in range(n)])
    s2_b = torch.cat([s2[k * per_t + half : (k + 1) * per_t] for k in range(n)])

    stats_a = SigmaDataBuffer._suff_stats_per_t(idx, mu_a, s2_a)
    stats_b = SigmaDataBuffer._suff_stats_per_t(idx, mu_b, s2_b)

    # Mock all-reduce SUM: each rank receives the global sum
    combined = {
        "sum_mu": stats_a["sum_mu"] + stats_b["sum_mu"],
        "sum_mu2_total": stats_a["sum_mu2_total"] + stats_b["sum_mu2_total"],
        "sum_s2_total": stats_a["sum_s2_total"] + stats_b["sum_s2_total"],
        "count": stats_a["count"] + stats_b["count"],
    }

    bar_a = SigmaDataBuffer._estimator_from_suff_stats(combined, d)
    bar_b = SigmaDataBuffer._estimator_from_suff_stats(combined, d)

    # Both ranks produce identical results after reduce
    assert torch.equal(bar_a, bar_b), "Ranks produce different estimates after reduce"

    # Also verify against single-rank full-batch result
    stats_full = SigmaDataBuffer._suff_stats_per_t(idx, mu, s2)
    bar_full = SigmaDataBuffer._estimator_from_suff_stats(stats_full, d)
    assert torch.allclose(bar_a, bar_full, atol=1e-5), (
        f"Mock-DDP result differs from full-batch:\n"
        f"  bar_a={bar_a.tolist()}\n  bar_full={bar_full.tolist()}"
    )


def test_per_t_one_fallback_uses_combined_count() -> None:
    """The ``per_t==1 → mu_var=0`` fallback operates on the *combined* count.

    Single rank with per_t=1 → count=1 → mu_var=0 (no dispersion info).
    Two mocked ranks, each with per_t=1 → combined count=2 → mu_var is NOT
    zeroed (real dispersion exists between the two samples).
    """
    torch.manual_seed(M2_SEED + 20)
    n = M2_N_T
    d = M2_D
    idx = torch.tensor([1, 2, 3], dtype=torch.long)

    # Each "rank" has exactly 1 sample per t
    mu_a = torch.randn(n, d)
    s2_a = torch.exp(torch.randn(n, d) * 0.3)
    mu_b = torch.randn(n, d)
    s2_b = torch.exp(torch.randn(n, d) * 0.3)

    # Single-rank: per_t=1 → mu_var must be 0
    stats_single = SigmaDataBuffer._suff_stats_per_t(idx, mu_a, s2_a)
    bar_single = SigmaDataBuffer._estimator_from_suff_stats(stats_single, d)
    # With count=1, mu_var=0; bar = avg_post_var / d
    expected_single = stats_single["sum_s2_total"] / stats_single["count"] / float(d)
    assert torch.allclose(bar_single, expected_single, atol=1e-6), (
        "Single-rank per_t=1 should zero mu_var"
    )

    # Two mocked ranks: combine their suff stats (SUM all-reduce)
    stats_a = SigmaDataBuffer._suff_stats_per_t(idx, mu_a, s2_a)
    stats_b = SigmaDataBuffer._suff_stats_per_t(idx, mu_b, s2_b)
    combined = {
        "sum_mu": stats_a["sum_mu"] + stats_b["sum_mu"],
        "sum_mu2_total": stats_a["sum_mu2_total"] + stats_b["sum_mu2_total"],
        "sum_s2_total": stats_a["sum_s2_total"] + stats_b["sum_s2_total"],
        "count": stats_a["count"] + stats_b["count"],
    }
    # combined count = 2 per t → mu_var should be nonzero (real dispersion)
    assert torch.all(combined["count"] == 2), "Combined count should be 2"

    bar_combined = SigmaDataBuffer._estimator_from_suff_stats(combined, d)
    # Compute what mu_var should be: Bessel-corrected across the 2 samples per t
    # mu_var = (sum_mu2_total - sum_mu^2/count) / (count - 1)
    mu_var_expected = (
        combined["sum_mu2_total"]
        - combined["sum_mu"].pow(2).sum(dim=1) / combined["count"]
    ) / (combined["count"] - 1)
    # mu_var should be non-zero (mu_a != mu_b)
    assert torch.any(mu_var_expected > 0), (
        "Expected non-zero mu_var when combining two distinct rank samples"
    )
    # bar_combined should NOT equal the avg_post_var-only result
    avg_post_var = combined["sum_s2_total"] / combined["count"] / float(d)
    assert not torch.allclose(bar_combined, avg_post_var, atol=1e-6), (
        "Combined result should include mu_var contribution"
    )
