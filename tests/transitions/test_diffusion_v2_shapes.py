"""Interface, shape, and error-handling tests for ``DiffusionV2Transition``."""

from __future__ import annotations

import torch
import pytest

from .conftest import EMB_TIME, LATENT_DIM, J, make_dummy_ctx


def test_forward_shapes(transition, fixed_batch):
    """``transition_kl`` returns scalar tensors with the expected keys."""
    zs, enc_stats, time_embed, logq_paths = fixed_batch
    out = transition.transition_kl(
        enc_stats=enc_stats, zs=zs, logq_paths=logq_paths, time_embed=time_embed,
    )
    assert set(out.keys()) == {"kl", "L_p", "L_q"}
    for v in out.values():
        assert v.ndim == 0
        assert torch.isfinite(v).item()
    assert torch.allclose(out["kl"], out["L_p"] - out["L_q"], atol=1e-4, rtol=1e-3)


def test_sample_returns_correct_shape(transition):
    """``sample(z_hist, ctx=...)`` returns shape ``(B, 1, d)``."""
    B = 3
    z_hist = torch.randn(B, LATENT_DIM, J)
    ctx = make_dummy_ctx(B, J, EMB_TIME)
    out = transition.sample(z_hist, ctx=ctx)
    assert out.shape == (B, 1, LATENT_DIM)
    assert torch.all(torch.isfinite(out))


def test_fallback_when_no_encoder_stats(transition, fixed_batch):
    """Missing encoder stats falls back to DSM-on-z_t and MC-entropy without crashing."""
    zs, _enc_stats, time_embed, logq_paths = fixed_batch
    # None
    out_none = transition.transition_kl(
        enc_stats=None, zs=zs, logq_paths=logq_paths, time_embed=time_embed,
    )
    # Empty dict
    out_empty = transition.transition_kl(
        enc_stats={}, zs=zs, logq_paths=logq_paths, time_embed=time_embed,
    )
    for out in (out_none, out_empty):
        assert set(out.keys()) == {"kl", "L_p", "L_q"}
        for v in out.values():
            assert v.ndim == 0
            assert torch.isfinite(v).item()


def test_disabled_methods_raise(transition):
    """``log_prob`` / ``log_likelihood`` / ``forward_kl_loss`` raise ``NotImplementedError``."""
    z = torch.randn(2, LATENT_DIM)
    z_hist = torch.randn(2, LATENT_DIM, J)
    with pytest.raises(NotImplementedError):
        transition.log_prob(z=z, z_hist=z_hist, ctx=None)
    with pytest.raises(NotImplementedError):
        transition.forward_kl_loss(z)
    with pytest.raises(NotImplementedError):
        transition.log_likelihood()


def test_zero_target_steps(transition):
    """When ``T <= j``, ``L_p`` is a scalar zero tensor without crashing."""
    B, S, d = 2, 1, LATENT_DIM
    T = J  # n_target_steps == 0
    zs = torch.randn(B, S, d, T)
    logq_paths = torch.randn(B, S, T)
    time_embed = torch.randn(B, T, EMB_TIME)
    mus = torch.randn(B, S, d, T)
    logvars = torch.randn(B, S, d, T) * 0.1 - 1.0
    enc_stats = {"mus": mus, "logvars": logvars}
    out = transition.transition_kl(
        enc_stats=enc_stats, zs=zs, logq_paths=logq_paths, time_embed=time_embed,
    )
    assert set(out.keys()) == {"kl", "L_p", "L_q"}
    assert torch.equal(out["kl"], torch.zeros_like(out["kl"]))
    # L_p == kl + L_q
    assert torch.allclose(out["L_p"], out["L_q"])
