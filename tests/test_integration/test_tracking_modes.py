"""The three tracking modes behave as ``model-v2.org`` § Tracking-mode variants predicts.

* ``fixed``      — a true constant σ_data²=init_value, frozen from
                    construction; ``update`` is a permanent no-op.
* ``global_ema`` — one shared scalar across all t.
* ``per_t``      — an independent EMA per timestep.

These tests verify each mode's integrated behaviour through a training
loop, complementing the unit-level checks in
``tests/test_centering/test_sigma_data.py``.
"""

from __future__ import annotations

import torch
import pytest

from .conftest import run_stage, make_vhp_model, make_smooth_sine_data

pytestmark = pytest.mark.slow


def test_global_ema_all_per_t_buffer_entries_equal() -> None:
    """Under ``global_ema``, every per-t buffer entry is the same scalar."""
    torch.manual_seed(0)
    model = make_vhp_model(
        baseline_form="persistence",
        tracking_mode="global_ema",
        sigma_data_init=1.0,
    )
    run_stage(
        model=model,
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=8,
            T=8,
            seed=torch.randint(0, 10_000, (1,)).item(),
        ),
        n_steps=20,
    )
    buf = model.sigma_data.sigma_data2
    assert torch.allclose(buf, buf[0].expand_as(buf), atol=1e-6), (
        f"global_ema buffer is not uniform: {buf.tolist()}"
    )


def test_per_t_buffer_diverges_more_than_global_ema() -> None:
    """``per_t`` keeps strictly more per-t resolution than ``global_ema``."""

    def _run_and_get_buf(tracking_mode: str) -> torch.Tensor:
        torch.manual_seed(0)
        model = make_vhp_model(
            baseline_form="persistence",
            tracking_mode=tracking_mode,
            sigma_data_init=1.0,
        )
        run_stage(
            model=model,
            data_factory=lambda: make_smooth_sine_data(
                n_seqs=4,
                T=6,
                seed=int(torch.randint(0, 10_000, (1,)).item()),
            ),
            n_steps=40,
        )
        return model.sigma_data.sigma_data2[:6]  # t = 1..6

    per_t_buf = _run_and_get_buf("per_t")
    global_buf = _run_and_get_buf("global_ema")

    per_t_spread = float((per_t_buf.max() - per_t_buf.min()).item())
    global_spread = float((global_buf.max() - global_buf.min()).item())
    assert global_spread == 0.0, (
        f"global_ema spread should be exactly 0, got {global_spread:.2e}"
    )
    assert per_t_spread > 0.0, (
        f"per_t spread should be > 0, got {per_t_spread:.2e} (buf={per_t_buf.tolist()})"
    )


def test_fixed_tracking_is_frozen_from_construction() -> None:
    """Under ``fixed`` the buffer is frozen at ``init_value`` from step 0."""
    torch.manual_seed(0)
    model = make_vhp_model(
        baseline_form="persistence",
        tracking_mode="fixed",
        sigma_data_init=1.0,
    )
    assert model.sigma_data.frozen is True
    pre_buf = model.sigma_data.sigma_data2.clone()

    run_stage(
        model=model,
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=4,
            T=8,
            seed=torch.randint(0, 10_000, (1,)).item(),
        ),
        n_steps=20,
    )
    post_buf = model.sigma_data.sigma_data2
    assert torch.equal(pre_buf, post_buf), (
        "fixed-mode buffer changed during training "
        f"(drift = {(post_buf - pre_buf).abs().max().item():.2e})"
    )
    assert torch.all(post_buf == 1.0)


def test_per_t_buffer_updates_during_training() -> None:
    """Under ``per_t`` the buffer moves off its init value during training."""
    torch.manual_seed(0)
    model = make_vhp_model(
        baseline_form="persistence",
        tracking_mode="per_t",
        sigma_data_init=1.0,
    )
    assert model.sigma_data.frozen is False
    pre_buf = model.sigma_data.sigma_data2.clone()

    run_stage(
        model=model,
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=4,
            T=8,
            seed=torch.randint(0, 10_000, (1,)).item(),
        ),
        n_steps=20,
    )
    post_buf = model.sigma_data.sigma_data2
    assert not torch.equal(pre_buf, post_buf), (
        "per_t buffer should have updated during training"
    )
