"""Math claim: the three tracking modes behave as the doc predicts.

From ``model-v2.org`` § Tracking-mode variants:

* "Fixed (frozen at handoff value).  Hold σ_data²[t] at the per-t
  value the EMA buffer reached at handoff and skip the stage-2 EMA
  update."
* "Global EMA.  Tie all per-t buffers to a single scalar σ_data²,
  updated by bar_σ_data² = (1/|T|) Σ_t bar_σ_data²(t) ..."
* "Per-t EMA.  Independent buffers σ_data²[t], t = 2, …, T,
  updated as in /EMA update and gradient flow/ above."

These tests verify each mode's *integrated* behaviour through a
training loop, complementing the unit-level checks in
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
        baseline_form="mlp",
        tracking_mode="global_ema",
        lambda_sigma_p=1e-2,
        sigma_data_init=1.0,
    )
    run_stage(
        model=model,
        stage="stage_1",
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=8, T=8, seed=torch.randint(0, 10_000, (1,)).item(),
        ),
        n_steps=20,
    )
    buf = model.sigma_data.sigma_data2
    # All entries should be identical under "global_ema".
    assert torch.allclose(buf, buf[0].expand_as(buf), atol=1e-6), (
        f"global_ema buffer is not uniform: {buf.tolist()}"
    )


def test_per_t_buffer_diverges_more_than_global_ema() -> None:
    """``per_t`` keeps strictly more per-t resolution than ``global_ema``.

    On smooth synthetic data the absolute per-t spread can be small
    (the encoder learns a roughly uniform representation across t),
    but ``per_t`` should always retain *some* per-t spread while
    ``global_ema`` collapses every slot to a single scalar.  The
    comparison-against-global is the meaningful invariant.
    """
    def _run_and_get_buf(tracking_mode: str) -> torch.Tensor:
        torch.manual_seed(0)
        model = make_vhp_model(
            baseline_form="mlp",
            tracking_mode=tracking_mode,
            lambda_sigma_p=1e-2,
            sigma_data_init=1.0,
        )
        run_stage(
            model=model,
            stage="stage_1",
            data_factory=lambda: make_smooth_sine_data(
                n_seqs=4, T=6, seed=int(torch.randint(0, 10_000, (1,)).item()),
            ),
            n_steps=40,
        )
        return model.sigma_data.sigma_data2[:6]  # t = 1..6

    per_t_buf = _run_and_get_buf("per_t")
    global_buf = _run_and_get_buf("global_ema")

    per_t_spread = float((per_t_buf.max() - per_t_buf.min()).item())
    global_spread = float((global_buf.max() - global_buf.min()).item())
    # global_ema collapses to a single scalar — spread is exactly 0.
    assert global_spread == 0.0, (
        f"global_ema spread should be exactly 0, got {global_spread:.2e}"
    )
    # per_t should retain measurable spread.
    assert per_t_spread > 0.0, (
        f"per_t spread should be > 0, got {per_t_spread:.2e} "
        f"(buf={per_t_buf.tolist()})"
    )


def test_fixed_tracking_frozen_after_reset() -> None:
    """After ``reset_schedule`` under "fixed" mode, further updates are no-ops."""
    torch.manual_seed(0)
    model = make_vhp_model(
        baseline_form="mlp",
        tracking_mode="fixed",
        sigma_data_init=1.0,
    )
    # Run a few stage-1 steps so the buffer accumulates non-trivial values.
    run_stage(
        model=model,
        stage="stage_1",
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=4, T=8, seed=torch.randint(0, 10_000, (1,)).item(),
        ),
        n_steps=20,
    )
    pre_buf = model.sigma_data.sigma_data2.clone()
    # Simulate the handoff's schedule reset (sets ``frozen = True`` under
    # ``"fixed"`` tracking).
    model.sigma_data.reset_schedule()
    assert model.sigma_data.frozen is True

    # Further stage-2 updates should leave the buffer untouched.
    run_stage(
        model=model,
        stage="stage_2",
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=4, T=8, seed=torch.randint(0, 10_000, (1,)).item(),
        ),
        n_steps=10,
    )
    post_buf = model.sigma_data.sigma_data2
    assert torch.equal(pre_buf, post_buf), (
        "fixed-mode buffer changed after reset_schedule (drift = "
        f"{(post_buf - pre_buf).abs().max().item():.2e})"
    )


def test_per_t_buffer_continues_to_update_in_stage2() -> None:
    """Under ``per_t`` (or ``global_ema``), the buffer keeps updating in stage 2."""
    torch.manual_seed(0)
    model = make_vhp_model(
        baseline_form="mlp",
        tracking_mode="per_t",
        lambda_sigma_p=1e-2,
        sigma_data_init=1.0,
    )
    # Stage 1 a little to seed the buffer.
    run_stage(
        model=model,
        stage="stage_1",
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=4, T=8, seed=torch.randint(0, 10_000, (1,)).item(),
        ),
        n_steps=10,
    )
    # Reset the schedule (mimics the handoff under "per_t" — value
    # persists, schedule resets, but the buffer remains writable).
    pre_buf = model.sigma_data.sigma_data2.clone()
    model.sigma_data.reset_schedule()
    assert model.sigma_data.frozen is False  # "per_t" doesn't freeze

    # Stage 2 continues to update the buffer.
    run_stage(
        model=model,
        stage="stage_2",
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=4, T=8, seed=torch.randint(0, 10_000, (1,)).item(),
        ),
        n_steps=10,
    )
    post_buf = model.sigma_data.sigma_data2
    assert not torch.equal(pre_buf, post_buf), (
        "per_t buffer should have continued updating in stage 2"
    )
