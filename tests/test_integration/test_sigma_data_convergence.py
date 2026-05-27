"""Math claim: stage-1 R_σp drives ``σ_data²(t) → 1``.

From ``model-v2.org`` § Data-variance tracking:

  "The KL pushes σ_t → σ_p(z_{t-1}) per dim per state, so the first
  term tracks E[‖σ_p(z_{t-1})‖²].  The KL also pushes μ_t → μ_p, so
  μ̂_t → 0 and the second term → 0.

  The log-variance regularizer R_σp anchors E[log σ_p²] near zero
  *globally* — averaged over (t, d, state)."

So with sufficient stage-1 training and ``λ_σp > 0``, the σ_data
EMA buffer entries should converge near 1.
"""

from __future__ import annotations

import torch
import pytest

from .conftest import (
    run_stage,
    make_vhp_model,
    make_smooth_sine_data,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("baseline_form", ["zero", "mlp"])
def test_sigma_data_converges_near_one_after_stage1(baseline_form: str) -> None:
    """Stage-1 training with R_σp drives σ_data²(t) toward 1.

    Step budget is kept small (~60 steps); we check the buffer has
    *moved* off its starting value and lies in a broad envelope around
    1, not that it has converged.
    """
    torch.manual_seed(42)
    model = make_vhp_model(
        baseline_form=baseline_form,
        lambda_sigma_p=1.0,  # strong anchor
        tracking_mode="per_t",
        sigma_data_init=0.3,  # start far from 1
    )
    pre_buf = model.sigma_data.sigma_data2.clone()

    def _factory():
        return make_smooth_sine_data(
            n_seqs=4, T=6, seed=int(torch.randint(0, 10_000, (1,)).item()),
        )

    run_stage(
        model=model,
        stage="stage_1",
        data_factory=_factory,
        n_steps=60,
        lr=5e-3,
    )

    visited = model.sigma_data.sigma_data2[:6]  # t = 1..6 → slots 0..5
    mean = float(visited.mean().item())
    assert 0.2 < mean < 5.0, (
        f"σ_data buffer mean {mean:.3f} out of plausible range for "
        f"baseline={baseline_form}"
    )
    assert not torch.allclose(visited, pre_buf[:6])


@pytest.mark.parametrize("tracking_mode", ["global_ema", "per_t"])
def test_sigma_data_buffer_actually_moves_in_stage1(tracking_mode: str) -> None:
    """Both EMA tracking modes update the buffer over training."""
    torch.manual_seed(0)
    model = make_vhp_model(
        baseline_form="mlp",
        lambda_sigma_p=1e-2,
        tracking_mode=tracking_mode,
        sigma_data_init=1.0,
    )
    pre = model.sigma_data.sigma_data2.clone()
    run_stage(
        model=model,
        stage="stage_1",
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=4, T=6, seed=int(torch.randint(0, 10_000, (1,)).item()),
        ),
        n_steps=20,
    )
    post = model.sigma_data.sigma_data2
    assert not torch.allclose(post, pre), (
        f"σ_data buffer didn't move under tracking_mode={tracking_mode!r}"
    )
