"""The EMA tracking modes actually update the σ_data buffer during training.

Post-refactor there is no ``R_σp`` regularizer anchoring
``σ_data²(t) → 1``, so the convergence-toward-1 claim from
``model-v2.org`` no longer applies. We keep the weaker (and still
useful) invariant that the ``global_ema`` and ``per_t`` buffers move off
their init value during a short training run.
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


@pytest.mark.parametrize("tracking_mode", ["global_ema", "per_t"])
def test_sigma_data_buffer_moves_during_training(tracking_mode: str) -> None:
    """Both EMA tracking modes update the buffer over training."""
    torch.manual_seed(0)
    model = make_vhp_model(
        baseline_form="persistence",
        tracking_mode=tracking_mode,
        sigma_data_init=1.0,
    )
    pre = model.sigma_data.sigma_data2.clone()
    run_stage(
        model=model,
        data_factory=lambda: make_smooth_sine_data(
            n_seqs=4,
            T=6,
            seed=int(torch.randint(0, 10_000, (1,)).item()),
        ),
        n_steps=20,
    )
    post = model.sigma_data.sigma_data2
    assert not torch.allclose(post, pre), (
        f"σ_data buffer didn't move under tracking_mode={tracking_mode!r}"
    )
