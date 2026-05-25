"""The simplest cell reduces to :class:`DiffusionV2Transition`.

This is the headline correctness anchor for Phase D: the simplest cell
in the 18-cell grid — ``(baseline_form="zero", baseline_mode="pinned",
tracking_mode="fixed")`` — is *configured* such that its stage-2
transition agrees with plain ``DiffusionV2Transition`` in the regime
where the two are mathematically equivalent.

Mathematical equivalence requires three things:

1. Baseline is :class:`ZeroBaseline` (μ_p ≡ 0), so V3's centered ESM
   target ẑ = z̃ − μ_p collapses to z̃ — V2's uncentered target.
2. :class:`SigmaDataBuffer` is in ``"fixed"`` tracking mode with
   ``init_value=1.0`` so the per-call EDM constants V3 computes from
   σ_data²(t) reduce to V2's hardcoded constants (see
   :func:`tests.test_transitions.test_diffusion_v3.test_edm_constants_reduce_to_v2_at_sigma_data_unit`).
3. The same baseline object is *shared* between the stage-1 and
   stage-2 transitions, so μ_p ≡ 0 is consistently applied across the
   handoff.

The numerical leg piggybacks on the existing transition-level
reduction test pattern at
``tests/test_transitions/test_diffusion_v3.py:192-261``: build a
matched :class:`DiffusionV2Transition` with the same dims and
schedule, call ``_vp_precondition`` on identical inputs on both
transitions, and assert the resulting ``(z_in, F_target)`` tensors
agree to ``atol=1e-4, rtol=1e-4``.
"""

from __future__ import annotations

import math
from functools import partial

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate

from conf.registry import store
from ddssm._experiment_registry import register_experiments
from ddssm.centering.baselines import ZeroBaseline
from ddssm.diffnets import (
    CSDIUnet,
    DiffResidualBlockConfig,
    FeatureMixerConfig,
)
from ddssm.transitions.diffusion_v2 import (
    DiffusionV2ScheduleConfig,
    DiffusionV2Transition,
)
from ddssm.transitions.diffusion_v3 import DiffusionV3Transition
from experiments.init_centering.cells import cell_name


SIMPLEST_CELL_NAME = cell_name("zero", "pinned", "fixed")


@pytest.fixture(autouse=True)
def _clear_global_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    register_experiments()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def _instantiate_simplest_cell():
    cfg = store["experiment"]["experiment", SIMPLEST_CELL_NAME]
    return instantiate(cfg)


# ---------------------------------------------------------------------------
# Structural lockdown: the cell is configured to satisfy the V2-reduction
# preconditions.
# ---------------------------------------------------------------------------


def test_simplest_cell_uses_zero_baseline() -> None:
    """μ_p ≡ 0 — the centering offset vanishes."""
    exp = _instantiate_simplest_cell()
    assert isinstance(exp.model.baseline, ZeroBaseline)


def test_simplest_cell_uses_fixed_unit_sigma_data() -> None:
    """σ_data ≡ 1 — V3's per-call EDM constants reduce to V2's hardcoded values."""
    exp = _instantiate_simplest_cell()
    sigma_data = exp.model.sigma_data
    assert sigma_data.tracking_mode == "fixed"
    assert math.isclose(sigma_data.init_value, 1.0)


def test_simplest_cell_shares_baseline_across_handoff() -> None:
    """Stage 1 and stage 2 see the same μ_p object (no parameter divergence)."""
    exp = _instantiate_simplest_cell()
    assert exp.model.stage1_transition.baseline is exp.model.baseline
    assert exp.model.transition.baseline is exp.model.baseline


# ---------------------------------------------------------------------------
# Numerical lockdown: V3._vp_precondition agrees with V2's on identical input.
# ---------------------------------------------------------------------------


def _tiny_unet_factory(channels: int = 8, n_layers: int = 1, nheads: int = 4):
    """A minimal CSDIUnet builder shared between the matched V2 and V3 transitions."""
    return partial(
        CSDIUnet,
        channels=channels,
        n_layers=n_layers,
        embedding_dim=channels,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=nheads, n_layers=1)
        ),
    )


def test_simplest_cell_vp_precondition_matches_v2() -> None:
    """V3's centered preconditioning collapses to V2's at μ_p≡0 + σ_data≡1.

    Build the cell's V3 transition through the Hydra preset; build a
    matched V2 transition with the same latent_dim / j / emb_time_dim
    and a schedule whose VP-SDE knobs are identical.  Feed both
    ``_vp_precondition`` calls the same ``(mu, sigma2, k_idx, eps)``
    and assert the output tensors agree to ``atol=1e-4, rtol=1e-4``.
    """
    exp = _instantiate_simplest_cell()
    v3: DiffusionV3Transition = exp.model.transition
    assert isinstance(v3, DiffusionV3Transition)

    # Build the matched V2 with the same VP-SDE schedule knobs the cell
    # uses by default (see ``_build_init_centering_model``: diffusion_*).
    schedule_v2 = DiffusionV2ScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=v3.num_steps,
        beta_min=v3.schedule.beta_min,
        beta_max=v3.schedule.beta_max,
        tau_min=v3.schedule.tau_min,
        k_sampling_mode=v3.schedule.k_sampling_mode,
    )
    v2 = DiffusionV2Transition(
        latent_dim=v3.latent_dim,
        j=v3.j,
        emb_time_dim=v3.emb_time_dim,
        unet=_tiny_unet_factory(),
        schedule=schedule_v2,
    )

    # Identical inputs for both preconditioners (μ̂ = μ since baseline is Zero).
    torch.manual_seed(0)
    N = 3
    d = v3.latent_dim
    mu = torch.randn(N, d)
    sigma2 = 0.1 + torch.rand(N, d)
    k_idx = torch.tensor([[0], [v3.num_steps // 2], [v3.num_steps - 1]])
    eps = torch.randn(N, d, 1)

    z_in_v2, F_tgt_v2 = v2._vp_precondition(
        mu_t=mu, sigma2_t=sigma2, k_idx=k_idx, eps=eps,
    )
    z_in_v3, F_tgt_v3 = v3._vp_precondition(
        mu_hat_t=mu,                 # μ̂ = μ − μ_p = μ − 0 = μ
        sigma2_t=sigma2,
        k_idx=k_idx,
        eps=eps,
        sigma_d2_per_row=torch.ones(N),  # σ_data² ≡ 1 (fixed buffer at init_value=1)
    )

    torch.testing.assert_close(z_in_v3, z_in_v2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(F_tgt_v3, F_tgt_v2, atol=1e-4, rtol=1e-4)
