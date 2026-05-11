"""ESM vs DSM ablation tests for ``DiffusionV2Transition``."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from ddssm.transitions.diffusion_v2 import DiffusionV2ScheduleConfig

from .conftest import make_transition, compute_per_sample_loss


def _build_pair(num_steps: int = 16):
    """Build matching ESM and DSM transitions sharing weights."""
    torch.manual_seed(0)
    cfg_esm = DiffusionV2ScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=num_steps,
        k_sampling_mode="uniform",
        objective="esm",
    )
    cfg_dsm = DiffusionV2ScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=num_steps,
        k_sampling_mode="uniform",
        objective="dsm",
    )
    t_esm = make_transition(schedule=cfg_esm)
    t_dsm = make_transition(schedule=cfg_dsm)
    # Sync weights
    t_dsm.load_state_dict(t_esm.state_dict())
    return t_esm, t_dsm


@pytest.mark.slow
def test_esm_dsm_both_finite_and_dsm_geq_esm(fixed_batch):
    """Sanity: both ESM and DSM produce finite, comparable losses.

    The textbook claim "DSM and ESM agree in expectation" applies to the
    *gradient* (Vincent 2011) but the loss values differ by a constant equal
    to the difference of marginal-vs-conditional score-norm expectations.  The
    DSM estimator's expected loss is therefore generally *greater than* the
    ESM expected loss by that non-negative constant.
    """
    t_esm, t_dsm = _build_pair(num_steps=16)
    N = 200
    losses_esm = compute_per_sample_loss(
        t_esm, fixed_batch, N, seed=1000, resample_zs=True
    )
    losses_dsm = compute_per_sample_loss(
        t_dsm, fixed_batch, N, seed=2000, resample_zs=True
    )

    assert np.all(np.isfinite(losses_esm))
    assert np.all(np.isfinite(losses_dsm))
    # DSM expected loss should not be smaller than ESM expected loss.
    sem = max(np.std(losses_esm), np.std(losses_dsm)) / np.sqrt(N)
    assert np.mean(losses_dsm) >= np.mean(losses_esm) - 5.0 * max(sem, 1e-8)


@pytest.mark.slow
def test_esm_variance_lower_than_dsm(fixed_batch):
    """Rao–Blackwell prediction: Var(L_p_esm) <= Var(L_p_dsm) (with slack).

    Both estimators are evaluated on the same resampled-``zs`` schedule so the
    comparison is apples-to-apples.
    """
    t_esm, t_dsm = _build_pair(num_steps=16)
    N = 400
    losses_esm = compute_per_sample_loss(
        t_esm, fixed_batch, N, seed=10, resample_zs=True
    )
    losses_dsm = compute_per_sample_loss(
        t_dsm, fixed_batch, N, seed=20, resample_zs=True
    )
    var_esm = float(np.var(losses_esm))
    var_dsm = float(np.var(losses_dsm))
    # 20% slack: the variance gap is most pronounced when the encoder noise is
    # large; with a small batch and modest sigma2_t the gap is finite but real.
    assert var_esm <= var_dsm * 1.20, f"var_esm={var_esm}, var_dsm={var_dsm}"


def test_esm_dsm_agree_in_degenerate_limit(transition, fixed_batch):
    """When encoder variance is ~0, ESM and DSM losses agree pointwise."""
    zs, enc_stats, time_embed, logq_paths = fixed_batch
    # Set logvars very negative -> sigma2_t ~ 0
    enc_stats_deg = {
        "mus": zs.clone(),  # mu_t = z_t at every timestep
        "logvars": torch.full_like(enc_stats["logvars"], -100.0),
    }

    cfg_esm = DiffusionV2ScheduleConfig(S_k=1, k_chunk=1, num_steps=16, objective="esm")
    cfg_dsm = DiffusionV2ScheduleConfig(S_k=1, k_chunk=1, num_steps=16, objective="dsm")
    torch.manual_seed(0)
    t_esm = make_transition(schedule=cfg_esm)
    torch.manual_seed(0)
    t_dsm = make_transition(schedule=cfg_dsm)
    t_dsm.load_state_dict(t_esm.state_dict())

    torch.manual_seed(42)
    out_esm = t_esm.transition_kl(
        enc_stats=enc_stats_deg,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
    )
    torch.manual_seed(42)
    out_dsm = t_dsm.transition_kl(
        enc_stats=enc_stats_deg,
        zs=zs,
        logq_paths=logq_paths,
        time_embed=time_embed,
    )
    # L_p should agree to high precision (logvars=-100 -> exp ~ 0).
    assert torch.allclose(out_esm["L_p"], out_dsm["L_p"], atol=1e-3, rtol=1e-3)
