"""Training smoke tests for ``DiffusionV2Transition``."""

from __future__ import annotations

import pytest
import torch


def test_gradient_flows_to_score_net(transition, fixed_batch):
    """Backprop through ``L_p`` populates gradients on every score-net parameter."""
    zs, enc_stats, time_embed, logq_paths = fixed_batch
    transition.train()
    transition.zero_grad()
    out = transition.transition_kl(
        enc_stats=enc_stats, zs=zs, logq_paths=logq_paths, time_embed=time_embed,
    )
    out["L_p"].backward()
    n_with_grad = 0
    n_total = 0
    for p in transition.diffmodel.parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        if p.grad is not None:
            n_with_grad += 1
    # Every score-net parameter should have its grad attribute populated
    # (some may be exactly zero with S_k=1, but ``grad is not None`` should hold).
    assert n_total > 0
    assert n_with_grad == n_total


@pytest.mark.slow
def test_overfit_single_batch(fixed_batch):
    """Training on a single fixed batch reduces ``L_p`` substantially.

    Asserts on the ``kl`` component (the trainable score-matching loss): the
    closed-form Gaussian entropy ``L_q`` is constant across optimizer steps
    so monitoring ``L_p = kl + L_q`` masks the kl reduction.
    """
    from .conftest import make_transition
    from ddssm.transitions.diffusion_v2 import DiffusionV2ScheduleConfig

    cfg = DiffusionV2ScheduleConfig(
        S_k=16, k_chunk=16, num_steps=20, k_sampling_mode="uniform",
    )
    torch.manual_seed(0)
    transition = make_transition(schedule=cfg)
    transition.train()

    zs, enc_stats, time_embed, logq_paths = fixed_batch
    optim = torch.optim.Adam(transition.parameters(), lr=3e-3)

    kl_losses = []
    n_steps = 200
    for _ in range(n_steps):
        optim.zero_grad()
        out = transition.transition_kl(
            enc_stats=enc_stats, zs=zs, logq_paths=logq_paths, time_embed=time_embed,
        )
        loss = out["kl"]
        loss.backward()
        optim.step()
        kl_losses.append(float(loss.item()))

    init = sum(kl_losses[:10]) / 10.0
    final = sum(kl_losses[-20:]) / 20.0
    assert final < init * 0.7, f"kl did not decrease: init={init}, final={final}"


def test_device_consistency(transition, fixed_batch):
    """Forward pass on CUDA does not produce device-mismatch errors (if CUDA available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    zs, enc_stats, time_embed, logq_paths = fixed_batch
    device = torch.device("cuda")
    transition.to(device)
    zs = zs.to(device)
    time_embed = time_embed.to(device)
    logq_paths = logq_paths.to(device)
    enc_stats_d = {k: v.to(device) for k, v in enc_stats.items()}
    out = transition.transition_kl(
        enc_stats=enc_stats_d, zs=zs, logq_paths=logq_paths, time_embed=time_embed,
    )
    for v in out.values():
        assert v.device.type == "cuda"
        assert torch.isfinite(v).item()
