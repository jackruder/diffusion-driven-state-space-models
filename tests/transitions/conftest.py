"""Shared fixtures and helpers for ``DiffusionV2Transition`` tests."""

from __future__ import annotations

from typing import Dict, Optional
from functools import partial

import numpy as np
import torch
import pytest

from ddssm.diffnets import (
    CSDIUnet,
    FeatureMixerConfig,
    DiffResidualBlockConfig,
)
from ddssm.transitions.diffusion_v2 import (
    DiffusionV2Transition,
    DiffusionV2ScheduleConfig,
)

# ---------------------------------------------------------------------------
# Tiny architectural constants (mirrors ``tests/test_model.py``).
# ---------------------------------------------------------------------------

J = 2
LATENT_DIM = 2
EMB_TIME = 8
CHANNELS = 8
NHEADS = 4


def _tiny_unet():
    """Tiny UNet builder for fast tests."""
    return partial(
        CSDIUnet,
        channels=CHANNELS,
        n_layers=1,
        embedding_dim=CHANNELS,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
    )


def make_transition(
    schedule: Optional[DiffusionV2ScheduleConfig] = None,
    latent_dim: int = LATENT_DIM,
    j: int = J,
    emb_time_dim: int = EMB_TIME,
) -> DiffusionV2Transition:
    """Construct a small ``DiffusionV2Transition`` for tests."""
    if schedule is None:
        schedule = DiffusionV2ScheduleConfig(
            S_k=1,
            k_chunk=1,
            num_steps=50,
            beta_min=0.1,
            beta_max=20.0,
            tau_min=1e-3,
            k_sampling_mode="uniform",
        )
    return DiffusionV2Transition(
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        unet=_tiny_unet(),
        schedule=schedule,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config() -> DiffusionV2ScheduleConfig:
    """Default schedule used by most tests."""
    return DiffusionV2ScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=50,
        beta_min=0.1,
        beta_max=20.0,
        tau_min=1e-3,
        k_sampling_mode="uniform",
    )


@pytest.fixture
def transition(small_config: DiffusionV2ScheduleConfig) -> DiffusionV2Transition:
    """A minimal ``DiffusionV2Transition`` instance shared across tests."""
    torch.manual_seed(0)
    return make_transition(schedule=small_config)


@pytest.fixture
def fixed_batch():
    """Return ``(zs, enc_stats, time_embed, logq_paths)`` with non-trivial encoder stats."""
    torch.manual_seed(123)
    B, S, d, T = 4, 2, LATENT_DIM, 8
    zs = torch.randn(B, S, d, T)
    mus = 0.5 * torch.randn(B, S, d, T)
    # logvars in a moderate range so sigma2_t = exp(logvars) is non-trivial
    logvars = -1.0 + 0.3 * torch.randn(B, S, d, T)
    time_embed = torch.randn(B, T, EMB_TIME)
    logq_paths = torch.randn(B, S, T)
    enc_stats = {"mus": mus, "logvars": logvars}
    return zs, enc_stats, time_embed, logq_paths


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def make_dummy_ctx(
    N: int,
    j: int,
    emb_dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Build a valid ``ctx`` dict with ``hist_time_emb`` / ``target_time_emb``.

    Shapes match what ``DiffusionV2Transition.sample`` expects:
      - ``hist_time_emb``: (N, j, emb_dim)
      - ``target_time_emb``: (N, 1, emb_dim)
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    return {
        "hist_time_emb": torch.zeros(N, j, emb_dim, device=device, dtype=dtype),
        "target_time_emb": torch.zeros(N, 1, emb_dim, device=device, dtype=dtype),
    }


def compute_per_sample_loss(
    transition: DiffusionV2Transition,
    batch,
    n_replicas: int,
    seed: int = 0,
    resample_zs: bool = False,
) -> np.ndarray:
    """Run ``transition_kl`` ``n_replicas`` times on a fixed batch.

    Returns ``L_p`` values as a numpy array (shape ``(n_replicas,)``).  Useful
    for variance-reduction / Rao–Blackwell comparison tests.

    If ``resample_zs`` is True, the latent paths ``zs`` are re-drawn from the
    encoder's Gaussian ``N(mus, exp(logvars))`` at every replica.  This is
    required when comparing ESM and DSM in expectation: ESM marginalises
    ``z_t`` analytically, while DSM consumes a sampled ``z_t``, so for the
    Rao–Blackwell identity ``E[L_p_dsm] = E[L_p_esm]`` to hold under
    Monte-Carlo we must also average over the ``z_t`` distribution.
    """
    zs, enc_stats, time_embed, logq_paths = batch
    losses = np.zeros(n_replicas, dtype=np.float64)
    transition.eval()
    mus = enc_stats.get("mus") if enc_stats is not None else None
    logvars = enc_stats.get("logvars") if enc_stats is not None else None
    with torch.no_grad():
        for i in range(n_replicas):
            torch.manual_seed(seed + i)
            if resample_zs and mus is not None and logvars is not None:
                std = (0.5 * logvars).exp()
                zs_i = mus + std * torch.randn_like(mus)
            else:
                zs_i = zs
            out = transition.transition_kl(
                enc_stats=enc_stats,
                zs=zs_i,
                logq_paths=logq_paths,
                time_embed=time_embed,
            )
            losses[i] = float(out["L_p"].item())
    return losses
