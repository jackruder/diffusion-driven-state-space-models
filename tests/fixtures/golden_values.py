"""Golden values for split-loss TDD (M1/M2/M8).

Captured at commit 1d31804 on branch split-loss-training
(pre-refactor state — these are the pre-change ground truth values).

DO NOT EDIT MANUALLY — regenerate via:
    LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:$LD_LIBRARY_PATH \
    TORCHDYNAMO_DISABLE=1 uv run python tools/capture_goldens.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

# ---------------------------------------------------------------------------
# M1 — _esm_chunk_loss golden
# ---------------------------------------------------------------------------
# Reproducer: build a tiny DiffusionTransition under SEED_M1=42,
# call _esm_chunk_loss on fixed inputs, capture the summed scalar.
#
# Key: M1 test_esm_chunk_loss_phith_reproduces_prior_single_loss checks
# that after refactor, `sum_phith / S_k` == this golden (bit-level).
# Under unit sigma_data and uniform p_k, weighted sqerr == sqerr * weight,
# and the pre-refactor single accumulator IS the phith accumulator.

M1_SEED: int = 42
M1_N: int = 4
M1_D: int = 2
M1_ESM_CHUNK_LOSS_SCALAR: float = 1.7797574996948242
M1_ESM_CHUNK_LOSS_PER_SAMPLE: list[float] = [0.07197477668523788, 1.4784289598464966, 0.03759221360087395, 0.1917615830898285]

# M1 model construction parameters (mirrors _make_diffusion_m1 in capture_goldens.py)
M1_B: int = 2
M1_S: int = 2
M1_J: int = 1
M1_EMB_TIME: int = 8
M1_T_MAX: int = 10
M1_CHANNELS: int = 16
M1_NHEADS: int = 2
M1_S_K: int = 2        # MC samples; >1 so weights != 1 (phith vs psi differ)


def make_m1_transition():
    """Build the fixed DiffusionTransition used for M1 golden capture."""
    from ddssm.nn.diffnets import CSDIUnet, FeatureMixerConfig, DiffResidualBlockConfig
    from ddssm.model.centering.baselines import ZeroBaseline
    from ddssm.model.transitions.diffusion import DiffusionTransition, DiffusionScheduleConfig

    torch.manual_seed(M1_SEED)
    np.random.seed(M1_SEED)
    baseline = ZeroBaseline(latent_dim=M1_D, j=M1_J)
    schedule = DiffusionScheduleConfig(
        S_k=M1_S_K,
        k_chunk=M1_S_K,
        num_steps=20,
        beta_min=0.1,
        beta_max=20.0,
        tau_min=1e-3,
        k_sampling_mode='uniform',
    )
    unet = partial(
        CSDIUnet,
        channels=M1_CHANNELS,
        n_layers=1,
        embedding_dim=M1_CHANNELS,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=M1_NHEADS, n_layers=1)
        ),
    )
    transition = DiffusionTransition(
        baseline=baseline,
        latent_dim=M1_D,
        j=M1_J,
        emb_time_dim=M1_EMB_TIME,
        T_max=M1_T_MAX,
        unet=unet,
        schedule=schedule,
    )
    transition.eval()
    return transition


def make_m1_inputs():
    """Build the fixed inputs used for M1 golden capture."""
    torch.manual_seed(M1_SEED + 1)
    N = M1_N
    d = M1_D
    J = M1_J
    EMB_TIME = M1_EMB_TIME
    mu_t = torch.randn(N, d)
    sigma2_t = torch.exp(torch.randn(N, d) * 0.5 - 1.0).clamp(1e-4, 2.0)
    z_hist = torch.randn(N, d, J)
    sigma_d2_per_row = torch.ones(N)
    padding_mask = torch.zeros(N, J + 1)
    time_win = torch.randn(N, J + 1, EMB_TIME)
    ctx = {
        'hist_time_emb': time_win[:, :J, :],
        'target_time_emb': time_win[:, J:, :],
    }
    return dict(
        mu_t=mu_t, sigma2_t=sigma2_t, z_hist=z_hist,
        ctx=ctx, sigma_d2_per_row=sigma_d2_per_row, padding_mask=padding_mask,
    )


# ---------------------------------------------------------------------------
# M2 — _estimator_per_t golden
# ---------------------------------------------------------------------------
# Reproducer: call SigmaDataBuffer._estimator_per_t directly on fixed tensors.
# M2 test_estimator_from_suff_stats_matches_original_estimator checks that
# after refactor, _estimator_from_suff_stats(suff_stats) == this golden (bit-level).

M2_SEED: int = 7
M2_N_T: int = 3          # distinct timesteps
M2_PER_T: int = 4        # samples per timestep
M2_D: int = 4          # feature dim
M2_ESTIMATOR_PER_T: list[float] = [1.7364695072174072, 2.1961143016815186, 2.3259286880493164]


def make_m2_inputs():
    """Build the fixed inputs used for M2 golden capture."""
    torch.manual_seed(M2_SEED)
    n = M2_N_T
    per_t = M2_PER_T
    N = n * per_t
    d = M2_D
    idx = torch.tensor([1, 2, 3], dtype=torch.long)
    mu_hat_batch = torch.randn(N, d)
    sigma_t2_batch = torch.exp(torch.randn(N, d) * 0.3)
    return dict(idx=idx, mu_hat_batch=mu_hat_batch, sigma_t2_batch=sigma_t2_batch)


# ---------------------------------------------------------------------------
# M8 — 5-step training checkpoint
# ---------------------------------------------------------------------------
# Reproducer: make_vhp_model() from tests/test_integration/conftest.py +
# run_stage(stage='stage_2', n_steps=5) under SEED_M8=0.
# use_split_loss=False (the only mode pre-refactor).
# M8 test_single_loss_off_path_regresses_none loads this checkpoint and
# checks that 5-step training produces bit-identical weights.

M8_SEED: int = 0
M8_N_STEPS: int = 5
M8_CONFIG: dict = {
    "baseline_form": "persistence",
    "tracking_mode": "fixed",
    "sigma_data_init": 1.0,
    "data_n_seqs": 4,
    "data_T": 8,
    "lr": 1e-3,
}
M8_CKPT_PATH: str = str(Path(__file__).parent / 'm8_5step_ckpt.pt')

# Final metrics from step 5 (for sanity checking):
M8_FINAL_METRICS: dict = {'loss/total': 11.77065658569336, 'loss/distortion/rec': 8.109908103942871, 'loss/rate/init/tot': 1.4702479839324951, 'loss/rate/init/vhp': 1.4702479839324951, 'loss/rate/init/entropy': 0.0, 'loss/rate/trans/kl': 2.190500497817993, 'loss/rate/total': 3.6607484817504883, 'calib/ratio_res2_to_sigma2': 0.3128020465373993, 'loss/rate/init/kl_aux': 0.6699551343917847, 'loss/rate/init/loss_init': 0.8002929091453552, 'loss/rate/init/loss_psi': 6.197436332702637, 'diag/sigma_data2/t=1': 1.0, 'diag/sigma_data2/t=2': 1.0, 'diag/sigma_data2/t=3': 1.0, 'diag/sigma_data2/t=4': 1.0, 'diag/sigma_data2/t=5': 1.0, 'diag/sigma_data2/t=6': 1.0, 'diag/sigma_data2/t=7': 1.0, 'diag/sigma_data2/t=8': 1.0, 'diag/sigma_data2/t=9': 1.0, 'diag/sigma_data2/t=10': 1.0, 'diag/sigma_data2/t=11': 1.0, 'diag/sigma_data2/t=12': 1.0, 'diag/sigma_data2/t=13': 1.0, 'diag/sigma_data2/t=14': 1.0, 'diag/sigma_data2/t=15': 1.0, 'diag/sigma_data2/t=16': 1.0, 'loss/rate/trans/kl_phith': 2.190500497817993, 'loss/rate/trans/kl_psi': 1.654646635055542}


def load_m8_checkpoint() -> dict:
    """Load the saved M8 5-step checkpoint."""
    return torch.load(M8_CKPT_PATH, weights_only=False)


# ---------------------------------------------------------------------------
# Sanity assertion (run on import in debug mode)
# ---------------------------------------------------------------------------

def _check_finite() -> None:
    assert torch.isfinite(torch.tensor(M1_ESM_CHUNK_LOSS_SCALAR)), 'M1 scalar non-finite'
    assert all(torch.isfinite(torch.tensor(x)) for x in M1_ESM_CHUNK_LOSS_PER_SAMPLE), 'M1 per_sample non-finite'
    assert all(torch.isfinite(torch.tensor(x)) for x in M2_ESTIMATOR_PER_T), 'M2 estimator non-finite'


_check_finite()
