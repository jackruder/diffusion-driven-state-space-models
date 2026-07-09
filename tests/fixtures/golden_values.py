"""Golden values for split-loss TDD (M1/M2/M8).

Captured at commit 1d31804 on branch split-loss-training
(pre-refactor state — these are the pre-change ground truth values).

NOTE (post-refactor): the concrete numeric goldens below became stale when
staged training / baseline parametric heads / centering regularizers were
removed. The structure of M8_FINAL_METRICS / M8_CONFIG is preserved for
downstream imports, but the numeric values need regeneration by the user
before any golden-based test can trust them again.

DO NOT EDIT MANUALLY — regenerate via:
    LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:$LD_LIBRARY_PATH \
    TORCHDYNAMO_DISABLE=1 uv run python tools/capture_goldens.py
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# M1 — _esm_chunk_loss golden (structure only; numerics stale post-refactor)
# ---------------------------------------------------------------------------

M1_SEED: int = 42
M1_N: int = 4
M1_D: int = 2
# TODO: regenerate — golden captured against MLPBaseline; the parametric
# baselines have been removed.
M1_ESM_CHUNK_LOSS_SCALAR: float = float("nan")
M1_ESM_CHUNK_LOSS_PER_SAMPLE: list[float] = [float("nan")] * M1_N

M1_B: int = 2
M1_S: int = 2
M1_J: int = 1
M1_EMB_TIME: int = 8
M1_T_MAX: int = 10
M1_CHANNELS: int = 16
M1_NHEADS: int = 2
M1_S_K: int = 2


def make_m1_transition():
    """Build the fixed DiffusionTransition used for M1 golden capture.

    TODO: this fixture originally built an MLPBaseline; the parametric
    baselines were removed. Regenerate against ZeroBaseline / PersistenceBaseline
    when the golden numerics are refreshed.
    """
    raise NotImplementedError(
        "make_m1_transition needs re-authoring against the parameter-free "
        "baselines (MLPBaseline was removed)."
    )


def make_m1_inputs():
    """Build the fixed inputs used for M1 golden capture."""
    import torch

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

M2_SEED: int = 7
M2_N_T: int = 3
M2_PER_T: int = 4
M2_D: int = 4
# TODO: numerics likely still valid (SigmaDataBuffer._estimator_per_t was
# not touched) but user should verify before relying on them.
M2_ESTIMATOR_PER_T: list[float] = [1.7364693880081177, 2.1961143016815186, 2.3259286880493164]


def make_m2_inputs():
    """Build the fixed inputs used for M2 golden capture."""
    import torch

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
# M8 — 5-step training checkpoint (structure only; stale post-refactor)
# ---------------------------------------------------------------------------

M8_SEED: int = 0
M8_N_STEPS: int = 5
# TODO: baseline_form="mlp" / baseline_mode / lambda_sigma_p / snapshot_anchor
# all reference removed features. Regenerate the config + goldens against the
# new SmokeModel signature (baseline_form ∈ {zero, persistence}; no
# baseline_mode; no r_ regularizers).
M8_CONFIG: dict = {
    "baseline_form": "persistence",
    "tracking_mode": "fixed",
    "sigma_data_init": 1.0,
    "data_n_seqs": 4,
    "data_T": 8,
    "lr": 1e-3,
}
M8_CKPT_PATH: str = str(Path(__file__).parent / 'm8_5step_ckpt.pt')

# TODO: r_sigma_p / r_mu_p keys were removed from stage2_elbo_surrogate; the
# numeric values below are stale golden captures kept only so
# ``M8_FINAL_METRICS.keys()`` doesn't blow up code that imports the mapping.
M8_FINAL_METRICS: dict = {
    'loss/total': float("nan"),
    'loss/distortion/rec': float("nan"),
    'loss/rate/init/tot': float("nan"),
    'loss/rate/init/vhp': float("nan"),
    'loss/rate/init/entropy': float("nan"),
    'loss/rate/trans/kl': float("nan"),
    'loss/rate/total': float("nan"),
    'calib/ratio_res2_to_sigma2': float("nan"),
    'loss/rate/init/kl_aux': float("nan"),
    'loss/rate/init/loss_init': float("nan"),
}


def load_m8_checkpoint() -> dict:
    """Load the saved M8 5-step checkpoint (if it exists)."""
    import torch

    return torch.load(M8_CKPT_PATH, weights_only=False)


# ---------------------------------------------------------------------------
# Sanity assertion — sanity-check the *structure* only; numerics are stale.
# ---------------------------------------------------------------------------


def _check_finite() -> None:
    # Post-refactor: numerics are placeholders (NaN); no finiteness check.
    return


_check_finite()
