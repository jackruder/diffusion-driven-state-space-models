"""Capture golden values for M1/M2/M8 red tests.

Run with:
    LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:$LD_LIBRARY_PATH \
    TORCHDYNAMO_DISABLE=1 uv run python tools/capture_goldens.py

Writes:
  tests/fixtures/golden_values.py  — Python module of float constants + reproducers
  tests/fixtures/m8_5step_ckpt.pt  — 5-step state_dict for M8 regression
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import partial
from types import SimpleNamespace

# Ensure repo root is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import torch
import numpy as np

# ---------------------------------------------------------------------------
# M1 golden — _esm_chunk_loss scalar under fixed RNG
# ---------------------------------------------------------------------------
# Use the same tiny diffusion as in tests/test_transitions/test_diffusion.py

from ddssm.nn.diffnets import (
    CSDIUnet,
    FeatureMixerConfig,
    DiffResidualBlockConfig,
)
from ddssm.model.centering.baselines import ZeroBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)

# Match the test file constants exactly
B = 2
S = 2
D = 2
J = 1
EMB_TIME = 8
T_MAX = 10
CHANNELS = 16
NHEADS = 2
SEED_M1 = 42


def _make_tiny_unet():
    return partial(
        CSDIUnet,
        channels=CHANNELS,
        n_layers=1,
        embedding_dim=CHANNELS,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
    )


def _make_diffusion_m1() -> DiffusionTransition:
    torch.manual_seed(SEED_M1)
    np.random.seed(SEED_M1)
    baseline = ZeroBaseline(latent_dim=D, j=J)
    schedule = DiffusionScheduleConfig(
        S_k=2,          # 2 MC samples so weights != 1 and _phith vs _psi would differ
        k_chunk=2,
        num_steps=20,
        beta_min=0.1,
        beta_max=20.0,
        tau_min=1e-3,
        k_sampling_mode="uniform",
    )
    transition = DiffusionTransition(
        baseline=baseline,
        latent_dim=D,
        j=J,
        emb_time_dim=EMB_TIME,
        T_max=T_MAX,
        unet=_make_tiny_unet(),
        schedule=schedule,
    )
    transition.eval()
    return transition


def capture_m1() -> dict:
    """Capture _esm_chunk_loss scalar and unit-weight comparison."""
    torch.manual_seed(SEED_M1)
    np.random.seed(SEED_M1)

    transition = _make_diffusion_m1()

    N = 4  # small batch (B*S for B=2, S=2)
    d = D

    # Fixed inputs
    torch.manual_seed(SEED_M1 + 1)
    mu_t = torch.randn(N, d)
    sigma2_t = torch.exp(torch.randn(N, d) * 0.5 - 1.0).clamp(1e-4, 2.0)
    z_hist = torch.randn(N, d, J)
    sigma_d2_per_row = torch.ones(N)   # unit sigma_data
    padding_mask = torch.zeros(N, J + 1)

    # Build ctx with required time embeddings
    time_win = torch.randn(N, J + 1, EMB_TIME)
    ctx = {
        "hist_time_emb": time_win[:, :J, :],
        "target_time_emb": time_win[:, J:, :],
    }

    # Capture with the SAME RNG state for reproducibility
    torch.manual_seed(SEED_M1 + 2)

    with torch.no_grad():
        loss_phith, loss_psi, _mu_hat = transition._esm_chunk_loss(
            mu_t=mu_t,
            sigma2_t=sigma2_t,
            z_hist=z_hist,
            ctx=ctx,
            sigma_d2_per_row=sigma_d2_per_row,
            padding_mask=padding_mask,
            return_per_sample=False,
        )

    # The scalar is: sum over N, divided by S_k inside _esm_chunk_loss.
    # We anchor the golden on the phith side (IS-weighted); psi is the
    # unit-weighted twin.
    golden_esm_scalar = float(loss_phith)

    torch.manual_seed(SEED_M1 + 2)
    with torch.no_grad():
        per_sample_phith, per_sample_psi, _ = transition._esm_chunk_loss(
            mu_t=mu_t,
            sigma2_t=sigma2_t,
            z_hist=z_hist,
            ctx=ctx,
            sigma_d2_per_row=sigma_d2_per_row,
            padding_mask=padding_mask,
            return_per_sample=True,
        )
    golden_per_sample = per_sample_phith.tolist()

    print(f"[M1] _esm_chunk_loss scalar (sum over N): {golden_esm_scalar}")
    print(f"[M1] _esm_chunk_loss per_sample: {golden_per_sample}")

    return {
        "esm_chunk_loss_scalar": golden_esm_scalar,
        "esm_chunk_loss_per_sample": golden_per_sample,
        "seed": SEED_M1,
        "N": N,
        "d": d,
    }


# ---------------------------------------------------------------------------
# M2 golden — _estimator_per_t output on a fixed batch
# ---------------------------------------------------------------------------

SEED_M2 = 7


def capture_m2() -> dict:
    """Capture SigmaDataBuffer._estimator_per_t output."""
    torch.manual_seed(SEED_M2)
    n = 3   # number of distinct timesteps
    per_t = 4   # samples per timestep
    N = n * per_t
    d = 4   # feature dim

    idx = torch.tensor([1, 2, 3], dtype=torch.long)  # 1-based t indices
    mu_hat_batch = torch.randn(N, d)
    sigma_t2_batch = torch.exp(torch.randn(N, d) * 0.3)

    result = SigmaDataBuffer._estimator_per_t(
        idx=idx,
        mu_hat_batch=mu_hat_batch,
        sigma_t2_batch=sigma_t2_batch,
    )

    golden_estimator = result.tolist()
    print(f"[M2] _estimator_per_t output (n=3 timesteps): {golden_estimator}")

    return {
        "estimator_per_t": golden_estimator,
        "seed": SEED_M2,
        "n": n,
        "per_t": per_t,
        "d": d,
    }


# ---------------------------------------------------------------------------
# M8 golden — 5-step training checkpoint from make_vhp_model
# ---------------------------------------------------------------------------

SEED_M8 = 0

# Import the integration test conftest's make_vhp_model
sys.path.insert(0, str(repo_root / "tests" / "test_integration"))
from conftest import (  # noqa: E402  (after sys.path mutation)
    make_vhp_model,
    make_random_walk_data,
    run_stage,
    DATA_DIM,
    LATENT_DIM,
    J as CONF_J,
    EMB_TIME as CONF_EMB_TIME,
    T_MAX as CONF_T_MAX,
)


def capture_m8(ckpt_path: Path) -> dict:
    """5-step train in stage_2, use_split_loss=False, save state_dict."""
    torch.manual_seed(SEED_M8)
    np.random.seed(SEED_M8)

    model = make_vhp_model(
        baseline_form="persistence",
        tracking_mode="fixed",
        sigma_data_init=1.0,
    )
    model.train()

    data_factory = lambda: make_random_walk_data(n_seqs=4, T=8, seed=SEED_M8)  # noqa: E731

    torch.manual_seed(SEED_M8)
    metrics_log = run_stage(
        model=model,
        data_factory=data_factory,
        n_steps=5,
        lr=1e-3,
    )

    # Capture final state_dict
    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state_dict,
            "seed": SEED_M8,
            "n_steps": 5,
            "config": {
                "baseline_form": "persistence",
                "tracking_mode": "fixed",
                "sigma_data_init": 1.0,
                "data_n_seqs": 4,
                "data_T": 8,
                "lr": 1e-3,
            },
            "final_metrics": {k: float(v) for k, v in metrics_log[-1].items()},
        },
        ckpt_path,
    )

    # Also capture a small summary of the final loss
    final_loss = metrics_log[-1].get("loss/total", None)
    if final_loss is None:
        final_loss = metrics_log[-1].get("loss/rate/trans/kl", None)

    print(f"[M8] 5-step checkpoint saved to {ckpt_path}")
    print(f"[M8] Final step metrics keys: {list(metrics_log[-1].keys())}")
    print(f"[M8] Final loss/total (or trans/kl): {float(final_loss) if final_loss is not None else 'N/A'}")

    # Return a small summary of final param norms so golden_values.py can include them
    param_norms = {}
    for name, p in model.named_parameters():
        param_norms[name] = float(p.detach().norm())
    # Just keep a few representative ones
    rep_keys = list(param_norms.keys())[:5]
    rep_norms = {k: param_norms[k] for k in rep_keys}

    return {
        "seed": SEED_M8,
        "n_steps": 5,
        "final_metrics": {k: float(v) for k, v in metrics_log[-1].items()},
        "representative_param_norms": rep_norms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    fixtures_dir = repo_root / "tests" / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Capturing M1 golden (_esm_chunk_loss)...")
    print("=" * 60)
    m1 = capture_m1()

    print()
    print("=" * 60)
    print("Capturing M2 golden (_estimator_per_t)...")
    print("=" * 60)
    m2 = capture_m2()

    print()
    print("=" * 60)
    print("Capturing M8 golden (5-step checkpoint)...")
    print("=" * 60)
    ckpt_path = fixtures_dir / "m8_5step_ckpt.pt"
    m8 = capture_m8(ckpt_path)

    # Verify all goldens are finite
    assert all(
        torch.isfinite(torch.tensor(v)) for v in [m1["esm_chunk_loss_scalar"]]
    ), "M1 scalar is not finite!"
    assert all(
        torch.isfinite(torch.tensor(x)) for x in m1["esm_chunk_loss_per_sample"]
    ), "M1 per_sample is not finite!"
    assert all(
        torch.isfinite(torch.tensor(x)) for x in m2["estimator_per_t"]
    ), "M2 estimator is not finite!"

    # Write the Python fixture module
    fixture_path = fixtures_dir / "golden_values.py"

    lines = [
        '"""Golden values for split-loss TDD (M1/M2/M8).',
        "",
        "Captured at commit 1d31804 on branch split-loss-training",
        "(pre-refactor state — these are the pre-change ground truth values).",
        "",
        "DO NOT EDIT MANUALLY — regenerate via:",
        "    LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:$LD_LIBRARY_PATH \\",
        "    TORCHDYNAMO_DISABLE=1 uv run python tools/capture_goldens.py",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from functools import partial",
        "from pathlib import Path",
        "from types import SimpleNamespace",
        "",
        "import torch",
        "import numpy as np",
        "",
        "# ---------------------------------------------------------------------------",
        "# M1 — _esm_chunk_loss golden",
        "# ---------------------------------------------------------------------------",
        "# Reproducer: build a tiny DiffusionTransition under SEED_M1=42,",
        "# call _esm_chunk_loss on fixed inputs, capture the summed scalar.",
        "#",
        "# Key: M1 test_esm_chunk_loss_phith_reproduces_prior_single_loss checks",
        "# that after refactor, `sum_phith / S_k` == this golden (bit-level).",
        "# Under unit sigma_data and uniform p_k, weighted sqerr == sqerr * weight,",
        "# and the pre-refactor single accumulator IS the phith accumulator.",
        "",
        f"M1_SEED: int = {m1['seed']}",
        f"M1_N: int = {m1['N']}",
        f"M1_D: int = {m1['d']}",
        f"M1_ESM_CHUNK_LOSS_SCALAR: float = {m1['esm_chunk_loss_scalar']!r}",
        f"M1_ESM_CHUNK_LOSS_PER_SAMPLE: list[float] = {m1['esm_chunk_loss_per_sample']!r}",
        "",
        "# M1 model construction parameters (mirrors _make_diffusion_m1 in capture_goldens.py)",
        "M1_B: int = 2",
        "M1_S: int = 2",
        "M1_J: int = 1",
        "M1_EMB_TIME: int = 8",
        "M1_T_MAX: int = 10",
        "M1_CHANNELS: int = 16",
        "M1_NHEADS: int = 2",
        "M1_S_K: int = 2        # MC samples; >1 so weights != 1 (phith vs psi differ)",
        "",
        "",
        "def make_m1_transition():",
        '    """Build the fixed DiffusionTransition used for M1 golden capture."""',
        "    from ddssm.nn.diffnets import CSDIUnet, FeatureMixerConfig, DiffResidualBlockConfig",
        "    from ddssm.model.centering.baselines import ZeroBaseline",
        "    from ddssm.model.transitions.diffusion import DiffusionTransition, DiffusionScheduleConfig",
        "",
        "    torch.manual_seed(M1_SEED)",
        "    np.random.seed(M1_SEED)",
        "    baseline = ZeroBaseline(latent_dim=M1_D, j=M1_J)",
        "    schedule = DiffusionScheduleConfig(",
        "        S_k=M1_S_K,",
        "        k_chunk=M1_S_K,",
        "        num_steps=20,",
        "        beta_min=0.1,",
        "        beta_max=20.0,",
        "        tau_min=1e-3,",
        "        k_sampling_mode='uniform',",
        "    )",
        "    unet = partial(",
        "        CSDIUnet,",
        "        channels=M1_CHANNELS,",
        "        n_layers=1,",
        "        embedding_dim=M1_CHANNELS,",
        "        residual_block=DiffResidualBlockConfig(",
        "            feature=FeatureMixerConfig(nheads=M1_NHEADS, n_layers=1)",
        "        ),",
        "    )",
        "    transition = DiffusionTransition(",
        "        baseline=baseline,",
        "        latent_dim=M1_D,",
        "        j=M1_J,",
        "        emb_time_dim=M1_EMB_TIME,",
        "        T_max=M1_T_MAX,",
        "        unet=unet,",
        "        schedule=schedule,",
        "    )",
        "    transition.eval()",
        "    return transition",
        "",
        "",
        "def make_m1_inputs():",
        '    """Build the fixed inputs used for M1 golden capture."""',
        "    torch.manual_seed(M1_SEED + 1)",
        "    N = M1_N",
        "    d = M1_D",
        "    J = M1_J",
        "    EMB_TIME = M1_EMB_TIME",
        "    mu_t = torch.randn(N, d)",
        "    sigma2_t = torch.exp(torch.randn(N, d) * 0.5 - 1.0).clamp(1e-4, 2.0)",
        "    z_hist = torch.randn(N, d, J)",
        "    sigma_d2_per_row = torch.ones(N)",
        "    padding_mask = torch.zeros(N, J + 1)",
        "    time_win = torch.randn(N, J + 1, EMB_TIME)",
        "    ctx = {",
        "        'hist_time_emb': time_win[:, :J, :],",
        "        'target_time_emb': time_win[:, J:, :],",
        "    }",
        "    return dict(",
        "        mu_t=mu_t, sigma2_t=sigma2_t, z_hist=z_hist,",
        "        ctx=ctx, sigma_d2_per_row=sigma_d2_per_row, padding_mask=padding_mask,",
        "    )",
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# M2 — _estimator_per_t golden",
        "# ---------------------------------------------------------------------------",
        "# Reproducer: call SigmaDataBuffer._estimator_per_t directly on fixed tensors.",
        "# M2 test_estimator_from_suff_stats_matches_original_estimator checks that",
        "# after refactor, _estimator_from_suff_stats(suff_stats) == this golden (bit-level).",
        "",
        f"M2_SEED: int = {m2['seed']}",
        f"M2_N_T: int = {m2['n']}          # distinct timesteps",
        f"M2_PER_T: int = {m2['per_t']}        # samples per timestep",
        f"M2_D: int = {m2['d']}          # feature dim",
        f"M2_ESTIMATOR_PER_T: list[float] = {m2['estimator_per_t']!r}",
        "",
        "",
        "def make_m2_inputs():",
        '    """Build the fixed inputs used for M2 golden capture."""',
        "    torch.manual_seed(M2_SEED)",
        "    n = M2_N_T",
        "    per_t = M2_PER_T",
        "    N = n * per_t",
        "    d = M2_D",
        "    idx = torch.tensor([1, 2, 3], dtype=torch.long)",
        "    mu_hat_batch = torch.randn(N, d)",
        "    sigma_t2_batch = torch.exp(torch.randn(N, d) * 0.3)",
        "    return dict(idx=idx, mu_hat_batch=mu_hat_batch, sigma_t2_batch=sigma_t2_batch)",
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# M8 — 5-step training checkpoint",
        "# ---------------------------------------------------------------------------",
        "# Reproducer: make_vhp_model() from tests/test_integration/conftest.py +",
        "# run_stage(stage='stage_2', n_steps=5) under SEED_M8=0.",
        "# use_split_loss=False (the only mode pre-refactor).",
        "# M8 test_single_loss_off_path_regresses_none loads this checkpoint and",
        "# checks that 5-step training produces bit-identical weights.",
        "",
        f"M8_SEED: int = {m8['seed']}",
        f"M8_N_STEPS: int = {m8['n_steps']}",
        "M8_CONFIG: dict = {",
        '    "baseline_form": "persistence",',
        '    "tracking_mode": "fixed",',
        '    "sigma_data_init": 1.0,',
        '    "data_n_seqs": 4,',
        '    "data_T": 8,',
        '    "lr": 1e-3,',
        "}",
        "M8_CKPT_PATH: str = str(Path(__file__).parent / 'm8_5step_ckpt.pt')",
        "",
        f"# Final metrics from step 5 (for sanity checking):",
        f"M8_FINAL_METRICS: dict = {m8['final_metrics']!r}",
        "",
        "",
        "def load_m8_checkpoint() -> dict:",
        '    """Load the saved M8 5-step checkpoint."""',
        "    return torch.load(M8_CKPT_PATH, weights_only=False)",
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# Sanity assertion (run on import in debug mode)",
        "# ---------------------------------------------------------------------------",
        "",
        "def _check_finite() -> None:",
        "    assert torch.isfinite(torch.tensor(M1_ESM_CHUNK_LOSS_SCALAR)), 'M1 scalar non-finite'",
        "    assert all(torch.isfinite(torch.tensor(x)) for x in M1_ESM_CHUNK_LOSS_PER_SAMPLE), 'M1 per_sample non-finite'",
        "    assert all(torch.isfinite(torch.tensor(x)) for x in M2_ESTIMATOR_PER_T), 'M2 estimator non-finite'",
        "",
        "",
        "_check_finite()",
    ]

    fixture_path.write_text("\n".join(lines) + "\n")
    print()
    print(f"Wrote {fixture_path}")
    print(f"Wrote {ckpt_path}")
    print()
    print("Summary:")
    print(f"  M1_ESM_CHUNK_LOSS_SCALAR = {m1['esm_chunk_loss_scalar']!r}")
    print(f"  M1_ESM_CHUNK_LOSS_PER_SAMPLE = {m1['esm_chunk_loss_per_sample']!r}")
    print(f"  M2_ESTIMATOR_PER_T = {m2['estimator_per_t']!r}")
    print(f"  M8 final metrics = {m8['final_metrics']}")


if __name__ == "__main__":
    main()
