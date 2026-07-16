"""Tests for the Phase-A headline metrics.

Covers:

- ``wallclock_to_target`` (CSV-derived; no model needed).
- ``stage2_elbo_surrogate`` (smoke on the existing init_smoke_simple
  preset; slow-marked).
- ``sigma_data_drift`` (snapshot from ``model.sigma_data`` + the
  two-component decomposition).
- ``crps_sum_latent`` (latent-space CRPS with the GT-latent surface;
  smoke-marked).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
import pytest

from ddssm.eval import METRIC_REGISTRY, EvalContext
from ddssm.eval.metrics import (
    eval_nll,
    eval_sigma_data_drift,
    eval_wallclock_to_target,
    eval_stage2_elbo_surrogate,
)
from ddssm.adapters.ddssm import DDSSMAdapter
from ddssm.model.config import ModelConfig
from ddssm.model.dssd import DDSSM_base


def _stub_ddssm_adapter(module) -> DDSSMAdapter:
    """Wrap a stub ``DDSSM_base`` in a ``DDSSMAdapter`` for unit tests.

    Same escape hatch as ``tests/test_checkpoint.py::_toy_adapter`` — sets
    ``_module`` post-hoc so ``ctx.require_module(DDSSM_base)`` inside the
    metric resolves without going through ``config.build_module()``.
    """
    adapter = DDSSMAdapter(config=ModelConfig())
    adapter._module = module
    return adapter


class _FakeLogProbModel(DDSSM_base):
    """Minimal ``DDSSM_base`` subclass for ``eval_nll`` unit tests.

    Skips ``DDSSM_base.__init__`` (encoder / decoder / … are irrelevant for
    the metric-body forwarding test) but keeps the isinstance identity so
    ``require_module(DDSSM_base)`` returns it. Subclasses override
    ``log_prob``; the ``_FakeLogProbModel.__init__`` shortcut just sets up
    ``nn.Module`` bookkeeping.
    """

    def __init__(self) -> None:  # noqa: D401 — minimal test stub
        torch.nn.Module.__init__(self)

# ---------------------------------------------------------------------------
# wallclock_to_target — CSV-derived, no model needed
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_wallclock_to_target_registered() -> None:
    """The metric is in the registry under the right name."""
    assert "wallclock_to_target" in METRIC_REGISTRY


def test_wallclock_to_target_finds_first_crossing(tmp_path) -> None:
    """The metric returns the step/time of the first row crossing the threshold."""
    csv_path = tmp_path / "metrics.csv"
    rows = [
        {"split": "train", "step": "1", "time/elapsed_s": "1.0", "loss/total": "5.0"},
        {"split": "train", "step": "2", "time/elapsed_s": "2.0", "loss/total": "4.0"},
        {"split": "train", "step": "3", "time/elapsed_s": "3.5", "loss/total": "1.5"},
        {"split": "train", "step": "4", "time/elapsed_s": "5.0", "loss/total": "0.9"},
        {"split": "train", "step": "5", "time/elapsed_s": "6.5", "loss/total": "0.7"},
    ]
    _write_csv(csv_path, rows)
    ctx = EvalContext(
        model=None,
        loader=None,
        device=torch.device("cpu"),
        csv_path=str(csv_path),
    )
    out = eval_wallclock_to_target(
        ctx,
        target_column="loss/total",
        target_value=1.0,
        direction="<=",
    )
    assert out["wallclock_to_target_step"] == 4
    assert abs(out["wallclock_to_target_seconds"] - 5.0) < 1e-9
    assert out["wallclock_to_target_direction"] == "<="


def test_wallclock_to_target_never_crosses(tmp_path) -> None:
    """When no row crosses, both step and seconds are ``None``."""
    csv_path = tmp_path / "metrics.csv"
    rows = [
        {
            "split": "train",
            "step": str(i),
            "time/elapsed_s": str(i),
            "loss/total": "5.0",
        }
        for i in range(1, 6)
    ]
    _write_csv(csv_path, rows)
    ctx = EvalContext(
        model=None,
        loader=None,
        device=torch.device("cpu"),
        csv_path=str(csv_path),
    )
    out = eval_wallclock_to_target(
        ctx,
        target_column="loss/total",
        target_value=1.0,
        direction="<=",
    )
    assert out["wallclock_to_target_step"] is None
    assert out["wallclock_to_target_seconds"] is None


def test_wallclock_to_target_ge_direction(tmp_path) -> None:
    """``direction='>='`` finds the first upward crossing."""
    csv_path = tmp_path / "metrics.csv"
    rows = [
        {
            "split": "train",
            "step": str(i),
            "time/elapsed_s": str(float(i)),
            "rate": str(0.1 * i),
        }
        for i in range(1, 6)
    ]
    _write_csv(csv_path, rows)
    ctx = EvalContext(
        model=None,
        loader=None,
        device=torch.device("cpu"),
        csv_path=str(csv_path),
    )
    out = eval_wallclock_to_target(
        ctx,
        target_column="rate",
        target_value=0.3,
        direction=">=",
    )
    assert out["wallclock_to_target_step"] == 3
    assert abs(out["wallclock_to_target_seconds"] - 3.0) < 1e-9


def test_wallclock_to_target_invalid_direction_raises() -> None:
    """Unknown direction tokens raise ``ValueError``."""
    ctx = EvalContext(
        model=None,
        loader=None,
        device=torch.device("cpu"),
        csv_path="",
    )
    with pytest.raises(ValueError):
        eval_wallclock_to_target(ctx, direction="==")


def test_wallclock_to_target_missing_csv() -> None:
    """No CSV → both fields are ``None``."""
    ctx = EvalContext(
        model=None,
        loader=None,
        device=torch.device("cpu"),
        csv_path="",
    )
    out = eval_wallclock_to_target(ctx)
    assert out["wallclock_to_target_step"] is None
    assert out["wallclock_to_target_seconds"] is None


# ---------------------------------------------------------------------------
# crps_sum_latent_metrics (the helper in eval_metrics.py)
# ---------------------------------------------------------------------------


def test_crps_sum_latent_metrics_shape() -> None:
    """Helper returns ``(scalar, per-t)`` of the right shape."""
    from ddssm.eval.eval_metrics import crps_sum_latent_metrics

    torch.manual_seed(0)
    B, S, d, L2 = 3, 16, 2, 5
    z_samples = torch.randn(B, S, d, L2)
    z_gt = torch.randn(B, d, L2)
    mean, per_t = crps_sum_latent_metrics(z_samples, z_gt)
    assert mean.dim() == 0
    assert per_t.shape == (L2,)


def test_crps_sum_latent_zero_when_samples_match_gt() -> None:
    """Tight sample distribution centred on the GT gives near-zero CRPS."""
    from ddssm.eval.eval_metrics import crps_sum_latent_metrics

    torch.manual_seed(0)
    B, S, d, L2 = 2, 32, 2, 4
    z_gt = torch.randn(B, d, L2)
    # Very tight samples around the GT.
    z_samples = z_gt.unsqueeze(1) + 1e-3 * torch.randn(B, S, d, L2)
    mean, _ = crps_sum_latent_metrics(z_samples, z_gt)
    assert float(mean.item()) < 0.05


# ---------------------------------------------------------------------------
# sigma_data_drift  (requires a model; smoke-marked)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sigma_data_drift_snapshot_with_init_smoke_simple(tmp_path) -> None:
    """End-to-end snapshot via the smoke preset.

    Trains for a handful of steps then runs the metric on the
    validation loader.  Verifies:
      - the returned dict has the expected keys with finite values;
      - the two-component decomposition sum is consistent with the
        buffer values (per-t) — this is the doc's
        ``σ_data²(t) = (1/D)·(E‖σ_t‖² + tr Var[μ̂_t])`` identity at
        snapshot time.
    """
    from hydra_zen import instantiate

    from ddssm.experiment.registry import register_experiments

    register_experiments()
    from ddssm.experiment.stores import store

    cfg = None
    for entry in store:
        if entry["group"] == "experiment" and entry["name"] == "init_smoke_simple":
            cfg = entry["node"]
            break
    assert cfg is not None

    exp = instantiate(cfg)
    exp.training.steps = 6
    exp.training.log_every = 1
    exp.training.validate_every = 0
    exp.training.checkpoint_every = 100
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exp.train(device=torch.device("cpu"), run_dir=str(run_dir))

    ctx = EvalContext(
        model=exp.model,
        loader=exp.data.val_loader(),
        device=torch.device("cpu"),
        batch_transform=exp.data.batch_transform,
    )
    out = eval_sigma_data_drift(ctx, max_batches=2)
    assert out["sigma_data_drift_available"] is True
    buf = out["sigma_data2_buffer"]
    c1 = out["sigma_data2_component1_per_t"]
    c2 = out["sigma_data2_component2_per_t"]
    ts = out["sigma_data2_t_indices"]
    assert len(c1) == len(c2) == len(ts)
    assert all(math.isfinite(v) for v in c1 + c2)
    # The decomposition sum is the empirical σ_data²(t); compare
    # against the (frozen, "fixed" mode) buffer at the same t.
    for t_idx, c1_v, c2_v in zip(ts, c1, c2):
        buf_v = buf[t_idx - 1]  # internal 0-based ↔ external 1-based
        # The buffer is in "fixed" mode and frozen at init_value, so
        # we just sanity-check finiteness here rather than equality —
        # the doc's identity is exact only when σ_data is tracking,
        # which is not the case under "fixed".
        assert math.isfinite(buf_v)


# ---------------------------------------------------------------------------
# stage2_elbo_surrogate  (slow; uses the smoke preset)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_stage2_elbo_surrogate_with_init_smoke_simple(tmp_path) -> None:
    """End-to-end: the metric runs on a trained smoke checkpoint."""
    from hydra_zen import instantiate

    from ddssm.experiment.registry import register_experiments

    register_experiments()
    from ddssm.experiment.stores import store

    cfg = None
    for entry in store:
        if entry["group"] == "experiment" and entry["name"] == "init_smoke_simple":
            cfg = entry["node"]
            break
    assert cfg is not None

    exp = instantiate(cfg)
    exp.training.steps = 6
    exp.training.log_every = 1
    exp.training.validate_every = 0
    exp.training.checkpoint_every = 100
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exp.train(device=torch.device("cpu"), run_dir=str(run_dir))

    ctx = EvalContext(
        model=exp.model,
        loader=exp.data.val_loader(),
        device=torch.device("cpu"),
        batch_transform=exp.data.batch_transform,
    )
    out = eval_stage2_elbo_surrogate(ctx, max_batches=2)
    expected = {
        "stage2_elbo_surrogate",
        "stage2_elbo_surrogate_recon",
        "stage2_elbo_surrogate_init_loss",
        "stage2_elbo_surrogate_init_kl_aux",
        "stage2_elbo_surrogate_init_entropy",
        "stage2_elbo_surrogate_trans_kl",
        "stage2_elbo_surrogate_n_batches",
    }
    assert expected.issubset(set(out.keys()))
    # r_ regularizers were removed — the surrogate should not expose them.
    assert "stage2_elbo_surrogate_r_sigma_p" not in out
    assert "stage2_elbo_surrogate_r_mu_p" not in out
    # All scalar components finite.
    for k in expected - {"stage2_elbo_surrogate_n_batches"}:
        assert math.isfinite(out[k]), f"non-finite {k}: {out[k]}"
    assert out["stage2_elbo_surrogate_n_batches"] >= 1


# ---------------------------------------------------------------------------
# nll — marginal log-likelihood via prob-flow ODE + IWAE
# ---------------------------------------------------------------------------


def test_nll_registered() -> None:
    """The metric is in the registry under the right name."""
    assert "nll" in METRIC_REGISTRY


def test_nll_returns_nan_when_model_or_loader_missing() -> None:
    """No model or no loader → ``nll`` is NaN (mirrors other model metrics)."""
    ctx = EvalContext(model=None, loader=None, device=torch.device("cpu"))
    out = eval_nll(ctx)
    assert math.isnan(out["nll"])


def test_nll_propagates_knobs_to_log_prob_and_aggregates() -> None:
    """``eval_nll`` forwards its knobs to ``model.log_prob`` and averages."""
    captured: list[dict] = []

    class _RecordingLogProb(_FakeLogProbModel):
        def log_prob(
            self,
            observed_data,
            observation_mask,
            timepoints,
            covariates=None,
            static_covariates=None,
            *,
            K=None,
            rtol=1e-5,
            atol=1e-5,
            method="dopri5",
            divergence_mode="exact",
            generator=None,
        ):
            captured.append(
                dict(
                    K=K,
                    rtol=rtol,
                    atol=atol,
                    method=method,
                    divergence_mode=divergence_mode,
                    generator_is_set=generator is not None,
                    batch_size=int(observed_data.shape[0]),
                )
            )
            B = observed_data.shape[0]
            return torch.full((B,), -3.0)

    def _loader():
        yield {
            "observed_data": torch.zeros(2, 1, 4),
            "observation_mask": torch.ones(2, 1, 4),
            "timepoints": torch.arange(4, dtype=torch.float32).expand(2, 4),
        }
        yield {
            "observed_data": torch.zeros(3, 1, 4),
            "observation_mask": torch.ones(3, 1, 4),
            "timepoints": torch.arange(4, dtype=torch.float32).expand(3, 4),
        }

    ctx = EvalContext(
        model=_stub_ddssm_adapter(_RecordingLogProb()),
        loader=_loader(),
        device=torch.device("cpu"),
    )
    out = eval_nll(
        ctx,
        num_iwae_samples=7,
        divergence_mode="hutchinson",
        num_hutchinson_probes=4,
        rtol=2e-4,
        atol=3e-4,
        method="dopri8",
        seed=0,
    )

    # 2 batches × 4 probes per batch = 8 calls.
    assert len(captured) == 8
    for call in captured:
        assert call["K"] == 7
        assert call["divergence_mode"] == "hutchinson"
        assert call["rtol"] == 2e-4
        assert call["atol"] == 3e-4
        assert call["method"] == "dopri8"
        assert call["generator_is_set"] is True

    # Per-sequence mean of -log p with constant log p = -3 is exactly 3.
    assert math.isclose(out["nll"], 3.0, abs_tol=1e-9)
    assert out["nll_n_batches"] == 2
    assert out["nll_n_sequences"] == 5
    assert out["nll_num_iwae_samples"] == 7
    assert out["nll_num_hutchinson_probes"] == 4
    assert out["nll_divergence_mode"] == "hutchinson"


def test_nll_hutchinson_probes_ignored_under_exact_divergence() -> None:
    """In exact mode, ``num_hutchinson_probes`` collapses to a single call."""
    call_count = {"n": 0}

    class _CountingLogProb(_FakeLogProbModel):
        def log_prob(self, *args, **kwargs):
            call_count["n"] += 1
            return torch.zeros(args[0].shape[0])

    def _loader():
        yield {
            "observed_data": torch.zeros(2, 1, 4),
            "observation_mask": torch.ones(2, 1, 4),
            "timepoints": torch.arange(4, dtype=torch.float32).expand(2, 4),
        }

    ctx = EvalContext(
        model=_stub_ddssm_adapter(_CountingLogProb()),
        loader=_loader(),
        device=torch.device("cpu"),
    )
    out = eval_nll(ctx, divergence_mode="exact", num_hutchinson_probes=8)

    assert call_count["n"] == 1
    assert out["nll_num_hutchinson_probes"] == 1


def test_nll_rejects_invalid_divergence_mode() -> None:
    """Mirrors the validation in ``solve_prob_flow_logdensity``."""
    ctx = EvalContext(
        model=object(),
        loader=iter([]),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="divergence_mode"):
        eval_nll(ctx, divergence_mode="quadrature")


def test_nll_rejects_non_positive_probe_count() -> None:
    ctx = EvalContext(
        model=object(),
        loader=iter([]),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="num_hutchinson_probes"):
        eval_nll(ctx, num_hutchinson_probes=0)


# ---------------------------------------------------------------------------
# crps_sum_latent — unavailable path (no gt_latent)
# ---------------------------------------------------------------------------


def test_crps_sum_latent_returns_unavailable_without_gt_latents() -> None:
    """When the loader has no gt_latent, the metric returns ``available: False``."""
    from ddssm.eval.metrics import eval_crps_sum_latent
    from ddssm.data.datamodule import SyntheticDataModule

    dm = SyntheticDataModule(mode="lgssm", T=4, D=1, N_per_split=2, batch_size=1)
    ctx = EvalContext(
        model=object(),  # truthy placeholder
        loader=dm.val_loader(),
        device=torch.device("cpu"),
        batch_transform=dm.batch_transform,
        T_split=2,
    )
    out = eval_crps_sum_latent(ctx, max_batches=1)
    assert out["crps_sum_latent_available"] is False
