"""Unit tests for variance-probe metric aggregation.

`loss_var` must report the across-replica variance of the loss ESTIMATOR
(one `L_p_scalar` per replica), not the within-batch per-sample spread of the
pooled `L_p` rows — and must return NaN for a cell with <2 replicas.
"""

from __future__ import annotations

import math

from ddssm.variance.metrics import ProbeContext, metric_loss_var


def _row(*, objective, mode, seed, batch_idx, replica, l_p, l_p_scalar):
    return {
        "kind": "replica",
        "objective": objective,
        "k_sampling_mode": mode,
        "seed": seed,
        "batch_idx": batch_idx,
        "replica": replica,
        "k_idx": -1,
        "sample_idx": 0,
        "L_p": float(l_p),
        "L_p_scalar": float(l_p_scalar),
        "grad_norm": 0.0,
    }


def _ctx(rows):
    return ProbeContext(
        model=None, transitions={}, loader=None, device=None,
        spec=None, run_dir="", rows=rows, summary={},
    )


def test_loss_var_uses_per_replica_scalar_not_per_sample_spread():
    """Two replicas (per-sample L_p deliberately noisy); variance is over the
    per-replica L_p_scalar means, not the pooled per-sample values."""
    rows = []
    # esm:uniform — replica 0 (L_p_scalar=1.0) and replica 1 (L_p_scalar=3.0);
    # per-sample L_p is wildly different so a per-sample pool would be huge.
    for s, lp in zip(range(3), (0.0, 1.0, 2.0)):
        rows.append(_row(objective="esm", mode="uniform", seed=0, batch_idx=0,
                         replica=0, l_p=lp, l_p_scalar=1.0))
        rows[-1]["sample_idx"] = s
    for s, lp in zip(range(3), (10.0, 11.0, 12.0)):
        rows.append(_row(objective="esm", mode="uniform", seed=0, batch_idx=0,
                         replica=1, l_p=lp, l_p_scalar=3.0))
        rows[-1]["sample_idx"] = s

    out = metric_loss_var(_ctx(rows))["loss_var"]
    # population var of {1.0, 3.0} = 1.0; a per-sample pool would give ~27.
    assert abs(out["esm:uniform"] - 1.0) < 1e-9


def test_loss_var_nan_for_single_replica():
    """A cell with one replica yields NaN, not a misleading 0.0."""
    rows = [
        _row(objective="dsm", mode="uniform", seed=0, batch_idx=0, replica=0,
             l_p=5.0, l_p_scalar=5.0),
        _row(objective="dsm", mode="uniform", seed=0, batch_idx=0, replica=0,
             l_p=6.0, l_p_scalar=5.0),
    ]
    out = metric_loss_var(_ctx(rows))["loss_var"]
    assert math.isnan(out["dsm:uniform"])
