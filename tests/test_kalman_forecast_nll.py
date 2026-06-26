"""The analytic Kalman forecast-NLL reference for the LGSSM cell.

Guards two things at once: (1) the variance recursion / Gaussian-NLL maths, and
(2) that the metric's assumed generative params still match ``data/synthetic.py``'s
``lgssm`` mode. A correctly-specified Gaussian predictive has
``E[-log N(x; μ, V)] = 0.5(1 + log 2πV)`` (the entropy floor), because
``E[(x-μ)^2] = V``. If the assumed (a, q, r) drift from the generator, the
residuals stop matching V and the empirical NLL pulls away from the floor.
"""

from __future__ import annotations

import math

import torch  # preload before numpy on NixOS
import numpy as np
import pytest
from hydra_zen import instantiate

from ddssm.data.presets import LGSSM
from ddssm.eval.metrics import (
    EvalContext,
    eval_kalman_forecast_nll,
    _LGSSM_A,
    _LGSSM_OBS_SIGMA,
    _LGSSM_PROC_SIGMA,
)

_L1 = 24  # history; horizon = T - L1 = 8 (matches the cell's T_split).


def _entropy_floor_per_t(L1: int, T: int, a: float, ps: float, os_: float) -> np.ndarray:
    """Analytic per-step NLL floor 0.5(1+log 2πV_t) from the variance recursion."""
    q, r = ps * ps, os_ * os_
    P_f = 0.0
    for _ in range(1, L1):
        P_pred = a * a * P_f + q
        K = P_pred / (P_pred + r)
        P_f = (1.0 - K) * P_pred
    P_lat = a * a * P_f + q
    floor = []
    for _ in range(T - L1):
        Vx = P_lat + r
        floor.append(0.5 * (1.0 + math.log(2.0 * math.pi * Vx)))
        P_lat = a * a * P_lat + q
    return np.asarray(floor)


def _ctx():
    torch.manual_seed(0)
    dm = instantiate(LGSSM)
    return EvalContext(
        model=None,
        loader=dm.test_loader(),
        device=torch.device("cpu"),
        batch_transform=dm.batch_transform,
        T_split=_L1,
        num_samples=1,
    )


def test_kalman_forecast_nll_shape_and_finiteness() -> None:
    out = eval_kalman_forecast_nll(_ctx())
    per_t = out["kalman_forecast_nll_per_t"]
    assert len(per_t) == 32 - _L1
    assert np.isfinite(out["kalman_forecast_nll"])
    assert np.isfinite(per_t).all()


def test_kalman_nll_matches_entropy_floor() -> None:
    """Empirical NLL ≈ the analytic floor ⇒ data matches the assumed (a, q, r)."""
    out = eval_kalman_forecast_nll(_ctx())
    per_t = np.asarray(out["kalman_forecast_nll_per_t"])
    floor = _entropy_floor_per_t(_L1, 32, _LGSSM_A, _LGSSM_PROC_SIGMA, _LGSSM_OBS_SIGMA)
    # 5σ+ slack: per-step Monte-Carlo std over ~512 samples is ~0.03.
    assert np.allclose(per_t, floor, atol=0.12)


def test_kalman_nll_beats_wrong_dynamics() -> None:
    """The true a=0.9 must score lower (better) than a mis-specified a=0.0."""
    ctx = _ctx()
    true_nll = eval_kalman_forecast_nll(ctx)["kalman_forecast_nll"]
    wrong_nll = eval_kalman_forecast_nll(ctx, a=0.0)["kalman_forecast_nll"]
    assert true_nll < wrong_nll
