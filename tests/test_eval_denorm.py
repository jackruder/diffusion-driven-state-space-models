"""De-normalization of obs-space forecast metrics (CSDI-scale comparability).

When the data module z-scores per series, the eval must de-normalize forecasts
back to the original scale before CRPS-sum/MAE/RMSE (CSDI's calc_quantile_CRPS
does the same). With ``means=0, stds=s`` the metric should scale by ``s``;
``means/stds=None`` (synthetic) must be a no-op.
"""

from __future__ import annotations

import torch

from ddssm.eval.metrics import EvalContext, eval_mae


class _StubForecastModel(torch.nn.Module):
    """Returns an all-zeros (normalized-space) forecast of the right shape."""

    def forecast(self, *, x_hist, future_time, num_samples, **_kw):
        B, D = x_hist.shape[0], x_hist.shape[1]
        L2 = future_time.shape[1]
        z = torch.zeros(B, D, L2)
        return {
            "pred_samples": z.unsqueeze(1).expand(B, num_samples, D, L2).contiguous(),
            "pred_mean": z,
        }


def test_denorm_scales_metric_by_std_and_noop_when_absent() -> None:
    torch.manual_seed(0)
    D, L1, L2 = 3, 4, 2
    obs = torch.randn(2, D, L1 + L2)
    batch = {
        "observed_data": obs,
        "observation_mask": torch.ones_like(obs),
        "timepoints": torch.arange(L1 + L2).float().unsqueeze(0).expand(2, L1 + L2).contiguous(),
    }

    def ctx(means, stds):
        return EvalContext(
            model=_StubForecastModel(),
            loader=[batch],
            device=torch.device("cpu"),
            batch_transform=None,
            T_split=L1,
            num_samples=4,
            means=means,
            stds=stds,
        )

    mae_norm = eval_mae(ctx(None, None))["mae"]
    # pred=0 (normalized) → de-norm pred = mean = 0; y_denorm = y_norm * 2.
    # So MAE on the de-normalized scale = 2 × the normalized-scale MAE.
    mae_denorm = eval_mae(ctx(torch.zeros(D), torch.full((D,), 2.0)))["mae"]
    assert abs(mae_denorm - 2.0 * mae_norm) < 1e-5
