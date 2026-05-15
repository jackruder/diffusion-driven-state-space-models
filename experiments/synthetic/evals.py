"""Eval + viz specs for the synthetic-data family.

Pairs each named ``Eval`` with a same-name ``Viz`` (so e.g.
``Forecast1D`` denotes both the metric set and the matching plot set).
Tweak shared options like ``T_split`` or ``num_samples`` at the top.
"""

from __future__ import annotations

from ddssm.builders import Eval, Plot, Viz

from conf.registry import eval_store, viz_store


# ---------------------------------------------------------------------------
# Shared knobs.
# ---------------------------------------------------------------------------

SPLIT = "val"
T_SPLIT = 16


# ---------------------------------------------------------------------------
# Plot building blocks (shared by multiple Viz specs).
# ---------------------------------------------------------------------------

_FORECAST_1D = Plot(
    name="forecast_1d", save_filename="forecast.png",
    kwargs={"n_show": 4},
)
_FORECAST_2D = Plot(
    name="forecast_2d_spatial", save_filename="forecast_2d.png",
    kwargs={"n_show": 4},
)
_LOSS_CSV = Plot(
    name="metrics_csv", save_filename="train_loss.png",
    kwargs={"keys": ["loss/total"], "log_y": True},
)


# ---------------------------------------------------------------------------
# Evals.
# ---------------------------------------------------------------------------

# 1D forecast (harmonic, robot 1D).
Forecast1D_Eval = Eval(
    metrics=["mae", "crps_sum"], split=SPLIT,
    num_samples=32, T_split=T_SPLIT,
)
# Energy-score bimodal (S=4).
BimodalEnergy_Eval = Eval(
    metrics=["energy_score", "crps_sum"], split=SPLIT,
    num_samples=64, T_split=T_SPLIT,
)
# 2D robot navigation.
Robot2D_Eval = Eval(
    metrics=["energy_score", "crps_sum"], split=SPLIT,
    num_samples=32, T_split=T_SPLIT,
)
# LGSSM smoke (no forecast — just recon).
LGSSM_Eval = Eval(metrics=["loss_tail", "recon_mse"], split=SPLIT)


# ---------------------------------------------------------------------------
# Viz specs.
# ---------------------------------------------------------------------------

Forecast1D_Viz = Viz(
    plots=[_FORECAST_1D, _LOSS_CSV],
    split=SPLIT, num_samples=32, T_split=T_SPLIT,
)
BimodalForecast1D_Viz = Viz(
    plots=[_FORECAST_1D, _LOSS_CSV],
    split=SPLIT, num_samples=64, T_split=T_SPLIT,
)
Robot2D_Viz = Viz(
    plots=[_FORECAST_2D, _LOSS_CSV],
    split=SPLIT, num_samples=32, T_split=T_SPLIT,
)
LGSSM_Viz = Viz(
    plots=[_LOSS_CSV],
    split=SPLIT, num_samples=10, T_split=T_SPLIT,
)


eval_store(Forecast1D_Eval, name="forecast_1d")
eval_store(BimodalEnergy_Eval, name="bimodal_energy")
eval_store(Robot2D_Eval, name="robot2d")
eval_store(LGSSM_Eval, name="lgssm")

viz_store(Forecast1D_Viz, name="forecast_1d")
viz_store(BimodalForecast1D_Viz, name="bimodal_forecast_1d")
viz_store(Robot2D_Viz, name="robot2d")
viz_store(LGSSM_Viz, name="lgssm")
