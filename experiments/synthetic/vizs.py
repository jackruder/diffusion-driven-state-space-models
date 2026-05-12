"""Viz specs for the synthetic-data experiments."""

from __future__ import annotations

from ddssm.builders import Plot, Viz

from conf.registry import viz_store


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


Forecast1D = Viz(
    plots=[_FORECAST_1D, _LOSS_CSV],
    split="val", num_samples=32, T_split=16,
)

BimodalForecast1D = Viz(
    plots=[_FORECAST_1D, _LOSS_CSV],
    split="val", num_samples=64, T_split=16,
)

Robot2DForecast = Viz(
    plots=[_FORECAST_2D, _LOSS_CSV],
    split="val", num_samples=32, T_split=16,
)

LGSSM = Viz(
    plots=[_LOSS_CSV],
    split="val", num_samples=10, T_split=16,
)

viz_store(Forecast1D, name="forecast_1d")
viz_store(BimodalForecast1D, name="bimodal_forecast_1d")
viz_store(Robot2DForecast, name="robot2d")
viz_store(LGSSM, name="lgssm")
