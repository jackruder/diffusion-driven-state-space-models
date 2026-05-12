"""KDD viz spec (forecast_1d on the test split)."""

from __future__ import annotations

from ddssm.builders import Plot, Viz

from conf.registry import viz_store


KDD = Viz(
    plots=[
        Plot(name="forecast_1d", save_filename="forecast.png",
             kwargs={"n_show": 4}),
        Plot(name="metrics_csv", save_filename="train_loss.png",
             kwargs={"keys": ["loss/total"], "log_y": True}),
    ],
    split="test", num_samples=32,
)
viz_store(KDD, name="kdd")
