"""Eval + viz specs for the KDD experiments."""

from __future__ import annotations

from ddssm.builders import Eval, Plot, Viz

from conf.registry import eval_store, viz_store


SPLIT = "test"
NUM_SAMPLES = 32


KDDEval = Eval(metrics=["mae", "crps_sum"], split=SPLIT, num_samples=NUM_SAMPLES)


KDDViz = Viz(
    plots=[
        Plot(name="forecast_1d", save_filename="forecast.png",
             kwargs={"n_show": 4}),
        Plot(name="metrics_csv", save_filename="train_loss.png",
             kwargs={"keys": ["loss/total"], "log_y": True}),
    ],
    split=SPLIT, num_samples=NUM_SAMPLES,
)


eval_store(KDDEval, name="kdd")
viz_store(KDDViz, name="kdd")
