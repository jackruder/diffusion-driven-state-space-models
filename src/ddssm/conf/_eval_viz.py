"""Eval and viz family defaults for each dataset group.

These shared ``EvalSpec`` / ``VizSpec`` configs are imported by every
experiment module so that adding a new experiment in a family requires
only a one-liner that points at the right family default — no need to
re-specify metrics or plot lists.
"""

from __future__ import annotations

from ._infra import VizSpecConf, EvalSpecConf, PlotSpecConf

# ---------------------------------------------------------------------------
# Eval / viz defaults for component-test and simple synthetic experiments.
# ---------------------------------------------------------------------------

# Synthetic smoke-test: recon-only metrics (no forecasting split required).
SynthEvalConf = EvalSpecConf(metrics=["loss_tail", "recon_mse"], split="val")
SynthVizConf = VizSpecConf(
    plots=[
        PlotSpecConf(
            name="metrics_csv",
            save_filename="train_loss.png",
            kwargs={"keys": ["loss/total"], "log_y": True},
        ),
    ],
    split="val",
    num_samples=10,
    T_split=32,  # half of synthetic's default T=64; override per-experiment if needed
)

# ---------------------------------------------------------------------------
# Eval / viz defaults for KDD Cup 2018.
# ---------------------------------------------------------------------------

KDDEvalConf = EvalSpecConf(metrics=["mae", "crps_sum"], split="test", num_samples=32)
KDDVizConf = VizSpecConf(
    plots=[
        PlotSpecConf(
            name="forecast_1d", save_filename="forecast.png", kwargs={"n_show": 4}
        ),
        PlotSpecConf(
            name="metrics_csv",
            save_filename="train_loss.png",
            kwargs={"keys": ["loss/total"], "log_y": True},
        ),
    ],
    split="test",
    num_samples=32,
    # T_split picks up data.metadata.forecast_split == L1 automatically.
)

# ---------------------------------------------------------------------------
# Eval / viz defaults for synthetic verification experiments.
# ---------------------------------------------------------------------------

HarmonicEvalConf = EvalSpecConf(
    metrics=["mae", "crps_sum"], split="val", num_samples=32, T_split=32
)
HarmonicVizConf = VizSpecConf(
    plots=[
        PlotSpecConf(
            name="forecast_1d", save_filename="forecast.png", kwargs={"n_show": 4}
        ),
        PlotSpecConf(
            name="metrics_csv",
            save_filename="train_loss.png",
            kwargs={"keys": ["loss/total"], "log_y": True},
        ),
    ],
    split="val",
    num_samples=32,
    T_split=32,
)

BimodalEvalConf = EvalSpecConf(
    metrics=["energy_score", "crps_sum"], split="val", num_samples=64, T_split=32
)
BimodalVizConf = VizSpecConf(
    plots=[
        PlotSpecConf(
            name="forecast_1d", save_filename="forecast.png", kwargs={"n_show": 4}
        ),
        PlotSpecConf(
            name="metrics_csv",
            save_filename="train_loss.png",
            kwargs={"keys": ["loss/total"], "log_y": True},
        ),
    ],
    split="val",
    num_samples=64,
    T_split=32,
)

Robot2DEvalConf = EvalSpecConf(
    metrics=["energy_score", "crps_sum"], split="val", num_samples=32, T_split=32
)
Robot2DVizConf = VizSpecConf(
    plots=[
        PlotSpecConf(
            name="forecast_2d_spatial",
            save_filename="forecast_2d.png",
            kwargs={"n_show": 4},
        ),
        PlotSpecConf(
            name="metrics_csv",
            save_filename="train_loss.png",
            kwargs={"keys": ["loss/total"], "log_y": True},
        ),
    ],
    split="val",
    num_samples=32,
    T_split=32,
)
