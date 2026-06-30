"""Eval and objective specs for the Lorenz sweep family.

Single-objective ``LorenzObjective`` minimises ``stage2_elbo_surrogate``.
Multi-objective ``LorenzMOObjective`` adds ``wallclock_to_target_step``
(steps to first reaching ``loss/total <= LORENZ_WALLCLOCK_TARGET``) so the
Pareto front separates fast-but-shallow configs from slow-but-deep ones.

``LORENZ_WALLCLOCK_TARGET = 120.0`` sits between the observed convergent
bands: lorenz_low_lr landed at 124.0 and lorenz_gaussian_full at 115.4,
so good diffusion configs should cross 120 at genuinely different step
counts while weak configs take the csv_tail_step penalty.

``LorenzEval`` (training-curve metrics only) is what the running sweep's
study cells reference; it must stay frozen until the sweep completes so
per-trial ``metrics.json`` stays homogeneous. ``LorenzForecastEval`` /
``LorenzViz`` extend it with forecast-quality metrics and plots for the
canonical presets and for post-hoc ``python -m ddssm.evaluate`` runs.
"""

from __future__ import annotations

from ddssm.experiment.builders import Plot, Viz, Eval, Objective, Objectives

LORENZ_WALLCLOCK_TARGET: float = 120.0

# Sequence datasets leave metadata.forecast_split None by design; the
# eval/viz spec chooses the past/future boundary explicitly. 32/32 spans
# ~1.5 Lyapunov times of horizon -> ~1 expected lobe switch.
LORENZ_T_SPLIT: int = 32

LorenzObjective = Objective(
    metric="stage2_elbo_surrogate",
    source="json",
)

LorenzMOObjective = Objectives(specs=[
    Objective(
        metric="wallclock_to_target_step",
        source="json",
        penalty="csv_tail_step",
    ),
    Objective(
        metric="stage2_elbo_surrogate",
        source="json",
    ),
])

LorenzEval = Eval(
    metrics=[
        "stage2_elbo_surrogate",
        "sigma_data_drift",
        "wallclock_to_target",
        "wallclock_to_relative_target",
    ],
    split="val",
    num_samples=16,
    output_filename="metrics.json",
    kwargs={
        "wallclock_to_target": {"target_value": LORENZ_WALLCLOCK_TARGET},
    },
)

LorenzForecastEval = Eval(
    metrics=[
        "stage2_elbo_surrogate",
        "sigma_data_drift",
        "wallclock_to_target",
        "wallclock_to_relative_target",
        "mae",
        "crps_sum",
        "energy_score",
        "regime",
        "recon_mse",
        "denoise_mse",
    ],
    split="test",
    num_samples=16,
    T_split=LORENZ_T_SPLIT,
    output_filename="metrics.json",
    kwargs={
        "wallclock_to_target": {"target_value": LORENZ_WALLCLOCK_TARGET},
        # Lobe = sign(x). z-scoring leaves the boundary at ~0 (the attractor
        # is symmetric under (x, y) -> (-x, -y)). Deadband 0.3: within-lobe
        # spirals dip x near 0, so 0.1 still chatters (~10% of runs < 3
        # steps); 0.3 is the smallest value with zero chatter and run stats
        # stable up to 0.7 (mean residence ~20 steps).
        "regime": {"channel": 0, "deadband": 0.3},
    },
)

LorenzViz = Viz(
    plots=[
        Plot(name="forecast_1d", kwargs={"n_show": 6}),
        Plot(
            name="forecast_2d_spatial",
            kwargs={
                "n_show": 4,
                # Default box/limits are for the robot dataset; the Lorenz
                # x-y projection shows both lobes in z-scored units.
                "obstacle_box": None,
                "xlim": (-3.0, 3.0),
                "ylim": (-3.0, 3.0),
            },
        ),
        # x-channel at t+16: inside the window where the forecast
        # distribution should have gone bimodal.
        Plot(name="forecast_distribution", kwargs={"dim_idx": 0, "t_future_idx": 15}),
        Plot(name="metrics_csv"),
    ],
    split="test",
    num_samples=16,
    T_split=LORENZ_T_SPLIT,
)

__all__ = [
    "LORENZ_T_SPLIT",
    "LORENZ_WALLCLOCK_TARGET",
    "LorenzEval",
    "LorenzForecastEval",
    "LorenzMOObjective",
    "LorenzObjective",
    "LorenzViz",
]
