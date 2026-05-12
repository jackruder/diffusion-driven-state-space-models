"""Eval specs for the synthetic-data experiments."""

from __future__ import annotations

from ddssm.builders import Eval

from conf.registry import eval_store


# 1D forecast (harmonic, robot 1D).
Forecast1D = Eval(
    metrics=["mae", "crps_sum"], split="val",
    num_samples=32, T_split=16,
)

# Energy-score bimodal (S=4).
BimodalEnergy = Eval(
    metrics=["energy_score", "crps_sum"], split="val",
    num_samples=64, T_split=16,
)

# 2D robot navigation.
Robot2D = Eval(
    metrics=["energy_score", "crps_sum"], split="val",
    num_samples=32, T_split=16,
)

# LGSSM smoke (no forecast — just recon).
LGSSM = Eval(metrics=["loss_tail", "recon_mse"], split="val")

eval_store(Forecast1D, name="forecast_1d")
eval_store(BimodalEnergy, name="bimodal_energy")
eval_store(Robot2D, name="robot2d")
eval_store(LGSSM, name="lgssm")
