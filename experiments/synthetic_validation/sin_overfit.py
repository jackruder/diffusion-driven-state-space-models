"""Sanity-check overfit preset: fit a tiny batch of mixed sine curves.

Goal: visually confirm that the full DDSSM stack (encoder + decoder +
init-prior + baseline-Gaussian / centered-diffusion transitions) can
overfit a small, low-noise, well-behaved dataset. This is a pipeline
correctness check, not a generalisation experiment.

Dataset: ``HarmonicMixed`` (``mode="harmonic-mixed"``) shrunk to 64
training sequences. Each is ``x_t = A sin(omega t + phi) + 0.05 eps``
with ``A ~ U[0.5, 2.0]``, ``omega ~ U[0.2, 0.6]``, ``phi ~ U[0, 2pi]``.

Model: ``SynthValModel`` (``data_dim=1, latent_dim=4, j=2``).

Training: stage-2-only with enough budget to reach the noise floor.

Eval + viz are wired so::

    python -m ddssm.app       experiment=sin_overfit
    python -m ddssm.evaluate  experiment=sin_overfit checkpoint=<path>
    python -m ddssm.visualize experiment=sin_overfit checkpoint=<path>

…produces a ``forecast_1d.png`` with one row per sequence showing
observed / reconstruction / forecast samples + mean. With T=32 and a
``T_split=20`` context, the model gets 20 steps of history and has to
roll out the remaining 12 steps.
"""

from __future__ import annotations

import dataclasses

from experiments._make import experiment
from ddssm.data.presets import HarmonicMixed
from ddssm.experiment.stores import experiment_store
from ddssm.experiment.builders import Viz, Eval, Plot, Hparams, Objective, Training
from experiments.synthetic_validation.model import SynthValModel

_SinOverfitData = dataclasses.replace(
    HarmonicMixed,
    N_per_split=64,
    batch_size=16,
)

_HPARAMS = Hparams(
    S=1,
    batch_size=16,
    grad_accum_steps=1,
    enc_lr=5e-4,
    dec_lr=5e-4,
    trans_lr=5e-4,
    ema_decay=0.997,
)

_TRAINING = Training(steps=5000, log_every=100, amp=False, checkpoint_every=5000)

_EVAL = Eval(
    metrics=["mae", "rmse", "crps_sum", "recon_mse"],
    split="val",
    num_samples=20,
    T_split=20,
)

_VIZ = Viz(
    plots=[
        Plot(
            name="forecast_1d",
            save_filename="forecast_1d.png",
            kwargs={"n_show": 8, "time_start_at_zero": True, "show_title": True},
        ),
        Plot(
            name="metrics_csv",
            save_filename="metrics_csv.png",
        ),
    ],
    split="train",
    num_samples=20,
    T_split=20,
)

sin_overfit = experiment(
    data=_SinOverfitData,
    model=SynthValModel(
        data_dim=1,
        latent_dim=4,
        j=4,
        hidden_dim=128,
        channels=64,
        diffusion_num_steps=64,
        diffusion_layers=2,
        baseline_form="zero",
        diffusion_time_chunk_size=32,
        recon_time_chunk=32,
    ),
    hparams=_HPARAMS,
    training=_TRAINING,
    eval=_EVAL,
    viz=_VIZ,
    objective=Objective(metric="rmse", source="json"),
)
experiment_store(sin_overfit, name="sin_overfit")

__all__ = ["sin_overfit"]
