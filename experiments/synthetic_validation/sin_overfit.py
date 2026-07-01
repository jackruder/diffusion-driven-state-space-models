"""Sanity-check overfit preset: fit a tiny batch of mixed sine curves.

Goal: visually confirm that the full DDSSM stack (encoder + decoder +
init-prior + baseline-Gaussian / centered-diffusion transitions) can
overfit a small, low-noise, well-behaved dataset. This is a pipeline
correctness check, not a generalisation experiment.

Dataset: ``HarmonicMixed`` (``mode="harmonic-mixed"``) shrunk to 16
training sequences. Each is ``x_t = A sin(omega t + phi) + 0.05 eps``
with ``A ~ U[0.5, 2.0]``, ``omega ~ U[0.2, 0.6]``, ``phi ~ U[0, 2pi]``.

Model: ``SynthValModel`` (``data_dim=1, latent_dim=1, j=1``).

Training: short two-stage budget (200 + 400 = 600 stage-relative
steps) — enough to overfit ~16 samples at ``batch_size=8``.

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
from ddssm.experiment.builders import Viz, Eval, Plot, Hparams, Training
from experiments.init_centering.hparams import StagesB
from experiments.synthetic_validation.model import SynthValModel

# 64 train / 64 val / 64 test sequences. v10 made do with 16 but the
# forecast extremes (longest periods, smallest amplitudes) suffered
# from sparse frequency/amplitude coverage. 64 gives ~4× tighter
# coverage at ~4× wall-clock per epoch (mitigated by larger batch).
_SinOverfitData = dataclasses.replace(
    HarmonicMixed,
    N_per_split=64,
    batch_size=16,
)

_HPARAMS = Hparams(
    S=1,
    batch_size=16,
    grad_accum_steps=1,
    enc_lr=1e-3,
    dec_lr=1e-3,
    trans_lr=1e-3,
    ema_decay=0.997,
)

# `steps` is ignored once stages are configured; kept > 0 for the
# sanity-check convention.
_TRAINING = Training(steps=600, log_every=25, amp=False, checkpoint_every=200)

# Stage budget tuned for overfit on 16 samples at batch_size=8 (= 2
# batches/epoch). 200 stage-1 steps ≈ 100 epochs of closed-form Gaussian
# pretraining; 400 stage-2 steps ≈ 200 epochs of centered diffusion.
_STAGES = StagesB(
    baseline_mode="pinned",
    # v7/v8 showed that any stage_1 with PersistenceBaseline drives
    # the encoder into posterior collapse (z ≈ 0 trivially satisfies
    # persistence). Drop stage_1 entirely — train stage_2 from scratch
    # so the encoder is shaped by the joint diffusion + recon loss,
    # not by the easier-to-collapse closed-form KL.
    run_stages=["stage_2"],
    n_pretrain=1,  # unused: stage_1 not in run_stages
    n_stage2=1500,
    base_lr=1e-3,
    log_every=10,
    checkpoint_every=300,
)

# Eval: forecast MAE + CRPS-sum on the val split.
_EVAL = Eval(
    metrics=["mae", "crps_sum", "recon_mse"],
    split="val",
    num_samples=20,
    # SyntheticDataModule reports forecast_split=None, so eval needs an
    # explicit T_split. Use the same context length the viz uses.
    T_split=20,
)

# Viz: one row per sequence; with T=32 and T_split=20 the model has to
# roll out a 12-step forecast for each sin curve. Drawing from the
# train split makes "did it overfit?" visually obvious; the eval CRPS
# above already covers generalisation to val.
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
    # latent_dim=4, j=2: with A, omega, phi all varying per-sample,
    # the latent has to encode (i) where in the trajectory we are
    # and (ii) which curve we're on (its A and omega). A 2-dim
    # latent is right at the edge — give it 4 dims so the transition
    # has enough state to disambiguate the per-sample parameters.
    # channels=32 + diffusion_num_steps=32 keeps the diffusion-
    # transition capacity modest so the CPU run stays fast.
    model=SynthValModel(
        data_dim=1,
        latent_dim=4,
        j=2,
        hidden_dim=32,
        channels=32,
        diffusion_num_steps=32,
        # k_sampling_mode defaults to "adaptive_is" via
        # DiffusionScheduleConfig — the loss-aware optimal IS density
        # per-t with live σ_d² (importance-sampling.org § Mean-
        # dominated regime). No override needed.
    ),
    hparams=_HPARAMS,
    training=_TRAINING,
    stages=_STAGES,
    eval=_EVAL,
    viz=_VIZ,
)
experiment_store(sin_overfit, name="sin_overfit")

__all__ = ["sin_overfit"]
