"""Reusable experiment bases — one factory per dataset family.

Each function returns a fully-formed :class:`Experiment` config given
a ``transition`` slot. Variant files (``harmonic_gauss.py``,
``harmonic_diffusion.py``, …) call the base with their transition and
:func:`tweak` any training/hparams scalars that differ.

Bases hardcode the dataset shape, data module, eval/viz lists, and
sensible default training scalars. Anything you'd want to tweak
between variants is exposed either as a base-function kwarg or
addressable via :func:`experiments._make.tweak`.

If you need a non-trivial variation (different data dim, different
eval set), copy the base into your variant file and modify it — bases
are starting points, not abstract base classes. They have no special
status beyond "I'm sick of typing this".
"""

from __future__ import annotations

from ddssm.builders import (
    DiffV2Transition, Eval, Hparams, KDD, Objective, Plot, Probe,
    Synthetic, Training, Unet, ScheduleV2, Viz,
)

from ._make import make_experiment


# ---------------------------------------------------------------------------
# Synthetic 1D: LGSSM smoke test (D=1, j=1, S=1).
# ---------------------------------------------------------------------------

def synthetic_base(transition, *, steps=500, lambda_warmup_steps=200):
    return make_experiment(
        data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
        data=Synthetic(mode="lgssm", T=64, N_per_split=512, batch_size=32),
        hparams=Hparams(
            S=1, batch_size=32, grad_accum_steps=1,
            lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
            lambda_warmup_steps=lambda_warmup_steps,
            enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
        ),
        training=Training(steps=steps, log_every=25, amp=False),
        transition=transition,
        eval=Eval(metrics=["loss_tail", "recon_mse"], split="val"),
        viz=Viz(
            plots=[Plot(name="metrics_csv", save_filename="train_loss.png",
                        kwargs={"keys": ["loss/total"], "log_y": True})],
            split="val", num_samples=10, T_split=32,
        ),
    )


# ---------------------------------------------------------------------------
# Harmonic synthetic (D=1, j=1, S=1) — clean sine signal.
# ---------------------------------------------------------------------------

def harmonic_base(transition, *, steps=1000, checkpoint_every=200,
                  lambda_warmup_steps=200):
    return make_experiment(
        data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
        data=Synthetic(mode="harmonic", T=64, N_per_split=1024, batch_size=32),
        hparams=Hparams(
            S=1, batch_size=32, grad_accum_steps=1,
            lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
            lambda_warmup_steps=lambda_warmup_steps,
            enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
        ),
        training=Training(steps=steps, log_every=25,
                          checkpoint_every=checkpoint_every, amp=False),
        transition=transition,
        eval=Eval(metrics=["mae", "crps_sum"], split="val",
                  num_samples=32, T_split=32),
        viz=Viz(
            plots=[
                Plot(name="forecast_1d", save_filename="forecast.png",
                     kwargs={"n_show": 4}),
                Plot(name="metrics_csv", save_filename="train_loss.png",
                     kwargs={"keys": ["loss/total"], "log_y": True}),
            ],
            split="val", num_samples=32, T_split=32,
        ),
    )


# ---------------------------------------------------------------------------
# Bimodal synthetic (D=1, j=1, S=4) — energy-score benchmark.
# ---------------------------------------------------------------------------

def bimodal_base(transition, *, mode="bimodal", steps=1000,
                 checkpoint_every=200, lambda_warmup_steps=200):
    return make_experiment(
        data_dim=1, latent_dim=4, j=1, emb_time_dim=16,
        data=Synthetic(mode=mode, T=64, N_per_split=1024, batch_size=32),
        hparams=Hparams(
            S=4, batch_size=32, grad_accum_steps=1,
            lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
            lambda_warmup_steps=lambda_warmup_steps,
            enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
        ),
        training=Training(steps=steps, log_every=25,
                          checkpoint_every=checkpoint_every, amp=False),
        transition=transition,
        eval=Eval(metrics=["energy_score", "crps_sum"], split="val",
                  num_samples=64, T_split=32),
        viz=Viz(
            plots=[
                Plot(name="forecast_1d", save_filename="forecast.png",
                     kwargs={"n_show": 4}),
                Plot(name="metrics_csv", save_filename="train_loss.png",
                     kwargs={"keys": ["loss/total"], "log_y": True}),
            ],
            split="val", num_samples=64, T_split=32,
        ),
    )


# ---------------------------------------------------------------------------
# Robot navigation 2D (D=2, j=2, S=1) — spatial trajectory.
# ---------------------------------------------------------------------------

def robot_2d_base(transition, *, steps=2000, checkpoint_every=500,
                  lambda_warmup_steps=400):
    return make_experiment(
        data_dim=2, latent_dim=6, j=2, emb_time_dim=16,
        data=Synthetic(mode="robot-basis-pursuit", T=64, D=2,
                       N_per_split=1024, batch_size=32),
        hparams=Hparams(
            S=1, batch_size=32, grad_accum_steps=1,
            lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
            lambda_warmup_steps=lambda_warmup_steps,
            enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
        ),
        training=Training(steps=steps, log_every=50,
                          checkpoint_every=checkpoint_every, amp=False),
        transition=transition,
        eval=Eval(metrics=["energy_score", "crps_sum"], split="val",
                  num_samples=32, T_split=32),
        viz=Viz(
            plots=[
                Plot(name="forecast_2d_spatial", save_filename="forecast_2d.png",
                     kwargs={"n_show": 4}),
                Plot(name="metrics_csv", save_filename="train_loss.png",
                     kwargs={"keys": ["loss/total"], "log_y": True}),
            ],
            split="val", num_samples=32, T_split=32,
        ),
    )


# ---------------------------------------------------------------------------
# KDD Cup 2018 (D=6, j=1, covariate_dim=3) — real PM2.5 forecasting.
# ---------------------------------------------------------------------------

def kdd_base(transition, *, steps=5000, batch_size=128,
             lambda_warmup_steps=500):
    return make_experiment(
        data_dim=6, latent_dim=8, j=1, emb_time_dim=32, covariate_dim=3,
        data=KDD(batch_size=batch_size, eval_step_size=24),
        hparams=Hparams(
            S=1, batch_size=batch_size, grad_accum_steps=1,
            lambda_schedule="linear", lambda_start=0.001, lambda_end=1.0,
            lambda_warmup_steps=lambda_warmup_steps,
            enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
        ),
        training=Training(steps=steps, log_every=50,
                          checkpoint_every=500, amp=True),
        transition=transition,
        eval=Eval(metrics=["mae", "crps_sum"], split="test", num_samples=32),
        viz=Viz(
            plots=[
                Plot(name="forecast_1d", save_filename="forecast.png",
                     kwargs={"n_show": 4}),
                Plot(name="metrics_csv", save_filename="train_loss.png",
                     kwargs={"keys": ["loss/total"], "log_y": True}),
            ],
            split="test", num_samples=32,
        ),
    )


# ---------------------------------------------------------------------------
# Variance probe — always Diffusion V2 + Probe spec, vary only the data.
# ---------------------------------------------------------------------------

def variance_probe_base(*, name, mode, data_dim=1, latent_dim=4):
    return make_experiment(
        data_dim=data_dim, latent_dim=latent_dim, j=1, emb_time_dim=16,
        checkpoint_dir=f"${{oc.env:PWD,.}}/runs/variance_probe/{name}/checkpoints",
        data=Synthetic(mode=mode, T=64, D=data_dim,
                       N_per_split=256, batch_size=32),
        hparams=Hparams(
            S=1, batch_size=32, grad_accum_steps=1,
            lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
            lambda_warmup_steps=50,
            enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
        ),
        training=Training(steps=300, log_every=20, checkpoint_every=100,
                          amp=False),
        transition=DiffV2Transition(unet=Unet(), schedule=ScheduleV2()),
        objective=Objective(metric="loss/total", split="train", tail_frac=0.1),
        variance=Probe(),
    )


__all__ = [
    "synthetic_base", "harmonic_base", "bimodal_base",
    "robot_2d_base", "kdd_base", "variance_probe_base",
]
