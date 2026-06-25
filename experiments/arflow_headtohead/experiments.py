"""Encoder head-to-head: gaussian vs IAF vs deterministic-causal, two phases.

Three encoders behind the same model/training (only ``encoder_type`` +
``arflow_stochastic_state`` differ), on an EASY (``lgssm``, D=1, analytic Kalman
optimum) and a HARD (``nonlin_bimodal_lift_mv``, D=8 obs / true latent 4) dataset:

* **gaussian** — the settled sequential encoder (reference).
* **iaf**      — parallel AR-flow-on-noise, ``arflow_stochastic_state=True``
                 (conditioner sees the noise history; μ,σ condition on the path).
* **det**      — parallel deterministic-causal, ``arflow_stochastic_state=False``
                 (μ,σ = f(h), z-history amortized).

Two phases, two questions:

* **Phase 1 — encoder CAPACITY** (``h2h_cap__<enc>__<ds>``): a pure autoencoder —
  λ pinned EXACTLY to 0 (``stage_1_lambda_start=end=0``, no ramp), stage-1 only.
  Objective = converged val recon (``loss/distortion/rec``); sweep base_lr only
  (``+sweep=h2h_lr_only``). Converges flat (no rate confound) → "can the encoder
  reconstruct?". NOTE λ=0 drives σ→0, so this measures the *deterministic backbone*
  capacity: iaf and det share a backbone and should land ~equal here.
* **Phase 2 — full MODEL** (``h2h__<enc>__<ds>``): the two-stage ELBO, all stage
  knobs SWEPT (``+sweep=h2h_full``). Objective = held-out **VAL forecast CRPS-sum**
  (``source="json"`` → a single post-training forecast eval per trial); finalists
  are re-scored on TEST. Selecting on CRPS (not val ELBO) follows the lesson that
  val ELBO is a poor forecast proxy.

The parallel encoders run at BUMPED capacity (``arflow_channels=128``,
``arflow_causal_layers=4``), decoupled from the transition's ``channels=48`` so the
encoder gets more punch without inflating the (already-swept-on-gaussian) diffusion.
Report param counts alongside recon so a "match" isn't secretly "2x the capacity".

Run a Phase-2 sweep (many 1-trial workers can share one study DB)::

    .venv/bin/python -m ddssm.app --multirun experiment=h2h__iaf__nlblmv \\
        +sweep=h2h_full hydra.sweeper.n_trials=35 \\
        hydra.sweeper.study_name=h2h_iaf_nlblmv \\
        hydra.sweeper.storage=sqlite:///h2h.db
"""

from __future__ import annotations

import dataclasses

from experiments._make import experiment
from ddssm.data.presets import LGSSM, NonlinBimodalLiftMV
from ddssm.experiment.stores import experiment_store
from ddssm.experiment.builders import Eval, Objective
from experiments.gluonts_forecast.model import GluonModel
from experiments.gluonts_forecast.hparams import (
    GluonHparams,
    GluonStages,
    GluonTraining,
)

_T = 32
_LATENT_DIM = 8  # held constant across datasets so only data + encoder vary.
# History 24 / horizon 8 — matches eval_baselines.py's T*3//4 split so the model
# and the marginal-baseline gate are scored on the SAME past/future boundary.
_T_SPLIT = 24

# encoder key -> the model kwargs that select it.
_ENCODERS = {
    "gaussian": dict(encoder_type="gaussian"),
    "iaf": dict(encoder_type="arflow", arflow_stochastic_state=True),
    "det": dict(encoder_type="arflow", arflow_stochastic_state=False),
}
# dataset key -> (data-module preset, obs dim). LGSSM exposes GT latents so
# finalists can be scored on latent recovery + the analytic Kalman reference.
_DATASETS = {
    "lgssm": (dataclasses.replace(LGSSM, expose_gt_latents=True), 1),
    "nlblmv": (NonlinBimodalLiftMV, 8),
}


def _model(encoder_key: str, data_dim: int, j: int = 1):
    """Same backbone for every cell; parallel encoders get bumped capacity."""
    caps = (
        {}
        if encoder_key == "gaussian"
        else dict(arflow_channels=128, arflow_causal_layers=4)
    )
    return GluonModel(
        data_dim=data_dim, latent_dim=_LATENT_DIM, j=j, T_max=_T,
        channels=48, nheads=2, summary_layers=1, diffusion_layers=3,
        num_steps=64, grad_checkpoint=False,
        **_ENCODERS[encoder_key], **caps,
    )


def _phase2_cell(encoder_key: str, dataset_key: str, j: int = 1):
    """Full two-stage ELBO cell; stage hyperparameters are SWEPT (un-pinned)."""
    data, data_dim = _DATASETS[dataset_key]
    return experiment(
        data=data,
        model=_model(encoder_key, data_dim, j),
        hparams=dataclasses.replace(GluonHparams, batch_size=32, clip_grad_norm=1.0),
        training=GluonTraining,
        # Budget/cadence fixed from calibration (plateau ~3700); everything else
        # (base_lr, dec/trans_mult, sigma_pert, λ starts + warmup fracs,
        # lambda_sigma_p) is left at neutral defaults for +sweep=h2h_full.
        stages=GluonStages(
            n_pretrain=450, n_stage2=4000,
            validate_every=100, log_every=50, checkpoint_every=1000,
        ),
        # Per-trial objective eval: forecast CRPS-sum on the VAL split (select on
        # val, report on test). One forecast pass per trial; finalists get the
        # full {crps,energy,nll}-on-test eval via `python -m ddssm.evaluate`.
        eval=Eval(
            metrics=["crps_sum"], split="val", num_samples=100,
            T_split=_T_SPLIT, output_filename="metrics.json",
        ),
        objective=Objective(metric="crps_sum", source="json"),
    )


def _phase1_cell(encoder_key: str, dataset_key: str, j: int = 1):
    """Pure-AE capacity probe: λ≡0 (no ramp), stage-1 only; objective = recon."""
    data, data_dim = _DATASETS[dataset_key]
    return experiment(
        data=data,
        model=_model(encoder_key, data_dim, j),
        hparams=dataclasses.replace(GluonHparams, batch_size=32, clip_grad_norm=1.0),
        training=GluonTraining,
        stages=GluonStages(
            run=["stage_1"],
            n_pretrain=3000,
            # λ ≡ 0 (start=end=0) AND no σ_p regulariser → loss == pure recon.
            stage_1_lambda_start=0.0, stage_1_lambda_end=0.0, lambda_sigma_p=0.0,
            validate_every=100, log_every=50, checkpoint_every=1000,
            early_stop_enabled=True, early_stop_window=500,
            early_stop_min_improvement=1e-4, early_stop_warmup_steps=500,
        ),
        # CAPACITY METRIC = reconstruction MSE on the decoded posterior MEAN. λ=0
        # leaves the decoder σ unregularised, so the distortion NLL collapses to −∞
        # (σ_dec→0) — degenerate and non-discriminating. recon_mse scores μ_x only,
        # so it is bounded and stays a clean encoder-capacity measure.
        eval=Eval(
            metrics=["recon_mse"], split="val", num_samples=1,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="recon_mse", source="json"),
    )


for _ds in ("lgssm", "nlblmv"):
    for _enc in ("gaussian", "iaf", "det"):
        experiment_store(_phase2_cell(_enc, _ds), name=f"h2h__{_enc}__{_ds}")
        experiment_store(_phase1_cell(_enc, _ds), name=f"h2h_cap__{_enc}__{_ds}")
