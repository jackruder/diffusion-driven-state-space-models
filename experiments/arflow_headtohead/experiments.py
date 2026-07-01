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
    # Same sequential encoder as `gaussian` but a LOCAL future-summary (filtering
    # q(z_t|x_t)) — tests whether a simpler encoder gives a cleaner near-identity
    # latent frame at latent_dim==data_dim==8.
    "gaussian_local": dict(encoder_type="gaussian_local"),
    "iaf": dict(encoder_type="arflow", arflow_stochastic_state=True),
    "det": dict(encoder_type="arflow", arflow_stochastic_state=False),
    # Forward-message variants (parallel encoder + a forward-causal data message):
    # o1_flow = option-1 (forward pass over b → o_t) sent through the IAF flow;
    # fb_mf / fb_flow = [f_t, b_t] context with a mean-field / IAF head respectively.
    "o1_flow": dict(
        encoder_type="arflow", arflow_stochastic_state=True,
        arflow_forward_message="fwd_summary",
    ),
    "fb_mf": dict(
        encoder_type="arflow", arflow_stochastic_state=False,
        arflow_forward_message="fwd_data",
    ),
    "fb_flow": dict(
        encoder_type="arflow", arflow_stochastic_state=True,
        arflow_forward_message="fwd_data",
    ),
    # Pinned identity encoder/decoder (z=x, requires latent_dim==data_dim): the
    # diffusion transition denoises in OBSERVATION space → a CSDI-style obs-space
    # model inside the DDSSM pipeline. Decisive isolation of the latent-frame
    # bottleneck: if this hits ~CSDI (58%), the encoder frame is the problem; if it
    # stays ~gaussian (26%), the transition/training is.
    "identity": dict(encoder_type="identity"),
    # identity enc/dec + the LITERAL vendored ermongroup CSDI in the transition
    # slot (transition_type="csdi"). With j==HIST this reproduces the 58% standalone
    # CSDI forecaster INSIDE the DDSSM ELBO pipeline → decisively indicts (≈58%) or
    # exonerates (≈22%) our own transition code, since the latent frame is already
    # cleared by the plain `identity` cell stalling at ~22%.
    "identity_csdi": dict(encoder_type="identity", transition_type="csdi"),
    # LEARNED gaussian frame (latent_dim=8) + the literal CSDI transition. Pairs
    # with identity_csdi (66%): if this also lands ~66% the learned frame is fine
    # for a correct transition; if it drops toward gaussian's ~26% the learned
    # frame is a real secondary bottleneck (co-evolution / amortization gap).
    "gaussian_csdi": dict(encoder_type="gaussian", transition_type="csdi"),
    # "KITCHEN-SINK" identity + OUR DiffusionTransition made CSDI-LIKE on every axis
    # found to differ from the literal CSDI (the 24%->66% gap on the identity frame):
    #   - k_sampling_mode=uniform     (CSDI samples noise levels uniformly)
    #   - emb_feature_dim=16          (CSDI's per-channel feature embedding; decoupled
    #                                  from emb_time_dim, which stays 0)
    #   - time_mixer=transformer      (non-causal RoPE attention over the window, the
    #                                  CSDI time axis; ours was a 3-tap conv)
    #   - diffusion_sampler=edm       (Karras Heun + stochastic churn; ours was a
    #                                  deterministic VP probability-flow Euler)
    #   - baseline_type=zero          (μ_p≡0: persistence centers on ONE bimodal mode,
    #                                  so the transition only modeled a residual)
    # All knobs are independent factory params, so single-axis ablations are CLI
    # overrides off this cell. Read vs identity+ours (24%) and identity+CSDI (66%):
    # if this closes toward 66%, the gap is the transition RECIPE (one of these axes);
    # if it stays ~24%, the gap is elsewhere (training/pipeline).
    "identity_csdilike": dict(
        encoder_type="identity",
        baseline_type="zero",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="uniform",
        channels=80,
        diffusion_layers=3,
        nheads=2,
    ),
    # Same as identity_csdilike but with adaptive_is k-sampling.
    "identity_csdilike_ais": dict(
        encoder_type="identity",
        baseline_type="zero",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=80,
        diffusion_layers=3,
        nheads=2,
    ),
    # bumped capacity: 96ch / 4 layers / 4 heads (704K params vs 409K)
    "identity_csdilike_ais_big": dict(
        encoder_type="identity",
        baseline_type="zero",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=4,
    ),
    # adaptive_is + persistence baseline
    "identity_csdilike_ais_persist": dict(
        encoder_type="identity",
        baseline_type="persistence",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=80,
        diffusion_layers=3,
        nheads=2,
    ),
}

# Single-axis ablations off the kitchen-sink (LEAVE-ONE-OUT): each reverts exactly
# ONE CSDI-like axis to the identity+ours (24%) baseline value, keeping the other
# four flipped, to attribute the 24->55% gain. Plus the FRAME-TRANSFER cell: the
# full recipe on the LEARNED gaussian frame (does the recipe win survive a lossy
# frame, or collapse like literal CSDI did 66->16?). Baked as presets so the
# eval_baselines/probe (which rebuild the model from the experiment NAME alone)
# reconstruct the exact architecture.
_KS = _ENCODERS["identity_csdilike"]
_ENCODERS.update(
    {
        # noise-level sampling: uniform -> adaptive_is (the 24% anchor's value)
        "identity_csdilike_ksamp": {
            **_KS, "k_sampling_mode": "adaptive_is", "pk_floor": 1e-3,
        },
        # per-channel feature embedding: 16 -> 0 (off)
        "identity_csdilike_embfeat": {**_KS, "emb_feature_dim": 0},
        # time mixer: transformer (non-causal RoPE) -> conv (3-tap)
        "identity_csdilike_timemix": {**_KS, "time_mixer": "conv"},
        # sampler: EDM Heun+churn -> legacy pf_ode (deterministic VP Euler).
        # Inference-only, so this cell re-scores the kitchen-sink checkpoint.
        "identity_csdilike_sampler": {
            **_KS, "diffusion_sampler": "pf_ode", "edm_s_churn": 0.0,
        },
        # baseline: zero (μ_p≡0) -> persistence (μ_p = z_{t-1})
        "identity_csdilike_baseline": {**_KS, "baseline_type": "persistence"},
        # FRAME TRANSFER: full kitchen-sink recipe on the learned gaussian frame
        "gaussian_csdilike": {**_KS, "encoder_type": "gaussian"},
    }
)

# dataset key -> (data-module preset, obs dim). LGSSM exposes GT latents so
# finalists can be scored on latent recovery + the analytic Kalman reference.
_DATASETS = {
    "lgssm": (dataclasses.replace(LGSSM, expose_gt_latents=True), 1),
    "nlblmv": (NonlinBimodalLiftMV, 8),
}


def _model(encoder_key: str, data_dim: int, j: int = 1):
    """Same backbone for every cell; parallel encoders get bumped capacity."""
    caps = (
        dict(arflow_channels=128, arflow_causal_layers=4)
        if _ENCODERS[encoder_key].get("encoder_type") == "arflow"
        else {}
    )
    enc = {k: v for k, v in _ENCODERS[encoder_key].items()
           if k not in ("channels", "nheads", "diffusion_layers")}
    return GluonModel(
        data_dim=data_dim, latent_dim=_LATENT_DIM, j=j, T_max=_T,
        channels=_ENCODERS[encoder_key].get("channels", 48),
        nheads=_ENCODERS[encoder_key].get("nheads", 2),
        summary_layers=1,
        diffusion_layers=_ENCODERS[encoder_key].get("diffusion_layers", 3),
        num_steps=64, grad_checkpoint=False,
        **enc, **caps,
    )


def _phase2_cell(encoder_key: str, dataset_key: str, j: int = 1):
    """Single-stage ELBO cell (stage-2-only); stage hyperparameters are SWEPT.

    Stage-1 pretraining is dropped here (harmful in early arflow runs): with
    ``run=["stage_2"]`` no centering handoff fires, so there is no μ_p
    snapshot/pin and no encoder perturbation, and ``per_t`` σ_data
    self-calibrates from cold via its EMA warmup.
    """
    data, data_dim = _DATASETS[dataset_key]
    return experiment(
        data=data,
        model=_model(encoder_key, data_dim, j),
        hparams=dataclasses.replace(GluonHparams, batch_size=32),
        training=GluonTraining,
        # Budget/cadence fixed from calibration (plateau ~3700); everything else
        # (base_lr, dec/trans_mult, λ start + warmup frac) is left at neutral
        # defaults for +sweep=h2h_full. Stage-2-only: the stage-1 knobs
        # (n_pretrain, sigma_pert, lambda_sigma_p, stage_1_*) are inert.
        stages=GluonStages(
            run=["stage_2"], n_stage2=10000,
            validate_every=100, log_every=50, checkpoint_every=2000,
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
        hparams=dataclasses.replace(GluonHparams, batch_size=32),
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

# Forward-message variants + the local-summary gaussian: only the discriminative
# nlblmv phase-2 cell (the p=0.85 head-to-head). Same model/training/eval — only the
# encoder differs.
for _enc in ("gaussian_local", "o1_flow", "fb_mf", "fb_flow"):
    experiment_store(_phase2_cell(_enc, "nlblmv"), name=f"h2h__{_enc}__nlblmv")

# j=2 variants on nlblmv: the true generative process is 2nd-order Markov in z
# (recovering the sign-persistence state s_{t-1} needs z_{t-1} AND z_{t-2}), so the
# j=1 cells above are structurally unable to model the 0.85/0.15 mode-weighting.
# Same cells, only the transition/encoder conditioning order changes.
for _enc in (
    "gaussian", "gaussian_local", "iaf", "det", "o1_flow", "fb_mf", "fb_flow",
    "identity",
):
    experiment_store(_phase2_cell(_enc, "nlblmv", j=2), name=f"h2h__{_enc}__nlblmv__j2")

# Faithful-CSDI control: identity enc/dec + the literal vendored CSDI transition at
# j=10 (== CSDI HIST=10, SEQ=11). Reproduces the 58% standalone forecaster inside
# the DDSSM pipeline; pass/stall cleanly indicts/exonerates our transition code.
experiment_store(
    _phase2_cell("identity_csdi", "nlblmv", j=10), name="h2h__csdi__nlblmv__j10"
)

# identity enc/dec + OUR DiffusionTransition at j=10 — the 4th cell of the 2x2
# {identity,gaussian} x {ours,CSDI}. Single-variable swap from the faithful CSDI run
# (only the transition differs): does our transition trail CSDI in obs-space too?
experiment_store(
    _phase2_cell("identity", "nlblmv", j=10), name="h2h__identity__nlblmv__j10"
)

# LEARNED gaussian frame (latent_dim=8) + literal CSDI transition at j=10. Pairs with
# the faithful identity+CSDI run (66%): ≈66% => the learned frame is fine for a correct
# transition; drop toward gaussian-our-transition (~26%) => the learned frame is a real
# secondary bottleneck (co-evolution / amortization gap).
experiment_store(
    _phase2_cell("gaussian_csdi", "nlblmv", j=10),
    name="h2h__gaussian_csdi__nlblmv__j10",
)

# Kitchen-sink CSDI-like identity + OUR transition at j=10: every axis that differed
# from the literal CSDI flipped at once (uniform noise sampling, feature embedding,
# transformer time mixer, EDM stochastic sampler, zero baseline). Read vs identity+ours
# (24%) and identity+CSDI (66%) to attribute the gap to the transition recipe.
experiment_store(
    _phase2_cell("identity_csdilike", "nlblmv", j=10),
    name="h2h__identity_csdilike__nlblmv__j10",
)

# Same as kitchen-sink but adaptive_is k-sampling (weights=1 via diffusion.py TEMP).
experiment_store(
    _phase2_cell("identity_csdilike_ais", "nlblmv", j=10),
    name="h2h__identity_csdilike_ais__nlblmv__j10",
)

# Persistence baseline variants (uniform and adaptive_is).
# identity_csdilike_baseline is already registered in the attribution loop below,
# so only register the adaptive_is + persistence variant here.
experiment_store(
    _phase2_cell("identity_csdilike_ais_persist", "nlblmv", j=10),
    name="h2h__identity_csdilike_ais_persist__nlblmv__j10",
)

# Attribution batch: 5 single-axis leave-one-out ablations off the kitchen-sink +
# the gaussian-frame transfer, all j=10 on nlblmv. Each ablation cell name encodes
# the axis reverted to the 24% baseline (see the _ENCODERS.update block above).
for _enc in (
    "identity_csdilike_ksamp",
    "identity_csdilike_embfeat",
    "identity_csdilike_timemix",
    "identity_csdilike_sampler",
    "identity_csdilike_baseline",
    "gaussian_csdilike",
):
    experiment_store(
        _phase2_cell(_enc, "nlblmv", j=10), name=f"h2h__{_enc}__nlblmv__j10"
    )
