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
from ddssm.experiment.builders import Eval, Objective, Probe
from experiments.gluonts_forecast.model import GluonModel
from experiments.gluonts_forecast.hparams import (
    GluonHparams,
    GluonTraining,
)
from ddssm.training.stages import (
    LambdaRampConf,
    LrScheduleConf,
    LrScheduleGroupConf,
)


def _training(steps: int, log_every: int = 50, validate_every: int = 100, checkpoint_every: int = 2000):
    """Convenience: derive a per-cell Training from the shared GluonTraining defaults."""
    return dataclasses.replace(
        GluonTraining,
        steps=steps,
        log_every=log_every,
        validate_every=validate_every,
        checkpoint_every=checkpoint_every,
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
        encoder_type="arflow",
        arflow_stochastic_state=True,
        arflow_forward_message="fwd_summary",
    ),
    "fb_mf": dict(
        encoder_type="arflow",
        arflow_stochastic_state=False,
        arflow_forward_message="fwd_data",
    ),
    "fb_flow": dict(
        encoder_type="arflow",
        arflow_stochastic_state=True,
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
    # Trial 2 capacity-reduction siblings of _ais_big. Same recipe (adaptive_is,
    # EDM, zero baseline, transformer time-mixer, emb_feature_dim=16), only
    # channels/diffusion_layers/nheads shrunk. head_dim = channels/nheads >= 8.
    "identity_csdilike_ais_small": dict(
        encoder_type="identity",
        baseline_type="zero",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=64,
        diffusion_layers=3,
        nheads=2,
    ),
    "identity_csdilike_ais_tiny": dict(
        encoder_type="identity",
        baseline_type="zero",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=48,
        diffusion_layers=2,
        nheads=2,
    ),
    # 64ch/3L but doubled heads vs _small (nheads=4, head_dim=16). Same conv-proj
    # param count as _small; probes whether finer-grained attention closes the
    # small→med JSD gap without adding channels.
    "identity_csdilike_ais_small_h4": dict(
        encoder_type="identity",
        baseline_type="zero",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=64,
        diffusion_layers=3,
        nheads=4,
    ),
    # Same transition recipe as identity_csdilike_ais_big, but swaps the frozen
    # identity encoder (σ_t pinned to e^{-3.5} ≈ 0.03) for a learned gaussian
    # encoder — puts σ_t into the O(1) regime early in training, which is
    # where the LSGM Rao-Blackwellization gap between ESM and DSM actually
    # blows up at low σ̃. Positive control for the variance probe.
    #
    # nheads=2 here (not 4 like the identity variant) because the gaussian
    # encoder's fut_summary transformer uses summary_dim = 2*latent_dim = 16,
    # and nheads=4 would give head_dim=4 which SDPA rejects (needs ≥ 8).
    "gaussian_csdilike_ais_big": dict(
        encoder_type="gaussian",
        baseline_type="zero",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
    ),
    # ArFlow (IAF) counterpart to gaussian_csdilike_ais_big_wideenc, matched by
    # ENCODER PARAMETER COUNT (~110K vs the wideenc-gaussian's ~118K). Same big
    # transition (96/4/2) + csdilike_ais recipe. NOTE: ArFlow is hard-locked to
    # PersistenceBaseline (dssd.py:63), so this cell centers μ_p = z_{t-1}
    # whereas the wideenc-gaussian centers μ_p ≡ 0 → real architectural gap.
    "iaf_csdilike_ais_big_matchedenc": dict(
        encoder_type="arflow",
        arflow_stochastic_state=True,
        baseline_type="persistence",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
        arflow_channels=32,
        arflow_causal_layers=4,
    ),
    # Sibling arflow variants, all sharing the iaf_matchedenc structure
    # (arflow_channels=32, arflow_causal_layers=4) + csdilike_ais recipe +
    # big diffusion transition + persistence baseline. Only differ in the
    # arflow-family axes: stochastic_state (T→IAF / F→deterministic-causal)
    # and forward_message (none / fwd_summary / fwd_data).
    "det_csdilike_ais_big_matchedenc": dict(
        encoder_type="arflow",
        arflow_stochastic_state=False,
        baseline_type="persistence",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
        arflow_channels=32,
        arflow_causal_layers=4,
    ),
    "o1_flow_csdilike_ais_big_matchedenc": dict(
        encoder_type="arflow",
        arflow_stochastic_state=True,
        arflow_forward_message="fwd_summary",
        baseline_type="persistence",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
        arflow_channels=32,
        arflow_causal_layers=4,
    ),
    "fb_mf_csdilike_ais_big_matchedenc": dict(
        encoder_type="arflow",
        arflow_stochastic_state=False,
        arflow_forward_message="fwd_data",
        baseline_type="persistence",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
        arflow_channels=32,
        arflow_causal_layers=4,
    ),
    # 8x-encoder-capacity sibling of fb_mf_matchedenc. arflow_channels 32→96
    # and arflow_causal_layers 4→6 → encoder 118K → 944K (~8.0×). All other
    # axes (transition, baseline, csdilike recipe) unchanged for a clean
    # arflow-only capacity comparison.
    "fb_mf_csdilike_ais_big_bigenc8x": dict(
        encoder_type="arflow",
        arflow_stochastic_state=False,
        arflow_forward_message="fwd_data",
        baseline_type="persistence",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
        arflow_channels=96,
        arflow_causal_layers=6,
    ),
    # Full-conv variant of fb_mf_bigenc8x. Same axes but every "core"
    # transformer is swapped for conv: score-net time+feature mixers and the
    # ArFlow causal_net backbone. Cheaper (fewer params per layer at same
    # channels) + fundamentally different inductive bias for the same task.
    "fb_mf_csdilike_ais_big_bigenc8x_conv": dict(
        encoder_type="arflow",
        arflow_stochastic_state=False,
        arflow_forward_message="fwd_data",
        baseline_type="persistence",
        emb_feature_dim=16,
        time_mixer="conv",
        feature_mixer="conv",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
        arflow_channels=96,
        arflow_causal_layers=6,
        arflow_backbone="conv",
    ),
    "fb_flow_csdilike_ais_big_matchedenc": dict(
        encoder_type="arflow",
        arflow_stochastic_state=True,
        arflow_forward_message="fwd_data",
        baseline_type="persistence",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
        arflow_channels=32,
        arflow_causal_layers=4,
    ),
    # Same as gaussian_csdilike_ais_big but overrides the "single width rule"
    # for the encoder side only. Encoder + fut_summary go from width=16
    # (2*latent) to width=64 → encoder capacity ~ 17K → ~200K. Decoder stays
    # at 2*latent (transition-input dim unchanged).
    "gaussian_csdilike_ais_big_wideenc": dict(
        encoder_type="gaussian",
        baseline_type="zero",
        emb_feature_dim=16,
        time_mixer="transformer",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
        encoder_hidden_dim=64,
    ),
    # Full-conv score-net sibling of gaussian_csdilike_ais_big_wideenc.
    # Score-net time+feature mixers switched from transformer to conv.
    # Encoder side (fut_summary transformer) unchanged (no conv option
    # for fut_summary — GRU is the only non-transformer alternative).
    "gaussian_csdilike_ais_big_wideenc_conv": dict(
        encoder_type="gaussian",
        baseline_type="zero",
        emb_feature_dim=16,
        time_mixer="conv",
        feature_mixer="conv",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
        encoder_hidden_dim=64,
    ),
    # Same as gaussian_csdilike_ais_big_wideenc_conv but also swaps the
    # fut_summary transformer for a GRU. Fully "conv+rnn" (no attention
    # anywhere in the model).
    "gaussian_csdilike_ais_big_wideenc_conv_gru": dict(
        encoder_type="gaussian",
        baseline_type="zero",
        emb_feature_dim=16,
        time_mixer="conv",
        feature_mixer="conv",
        diffusion_sampler="edm",
        edm_s_churn=16.0,
        edm_s_noise=1.0,
        k_sampling_mode="adaptive_is",
        channels=96,
        diffusion_layers=4,
        nheads=2,
        encoder_hidden_dim=64,
        fut_summary_type="gru",
        fut_summary_gru_layers=2,
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
_ENCODERS.update({
    # noise-level sampling: uniform -> adaptive_is (the 24% anchor's value)
    "identity_csdilike_ksamp": {
        **_KS,
        "k_sampling_mode": "adaptive_is",
        "pk_floor": 1e-3,
    },
    # per-channel feature embedding: 16 -> 0 (off)
    "identity_csdilike_embfeat": {**_KS, "emb_feature_dim": 0},
    # time mixer: transformer (non-causal RoPE) -> conv (3-tap)
    "identity_csdilike_timemix": {**_KS, "time_mixer": "conv"},
    # sampler: EDM Heun+churn -> legacy pf_ode (deterministic VP Euler).
    # Inference-only, so this cell re-scores the kitchen-sink checkpoint.
    "identity_csdilike_sampler": {
        **_KS,
        "diffusion_sampler": "pf_ode",
        "edm_s_churn": 0.0,
    },
    # baseline: zero (μ_p≡0) -> persistence (μ_p = z_{t-1})
    "identity_csdilike_baseline": {**_KS, "baseline_type": "persistence"},
    # FRAME TRANSFER: full kitchen-sink recipe on the learned gaussian frame
    "gaussian_csdilike": {**_KS, "encoder_type": "gaussian"},
})

# dataset key -> (data-module preset, obs dim). LGSSM exposes GT latents so
# finalists can be scored on latent recovery + the analytic Kalman reference.
_DATASETS = {
    "lgssm": (dataclasses.replace(LGSSM, expose_gt_latents=True), 1),
    "nlblmv": (NonlinBimodalLiftMV, 8),
}


def _model(encoder_key: str, data_dim: int, j: int = 1):
    """Same backbone for every cell; parallel encoders get bumped capacity."""
    _enc_dict = _ENCODERS[encoder_key]
    # Default arflow capacity bump — but skip axes already set in the encoder key.
    if _enc_dict.get("encoder_type") == "arflow":
        caps = {}
        if "arflow_channels" not in _enc_dict:
            caps["arflow_channels"] = 128
        if "arflow_causal_layers" not in _enc_dict:
            caps["arflow_causal_layers"] = 4
    else:
        caps = {}
    _consumed_here = (
        "channels", "nheads", "diffusion_layers", "summary_layers"
    )
    enc = {k: v for k, v in _ENCODERS[encoder_key].items()
           if k not in _consumed_here}
    return GluonModel(
        data_dim=data_dim,
        latent_dim=_LATENT_DIM,
        j=j,
        T_max=_T,
        channels=_ENCODERS[encoder_key].get("channels", 48),
        nheads=_ENCODERS[encoder_key].get("nheads", 2),
        summary_layers=_ENCODERS[encoder_key].get("summary_layers", 1),
        diffusion_layers=_ENCODERS[encoder_key].get("diffusion_layers", 3),
        num_steps=64,
        grad_checkpoint=False,
        **enc,
        **caps,
    )


def _phase2_cell(encoder_key: str, dataset_key: str, j: int = 1):
    """Single-phase ELBO cell; training hyperparameters may be SWEPT."""
    data, data_dim = _DATASETS[dataset_key]
    return experiment(
        data=data,
        model=_model(encoder_key, data_dim, j),
        hparams=dataclasses.replace(GluonHparams, batch_size=32),
        training=_training(steps=10000),
        eval=Eval(
            metrics=["crps_sum"],
            split="val",
            num_samples=100,
            T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="crps_sum", source="json"),
    )


def _phase1_cell(encoder_key: str, dataset_key: str, j: int = 1):
    """Pure-AE capacity probe (recon-only budget); objective = recon."""
    data, data_dim = _DATASETS[dataset_key]
    return experiment(
        data=data,
        model=_model(encoder_key, data_dim, j),
        hparams=dataclasses.replace(GluonHparams, batch_size=32),
        training=_training(steps=3000, checkpoint_every=1000),
        eval=Eval(
            metrics=["recon_mse"],
            split="val",
            num_samples=1,
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
    "gaussian",
    "gaussian_local",
    "iaf",
    "det",
    "o1_flow",
    "fb_mf",
    "fb_flow",
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

# j-sweep: top 4 configs at j=6,4,2,1 (j=10 already registered above).
for _j in (6, 4, 2, 1):
    for _enc in (
        "identity_csdilike_ais",
        "identity_csdilike_ais_persist",
        "identity_csdilike",
        "identity_csdi",
    ):
        experiment_store(
            _phase2_cell(_enc, "nlblmv", j=_j),
            name=f"h2h__{_enc}__nlblmv__j{_j}",
        )

# Bumped capacity (704K): 96ch / 4 layers / 4 heads at j=4.
experiment_store(
    _phase2_cell("identity_csdilike_ais_big", "nlblmv", j=4),
    name="h2h__identity_csdilike_ais_big__nlblmv__j4",
)

# 20K steps, LR=8e-4, bumped capacity.
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("identity_csdilike_ais_big", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams, batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["crps_sum"], split="val", num_samples=100,
            T_split=_T_SPLIT, output_filename="metrics.json",
        ),
        objective=Objective(metric="crps_sum", source="json"),
        # ESM vs DSM score-net gradient/loss variance, at both k-sampling modes
        # this cell actually trains under (uniform reference + its own
        # adaptive_is). Defaults: R=128 replicas x 3 seeds x 4 cells, plus the
        # force_per_k sweep over all 64 diffusion steps for the per-tau curves.
        variance=Probe(),
    ),
    name="h2h__identity_csdilike_ais_big_20k__nlblmv__j4",
)

# Positive-control sibling: identical to h2h__identity_csdilike_ais_big_20k but
# with a learned gaussian encoder (σ_t ~ O(1) early → the LSGM ESM/DSM
# blowup regime the variance probe wants to see).
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("gaussian_csdilike_ais_big", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams, batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["crps_sum"], split="val", num_samples=100,
            T_split=_T_SPLIT, output_filename="metrics.json",
        ),
        objective=Objective(metric="crps_sum", source="json"),
        variance=Probe(),
    ),
    name="h2h__gaussian_csdilike_ais_big_20k__nlblmv__j4",
)

# Trial 2 capacity reduction: shrink the score-net at fixed recipe. Same 20K
# budget as _big; goal is the smallest cell whose val CRPS stays on-par with
# the big cell. No variance probe on these — pure fit runs.
for _cap_key in (
    "identity_csdilike_ais",           # 80/3/2  ~409K  ("med")
    "identity_csdilike_ais_small",     # 64/3/2  ~296K
    "identity_csdilike_ais_small_h4",  # 64/3/4  ~296K (heads split; conv-proj same)
    "identity_csdilike_ais_tiny",      # 48/2/2  ~147K
):
    if _cap_key == "identity_csdilike_ais":
        _cap_short = "med"
    elif _cap_key == "identity_csdilike_ais_small_h4":
        _cap_short = "small_h4"
    else:
        _cap_short = _cap_key.split("_")[-1]
    experiment_store(
        experiment(
            data=NonlinBimodalLiftMV,
            model=_model(_cap_key, 8, j=4),
            hparams=dataclasses.replace(
                GluonHparams, batch_size=32,
                enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            ),
            training=_training(steps=20000, checkpoint_every=4000),
            eval=Eval(
                metrics=["obs_space_jsd", "crps_sum"], split="val",
                num_samples=100, T_split=_T_SPLIT,
                output_filename="metrics.json",
            ),
            objective=Objective(metric="obs_space_jsd_mean", source="json"),
        ),
        name=f"h2h__identity_csdilike_ais_{_cap_short}_20k__nlblmv__j4",
    )


# Gaussian encoder + big transition (96/4/2) + obs-space JSD as primary. Two
# cells: (a) constant LR (matches Trial 1 identity_big), (b) resolver-default
# LR schedule (φθ warmup 0 + cosine to 5%; ψ warmup 1250 + cosine to 20%).
# Both cells share the same λ-KL cosine ramp: floor 1e-5 → 1.0 over the first
# 5K steps (25%), then held at 1.0 through 20K.
_GAUSSIAN_BIG_JSD_LAMBDA_RAMP = LambdaRampConf(
    start=1e-5, end=1.0, steps=5000, delay=0,
)

for _lr_variant in ("matched", "lrsched"):
    if _lr_variant == "matched":
        _lr_schedule = None
        _name_suffix = ""
    else:
        # Fully-None group; resolve_lr_schedule_defaults fills from the default
        # table at train time using λ_end = 5000 (delay 0 + steps 5000).
        _lr_schedule = LrScheduleGroupConf(
            phith=LrScheduleConf(), psi=LrScheduleConf(),
        )
        _name_suffix = "_lrsched"
    experiment_store(
        experiment(
            data=NonlinBimodalLiftMV,
            model=_model("gaussian_csdilike_ais_big", 8, j=4),
            hparams=dataclasses.replace(
                GluonHparams,
                batch_size=32,
                enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
                lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP,
                lr_schedule=_lr_schedule,
            ),
            training=_training(steps=20000, checkpoint_every=4000),
            eval=Eval(
                metrics=["obs_space_jsd", "crps_sum"], split="val",
                num_samples=100, T_split=_T_SPLIT,
                output_filename="metrics.json",
            ),
            objective=Objective(metric="obs_space_jsd_mean", source="json"),
        ),
        name=f"h2h__gaussian_csdilike_ais_big_20k_gjsd{_name_suffix}__nlblmv__j4",
    )


# Low+slow λ warmup: floor 1e-7 (2 orders below the 1e-5 default), ramp over
# the first 15K steps (75% of training, 3× longer than the gjsd cells). Same
# gaussian_csdilike_ais_big model, batch=32, matched LR (const 8e-4). Two
# cells differ ONLY on ``use_split_loss``: split=False vs True.
_GAUSSIAN_BIG_JSD_LOWSLOW_LAMBDA_RAMP = LambdaRampConf(
    start=1e-7, end=1.0, steps=15000, delay=0,
)

for _split_variant in (False, True):
    _suffix = "_split" if _split_variant else ""
    experiment_store(
        experiment(
            data=NonlinBimodalLiftMV,
            model=_model("gaussian_csdilike_ais_big", 8, j=4),
            hparams=dataclasses.replace(
                GluonHparams,
                batch_size=32,
                enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
                lambda_ramp=_GAUSSIAN_BIG_JSD_LOWSLOW_LAMBDA_RAMP,
                use_split_loss=_split_variant,
            ),
            training=_training(steps=20000, checkpoint_every=4000),
            eval=Eval(
                metrics=["obs_space_jsd", "crps_sum"], split="val",
                num_samples=100, T_split=_T_SPLIT,
                output_filename="metrics.json",
            ),
            objective=Objective(metric="obs_space_jsd_mean", source="json"),
        ),
        name=(
            "h2h__gaussian_csdilike_ais_big_20k_gjsd_lowslow"
            f"{_suffix}__nlblmv__j4"
        ),
    )


# Same low+slow λ recipe (1e-7 → 1.0 over 15K) + gaussian encoder, but the
# transition slot is the LITERAL vendored ermongroup CSDI (transition_type
# = "csdi") instead of our DiffusionTransition. Reads: does the frame-JSD
# gap survive a decisively-good obs-space transition, or was our transition
# the bottleneck? j=4 (matches Trial 4 for controlled comparison); memory
# note that ``j == HIST`` reproduces standalone CSDI, so this is NOT the
# faithful-CSDI comparison — it's the transition-swap comparison.
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("gaussian_csdi", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_GAUSSIAN_BIG_JSD_LOWSLOW_LAMBDA_RAMP,
            use_split_loss=False,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__gaussian_csdi_20k_gjsd_lowslow__nlblmv__j4",
)


# Wide-encoder sibling of gaussian_csdilike_ais_big_20k_gjsd: same fast λ ramp
# (1e-5 → 1.0 over 5K, matches the best gaussian cell so far), same transition,
# same LRs/batch. Only the encoder is bigger: encoder_hidden_dim=64 → encoder
# param count 17K → 118K. Isolates encoder-capacity effect on obs-space JSD.
_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST = LambdaRampConf(
    start=1e-5, end=1.0, steps=5000, delay=0,
)

experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("gaussian_csdilike_ais_big_wideenc", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__gaussian_csdilike_ais_big_wideenc_20k_gjsd__nlblmv__j4",
)


# Wide-encoder + resolver-default LR schedule + split-loss training. Same
# fast λ ramp (1e-5 → 1.0 over 5K) as the plain wideenc cell. Isolates the
# combined effect of LR-sched + split-loss on top of the encoder-capacity fix.
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("gaussian_csdilike_ais_big_wideenc", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST,
            lr_schedule=LrScheduleGroupConf(
                phith=LrScheduleConf(), psi=LrScheduleConf(),
            ),
            use_split_loss=True,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__gaussian_csdilike_ais_big_wideenc_20k_gjsd_lrsched_split__nlblmv__j4",
)


# ArFlow-encoder counterpart to gaussian_wideenc_lrsched_split, matched by
# encoder param count (~110K vs the gaussian's ~118K). Same LR sched + split
# + fast λ ramp. Persistence baseline (arflow requires it) instead of zero.
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("iaf_csdilike_ais_big_matchedenc", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST,
            lr_schedule=LrScheduleGroupConf(
                phith=LrScheduleConf(), psi=LrScheduleConf(),
            ),
            use_split_loss=True,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__iaf_csdilike_ais_big_matchedenc_20k_gjsd_lrsched_split__nlblmv__j4",
)


# Sibling arflow variants at matched encoder capacity + same lrsched+split
# recipe as iaf_matchedenc. Fills out the parallel-encoder row of the table.
for _arflow_variant in ("det", "o1_flow", "fb_mf", "fb_flow"):
    experiment_store(
        experiment(
            data=NonlinBimodalLiftMV,
            model=_model(f"{_arflow_variant}_csdilike_ais_big_matchedenc", 8, j=4),
            hparams=dataclasses.replace(
                GluonHparams,
                batch_size=32,
                enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
                lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST,
                lr_schedule=LrScheduleGroupConf(
                    phith=LrScheduleConf(), psi=LrScheduleConf(),
                ),
                use_split_loss=True,
            ),
            training=_training(steps=20000, checkpoint_every=4000),
            eval=Eval(
                metrics=["obs_space_jsd", "crps_sum"], split="val",
                num_samples=100, T_split=_T_SPLIT,
                output_filename="metrics.json",
            ),
            objective=Objective(metric="obs_space_jsd_mean", source="json"),
        ),
        name=(
            f"h2h__{_arflow_variant}_csdilike_ais_big_matchedenc_20k_"
            f"gjsd_lrsched_split__nlblmv__j4"
        ),
    )


# Raw CSDI, fair-configured: identity encoder + literal ermongroup CSDI in
# the transition slot at CSDI's native defaults (channels=64, layers=4,
# nheads=8, num_steps=50). j=T_SPLIT=24 so CSDI sees the full history it
# expects (memory: "with j == HIST this reproduces the 58% standalone
# CSDI"). Data pipeline is ours (nlblmv, our eval). No split/lrsched —
# identity encoder has 0 params so φθ optimizer would be empty. Same fast
# λ ramp for compute-budget parity with other cells.
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("identity_csdi", 8, j=_T_SPLIT),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__identity_csdi_raw_20k_gjsd__nlblmv__j24",
)


# CSDI transition within a learned gaussian encoder. Same setup as
# gaussian_matched (0.1814) but transition swapped: DiffusionTransition →
# CSDITransition. Plain (17K) gaussian encoder; no split/lrsched. j=4 to
# stay comparable with the wideenc row.
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("gaussian_csdi", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__gaussian_csdi_20k_gjsd__nlblmv__j4",
)


# fb_mf at 8× encoder capacity (944K vs 118K). Isolates whether arflow's
# poor showing at ~110K was capacity-limited or architectural.
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("fb_mf_csdilike_ais_big_bigenc8x", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST,
            lr_schedule=LrScheduleGroupConf(
                phith=LrScheduleConf(), psi=LrScheduleConf(),
            ),
            use_split_loss=True,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__fb_mf_csdilike_ais_big_bigenc8x_20k_gjsd_lrsched_split__nlblmv__j4",
)


# fb_mf 8×-encoder with λ_end lowered 1.0 → 0.1. Gives the encoder more
# freedom to fit recon (rec was 248 at λ=1.0). Same shape otherwise
# (1e-5 → 0.1 cosine over 5K then held at 0.1). Split loss + lrsched kept.
_LAMBDA_RAMP_FAST_L01 = LambdaRampConf(
    start=1e-5, end=0.1, steps=5000, delay=0,
)

experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("fb_mf_csdilike_ais_big_bigenc8x", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_LAMBDA_RAMP_FAST_L01,
            lr_schedule=LrScheduleGroupConf(
                phith=LrScheduleConf(), psi=LrScheduleConf(),
            ),
            use_split_loss=True,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__fb_mf_csdilike_ais_big_bigenc8x_20k_gjsd_lrsched_split_l01__nlblmv__j4",
)


# Full-conv fb_mf_bigenc8x: score-net time+feature mixers and arflow causal_net
# backbone all switched from transformer to conv. Params 1.66M → 1.15M (−31%).
# Same λ ramp (1e-5→1.0 over 5K), split, lrsched as trial 8 for direct comparison.
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("fb_mf_csdilike_ais_big_bigenc8x_conv", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST,
            lr_schedule=LrScheduleGroupConf(
                phith=LrScheduleConf(), psi=LrScheduleConf(),
            ),
            use_split_loss=True,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__fb_mf_csdilike_ais_big_bigenc8x_conv_20k_gjsd_lrsched_split__nlblmv__j4",
)


# Full-conv score-net sibling of gaussian_wideenc_lrsched_split (JSD=0.1569).
# Same wideenc (118K) gaussian encoder, same λ ramp + lrsched + split. Only
# the score-net time+feature mixers change: transformer → conv. Fut_summary
# stays transformer (no conv variant available). Tests whether conv score-net
# at reduced params can match / beat the transformer wideenc baseline.
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("gaussian_csdilike_ais_big_wideenc_conv", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=256,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST,
            lr_schedule=LrScheduleGroupConf(
                phith=LrScheduleConf(), psi=LrScheduleConf(),
            ),
            use_split_loss=True,
        ),
        training=_training(steps=5000, checkpoint_every=1000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__gaussian_csdilike_ais_big_wideenc_conv_20k_gjsd_lrsched_split__nlblmv__j4",
)


# Sibling of gaussian_wideenc_conv but with a GRU fut_summary instead of the
# transformer. Fully "conv+rnn" (no attention anywhere in the model). Tests
# whether the fut_summary transformer was doing structural work.
experiment_store(
    experiment(
        data=NonlinBimodalLiftMV,
        model=_model("gaussian_csdilike_ais_big_wideenc_conv_gru", 8, j=4),
        hparams=dataclasses.replace(
            GluonHparams,
            batch_size=32,
            enc_lr=8e-4, dec_lr=8e-4, trans_lr=8e-4,
            lambda_ramp=_GAUSSIAN_BIG_JSD_LAMBDA_RAMP_FAST,
            lr_schedule=LrScheduleGroupConf(
                phith=LrScheduleConf(), psi=LrScheduleConf(),
            ),
            use_split_loss=True,
        ),
        training=_training(steps=20000, checkpoint_every=4000),
        eval=Eval(
            metrics=["obs_space_jsd", "crps_sum"], split="val",
            num_samples=100, T_split=_T_SPLIT,
            output_filename="metrics.json",
        ),
        objective=Objective(metric="obs_space_jsd_mean", source="json"),
    ),
    name="h2h__gaussian_csdilike_ais_big_wideenc_conv_gru_20k_gjsd_lrsched_split__nlblmv__j4",
)
