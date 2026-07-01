"""Composed DDSSM model factory for the gluonts_forecast benchmark family.

Sized for REAL data (D=137..2000, long L1, latent swept to 512), so it does NOT
reuse ``init_centering``'s synthetic-tuned ``SmokeModel``. Deviations (all
settled in the architecture drilling — see the plan):

* Future summary = ``TransformerFutureSummary`` (attends across the long L1=168
  history) — not the GRU.
* Score-net feature mixer = transformer (``nheads=8``) — not conv. ADR-0003
  sized attention out for *tiny* synthetic latents; here d goes to 512.
* ``channels=64`` FIXED, ``n_layers=4``, ``embedding_dim=128`` (CSDI parity) —
  NOT the synthetic ``16×latent`` rule.
* Single width rule: ``summary_dim = encoder hidden = decoder hidden = 2×latent``
  (uncapped). ``2×latent`` is ÷8 for the whole latent grid, so the summary heads
  divide cleanly.
* Gradient checkpointing on the score-net + future-summary so latent=512 fits 80 GB.
* baseline = persistence (pinned, param-free); Gaussian decoder; time-cond OFF.
"""

from __future__ import annotations

from functools import partial

from hydra_zen import builds

from ddssm.nn.futsum import IdentityFutureSummary, TransformerFutureSummary
from ddssm.model.dssd import DDSSM_base
from ddssm.nn.diffnets import (
    CSDIUnet,
    ContextProducer,
    TimeMixerConfig,
    FeatureMixerConfig,
    ResidualBlockConfig,
    DiffResidualBlockConfig,
)
from ddssm.model.decoder import GaussianDecoder, IdentityDecoder
from ddssm.model.encoder import ARFlowEncoder, GaussianEncoder, IdentityEncoder
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.experiment.stores import model_store
from ddssm.model.centering.baselines import ZeroBaseline, PersistenceBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)
from ddssm.model.transitions.csdi_transition import CSDITransition
from ddssm.model.transitions.baseline_gaussian import BaselineGaussianTransition


def build_gluonts_model(
    *,
    data_dim: int,
    latent_dim: int = 64,
    j: int = 1,
    T_max: int = 192,
    channels: int = 64,
    diffusion_layers: int = 4,
    embedding_dim: int = 128,
    num_steps: int = 128,
    nheads: int = 8,
    summary_layers: int = 2,
    k_sampling_mode: str = "adaptive_is",
    # Floor on the IS density p_k. The default 1e-12 lets esm_is assign p_k~1e-7 to
    # extreme noise levels, so the unbiased 1/(K·p_k) reweighting can hit ~2e5× and
    # explode when such a level is (rarely) sampled. Raise to ~1e-3 to bound the
    # weight (still unbiased — sampling & reweighting share the floored p_k).
    pk_floor: float = 1e-12,
    # Stage-2 transition type: "diffusion" (default, our CSDI-style denoiser),
    # "gaussian" (BaselineGaussianTransition — unimodal per-step prior; a JSD/CRPS
    # calibration baseline for the diffusion transition's expressive value), or
    # "csdi" (the *literal* vendored ermongroup CSDI dropped into the transition
    # slot — DDPM ε-MSE + ancestral sampler + masked conditioning; pairs with
    # encoder_type="identity" + j==HIST to reproduce the 58% standalone baseline
    # inside the DDSSM pipeline → a clean indictment/exoneration of our own code).
    transition_type: str = "diffusion",
    # "csdi"-transition capacity knobs (default = the 58% standalone nlblmv config:
    # 64ch / 4 layers / 8 heads / 50 ancestral steps). Independent of the model's
    # `channels`/`diffusion_layers`/`nheads`, which size the "diffusion" branch.
    csdi_channels: int = 64,
    csdi_layers: int = 4,
    csdi_nheads: int = 8,
    csdi_num_steps: int = 50,
    # "gaussian" = the settled sequential encoder (smoothing q(z_t|x_{t:T}) via a
    # Transformer future-summary); "gaussian_local" = same sequential encoder but a
    # LOCAL future-summary (h_t = Linear(x_t), identity time-mixer) → filtering
    # q(z_t|x_t), biased toward a clean near-identity frame at latent_dim==data_dim;
    # "arflow" = the parallel AR-flow-on-noise drop-in (opt-in).
    encoder_type: str = "gaussian",
    # Timesteps per score-net call in the ESM loss. With checkpointing the peak
    # is one chunk's d²-attention (~time_chunk·B·nheads·latent²), so this trades
    # training speed against memory — 16 keeps the worst corner (latent=512,
    # batch=128) ~34 GB; the pilot's compute-budget step tunes it per dataset.
    time_chunk: int = 16,
    tracking_mode: str = "per_t",
    sigma_data_ema_decay: float = 0.997,
    # ARFlow-only: σ init via the logσ²-head bias (σ=exp(½·bias)). 0 → σ=1 (init at prior).
    arflow_init_logvar_bias: float = 0.0,
    # ARFlow-only: True → IAF (conditioner sees the noise history; μ,σ condition on the
    # realized path). False → deterministic causal encoder (μ,σ = f(h), z_hist amortized).
    arflow_stochastic_state: bool = True,
    # ARFlow-only: data context the conditioner sees. "none" → backward summary b_t;
    # "fwd_data" → [f_t, b_t] (adds a forward-causal data message f_t = F_ϕ(x_{1:t}));
    # "fwd_summary" → o_t, a forward-causal pass over b (deterministic analog of z).
    arflow_forward_message: str = "none",
    # ARFlow encoder capacity, DECOUPLED from the transition's `channels` so the encoder
    # gets more punch without inflating the diffusion (vs the already-swept gaussian).
    arflow_channels: int | None = None,  # None → = channels; must be ÷ nheads
    arflow_causal_layers: int = 2,
    grad_checkpoint: bool = True,
    # Centering baseline for the diffusion/gaussian transition: "persistence"
    # (μ_p = last latent) or "zero" (μ_p ≡ 0). Persistence centers on the *last*
    # latent, which under a bimodal data conditional sits on one mode → the
    # transition only has to model the residual to that mode; "zero" makes the
    # transition model the full (bimodal) conditional directly.
    baseline_type: str = "persistence",
    # Per-channel feature embedding width for the "diffusion" transition's side-info.
    # 0 → off; >0 → CSDI-style learned per-channel embedding (decoupled from
    # time-conditioning, which stays off). Default 16 = literal CSDI's featureemb;
    # the kitchen-sink attribution made this the single dominant denoiser axis.
    emb_feature_dim: int = 16,
    # "diffusion"-transition time-axis mixer: "auto" (transformer for j>4, else conv),
    # "conv" (3-tap causal conv), or "transformer" (non-causal RoPE attention over the
    # j+1 window, CSDI-style). Transformer is locked in for non-trivial history.
    time_mixer: str = "auto",
    # "diffusion"-transition sampler: "edm" (default; Karras 2022 Heun + optional
    # stochastic churn, deterministic at edm_s_churn=0) or "pf_ode" (legacy
    # deterministic VP probability-flow Euler). Churn 16 is the locked-in default.
    diffusion_sampler: str = "edm",
    edm_s_churn: float = 16.0,
    edm_s_noise: float = 1.0,
    edm_rho: float = 7.0,
    # "csdi"-transition time embedding width (the vendored ermongroup CSDI). 0 turns
    # CSDI's time-conditioning OFF to match our emb_time_dim==0 (embedding parity).
    csdi_timeemb: int = 128,
) -> DDSSM_base:
    """Build a gluonts-forecast DDSSM (persistence-pinned, additive encoder)."""
    # Single width rule: 2×latent for summary + encoder + decoder hidden.
    width = 2 * latent_dim
    emb_time_dim = 0  # time-conditioning OFF (v1); collapses the time-cond ops out.

    # "auto" time-mixer: a j+1 window only wide enough to need attention (j>4) gets the
    # transformer; short windows keep the cheap 3-tap conv. Explicit values pass through.
    if time_mixer == "auto":
        time_mixer = "transformer" if j > 4 else "conv"

    # Shared centering baseline (param-free → pinned both stages); the SAME instance
    # threads stage-1 → stage-2 so the handoff snapshot is consistent.
    if baseline_type == "persistence":
        baseline = PersistenceBaseline(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=width,
            n_layers=2,
        )
    elif baseline_type == "zero":
        baseline = ZeroBaseline(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=width,
            n_layers=2,
        )
    else:
        raise ValueError(
            f"baseline_type must be 'persistence' or 'zero'; got {baseline_type!r}"
        )
    aux_posterior = AuxPosterior(
        latent_dim=latent_dim,
        j=j,
        hidden_dim=width,
        n_layers=2,
    )
    sigma_data = SigmaDataBuffer(
        T_max=T_max,
        tracking_mode=tracking_mode,
        init_value=1.0,
        ema_decay=sigma_data_ema_decay,
    )

    stage1_transition = BaselineGaussianTransition(
        baseline=baseline,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
    )

    if transition_type == "diffusion":
        unet = partial(
            CSDIUnet,
            channels=channels,
            n_layers=diffusion_layers,
            embedding_dim=embedding_dim,
            residual_block=DiffResidualBlockConfig(
                # dropout=0.0: the score-net is gradient-checkpointed (deterministic
                # recompute) and a denoiser shouldn't add noise to its ESM target.
                time=TimeMixerConfig(
                    type=time_mixer, nheads=nheads, n_layers=1, dropout=0.0
                ),
                feature=FeatureMixerConfig(
                    type="transformer", nheads=nheads, n_layers=1, dropout=0.0
                ),
            ),
        )
        schedule = DiffusionScheduleConfig(
            S_k=1, k_chunk=1, num_steps=num_steps, k_sampling_mode=k_sampling_mode,
            time_chunk_size=time_chunk, pk_floor=pk_floor,
        )
        stage2_transition = DiffusionTransition(
            baseline=baseline,
            latent_dim=latent_dim,
            j=j,
            emb_time_dim=emb_time_dim,
            T_max=T_max,
            unet=unet,
            schedule=schedule,
            grad_checkpoint=grad_checkpoint,
            emb_feature_dim=emb_feature_dim,
            sampler=diffusion_sampler,
            edm_s_churn=edm_s_churn,
            edm_s_noise=edm_s_noise,
            edm_rho=edm_rho,
        )
    elif transition_type == "gaussian":
        # JSD/CRPS calibration baseline: unimodal Gaussian transition prior.
        stage2_transition = BaselineGaussianTransition(
            baseline=baseline,
            latent_dim=latent_dim,
            j=j,
            emb_time_dim=emb_time_dim,
        )
    elif transition_type == "csdi":
        # Literal ermongroup CSDI in the transition slot (its own capacity, its own
        # quad β-schedule + ancestral sampler). Ignores the persistence baseline /
        # σ_data buffer — it is a self-contained conditional diffusion model.
        stage2_transition = CSDITransition(
            latent_dim=latent_dim,
            j=j,
            emb_time_dim=emb_time_dim,
            T_max=T_max,
            channels=csdi_channels,
            layers=csdi_layers,
            nheads=csdi_nheads,
            num_steps=csdi_num_steps,
            timeemb=csdi_timeemb,
        )
    else:
        raise ValueError(
            "transition_type must be 'diffusion', 'gaussian', or 'csdi'; got "
            f"{transition_type!r}"
        )

    fut_summary = partial(
        TransformerFutureSummary,
        summary_dim=width,
        nheads=nheads,
        transformer_layers=summary_layers,
    )
    # Forward-causal twin of the future summary (no time-flip) → the f_t message
    # for ARFlow's "fwd_data" context. Same width/heads/depth as the backward b_t.
    fut_summary_fwd = partial(
        TransformerFutureSummary,
        summary_dim=width,
        nheads=nheads,
        transformer_layers=summary_layers,
        reverse_time=False,
    )
    # Local twin of the future summary: h_t = Linear(x_t), no time mixing. Drives the
    # filtering "gaussian_local" encoder (matched inverse for the per-timestep lift).
    fut_summary_local = partial(IdentityFutureSummary, summary_dim=width)
    if encoder_type in ("gaussian", "gaussian_local"):
        encoder = GaussianEncoder(
            data_dim=data_dim,
            latent_dim=latent_dim,
            j=j,
            emb_time_dim=emb_time_dim,
            use_mask=False,
            hidden_dim=width,
            mu_mode="additive",
            fut_summary=(
                fut_summary_local if encoder_type == "gaussian_local" else fut_summary
            ),
            grad_checkpoint=grad_checkpoint,
        )
    elif encoder_type == "arflow":
        # Head logvar clamp pinned to the DDSSM_base default [-7, 7] (dssd.py:130-131)
        # so the in-encoder logq matches the KL's re-clamped logvars (dssd.py:697).
        encoder = ARFlowEncoder(
            data_dim=data_dim,
            latent_dim=latent_dim,
            j=j,
            emb_time_dim=emb_time_dim,
            use_mask=False,
            hidden_dim=width,
            fut_summary=fut_summary,
            channels=arflow_channels if arflow_channels is not None else channels,
            causal_layers=arflow_causal_layers,
            nheads=nheads,
            backbone="transformer",
            init_logvar_bias=arflow_init_logvar_bias,
            stochastic_state=arflow_stochastic_state,
            forward_message=arflow_forward_message,
            fwd_summary=fut_summary_fwd,
            fwd_layers=summary_layers,
            grad_checkpoint=grad_checkpoint,
        )
    elif encoder_type == "identity":
        # Pinned z_t = x_t (requires latent_dim == data_dim): the latent frame IS
        # the observation, so the diffusion transition denoises in OBSERVATION space
        # — a CSDI-style obs-space model inside the DDSSM pipeline. Isolates whether
        # the latent pipeline (not the transition) is the bottleneck vs obs-space CSDI.
        encoder = IdentityEncoder(
            data_dim=data_dim,
            latent_dim=latent_dim,
            j=j,
            emb_time_dim=emb_time_dim,
        )
    else:
        raise ValueError(
            "encoder_type must be 'gaussian', 'gaussian_local', 'arflow', or "
            f"'identity'; got {encoder_type!r}"
        )
    if encoder_type == "identity":
        # Matched identity emission x_t = z_t (fixed σ_x); predictive spread comes
        # from the transition, not a learnable decoder.
        decoder = IdentityDecoder(
            latent_dim=latent_dim,
            data_dim=data_dim,
            j=j,
            emb_time_dim=emb_time_dim,
        )
    else:
        decoder = GaussianDecoder(
            data_dim=data_dim,
            latent_dim=latent_dim,
            j=j,
            emb_time_dim=emb_time_dim,
            hidden_dim=width,
            # Deterministic decoder (dropout=0): required so the time-chunked recon is
            # batch-invariant and its checkpoint (preserve_rng_state=False) is exact —
            # the ContextProducer default carries dropout=0.1.
            context=partial(
                ContextProducer,
                channels=8,
                num_layers=2,
                residual_block=ResidualBlockConfig(
                    feature=FeatureMixerConfig(nheads=1, n_layers=2, dropout=0.0)
                ),
            ),
        )

    return DDSSM_base(
        encoder=encoder,
        decoder=decoder,
        transition=stage2_transition,
        j=j,
        data_dim=data_dim,
        latent_dim=latent_dim,
        emb_time_dim=emb_time_dim,
        use_observation_mask=False,
        aux_posterior=aux_posterior,
        baseline=baseline,
        baseline_anchor=None,  # populated by the handoff
        baseline_mode="pinned",
        sigma_data=sigma_data,
        stage1_transition=stage1_transition,
        # Decode the T-window in time chunks (batched, checkpointed) instead of a
        # 192× Python loop — the recon loop is the other launch-bound half of the
        # per-step cost. Same chunk knob as the diffusion ESM loss.
        recon_time_chunk=time_chunk,
        recon_grad_checkpoint=grad_checkpoint,
    )


# hydra-zen wrapper so the preset / Optuna sweep can override fields by name.
GluonModel = builds(build_gluonts_model, populate_full_signature=True)

model_store(GluonModel, name="gluonts_forecast")

__all__ = ["GluonModel", "build_gluonts_model"]
