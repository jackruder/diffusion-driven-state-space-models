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

from ddssm.model.dssd import DDSSM_base
from ddssm.nn.diffnets import (
    CSDIUnet,
    ContextProducer,
    FeatureMixerConfig,
    ResidualBlockConfig,
    DiffResidualBlockConfig,
)
from ddssm.nn.futsum import TransformerFutureSummary
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder, ARFlowEncoder
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.experiment.stores import model_store
from ddssm.model.centering.baselines import PersistenceBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)
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
    # Diffusion IS mode for noise-level selection: "lsgm_is" (default), "uniform",
    # or "esm_is" (p_k ∝ s/(1+s²)², peaks at σ̃≈0.6).
    k_sampling_mode: str = "lsgm_is",
    # Floor on the IS density p_k. The default 1e-12 lets esm_is assign p_k~1e-7 to
    # extreme noise levels, so the unbiased 1/(K·p_k) reweighting can hit ~2e5× and
    # explode when such a level is (rarely) sampled. Raise to ~1e-3 to bound the
    # weight (still unbiased — sampling & reweighting share the floored p_k).
    pk_floor: float = 1e-12,
    # "gaussian" = the settled sequential encoder; "arflow" = the parallel
    # AR-flow-on-noise drop-in (opt-in; flip the default only past the slice-9 gate).
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
    # ARFlow encoder capacity, DECOUPLED from the transition's `channels` so the encoder
    # gets more punch without inflating the diffusion (vs the already-swept gaussian).
    arflow_channels: int | None = None,  # None → = channels; must be ÷ nheads
    arflow_causal_layers: int = 2,
    grad_checkpoint: bool = True,
) -> DDSSM_base:
    """Build a gluonts-forecast DDSSM (persistence-pinned, additive encoder)."""
    # Single width rule: 2×latent for summary + encoder + decoder hidden.
    width = 2 * latent_dim
    emb_time_dim = 0  # time-conditioning OFF (v1); collapses the time-cond ops out.

    # Shared persistence baseline (param-free → pinned both stages); the SAME
    # instance threads stage-1 → stage-2 so the handoff snapshot is consistent.
    baseline = PersistenceBaseline(
        latent_dim=latent_dim, j=j, hidden_dim=width, n_layers=2,
    )
    aux_posterior = AuxPosterior(
        latent_dim=latent_dim, j=j, hidden_dim=width, n_layers=2,
    )
    sigma_data = SigmaDataBuffer(
        T_max=T_max, tracking_mode=tracking_mode, init_value=1.0,
        ema_decay=sigma_data_ema_decay,
    )

    stage1_transition = BaselineGaussianTransition(
        baseline=baseline, latent_dim=latent_dim, j=j, emb_time_dim=emb_time_dim,
    )

    unet = partial(
        CSDIUnet,
        channels=channels,
        n_layers=diffusion_layers,
        embedding_dim=embedding_dim,
        residual_block=DiffResidualBlockConfig(
            # dropout=0.0: the score-net is gradient-checkpointed (deterministic
            # recompute) and a denoiser shouldn't add noise to its ESM target.
            feature=FeatureMixerConfig(
                type="transformer", nheads=nheads, n_layers=1, dropout=0.0
            )
        ),
    )
    schedule = DiffusionScheduleConfig(
        S_k=1, k_chunk=1, num_steps=num_steps, k_sampling_mode=k_sampling_mode,
        time_chunk_size=time_chunk, pk_floor=pk_floor,
    )
    stage2_transition = DiffusionTransition(
        baseline=baseline, latent_dim=latent_dim, j=j, emb_time_dim=emb_time_dim,
        T_max=T_max, unet=unet, schedule=schedule, grad_checkpoint=grad_checkpoint,
    )

    fut_summary = partial(
        TransformerFutureSummary, summary_dim=width, nheads=nheads,
        transformer_layers=summary_layers,
    )
    if encoder_type == "gaussian":
        encoder = GaussianEncoder(
            data_dim=data_dim, latent_dim=latent_dim, j=j, emb_time_dim=emb_time_dim,
            use_mask=False, hidden_dim=width, mu_mode="additive",
            fut_summary=fut_summary, grad_checkpoint=grad_checkpoint,
        )
    elif encoder_type == "arflow":
        # Head logvar clamp pinned to the DDSSM_base default [-7, 7] (dssd.py:130-131)
        # so the in-encoder logq matches the KL's re-clamped logvars (dssd.py:697).
        encoder = ARFlowEncoder(
            data_dim=data_dim, latent_dim=latent_dim, j=j, emb_time_dim=emb_time_dim,
            use_mask=False, hidden_dim=width, fut_summary=fut_summary,
            channels=arflow_channels if arflow_channels is not None else channels,
            causal_layers=arflow_causal_layers, nheads=nheads, backbone="transformer",
            clamp_logvar_min=-7.0, clamp_logvar_max=7.0,
            init_logvar_bias=arflow_init_logvar_bias,
            stochastic_state=arflow_stochastic_state, grad_checkpoint=grad_checkpoint,
        )
    else:
        raise ValueError(
            f"encoder_type must be 'gaussian' or 'arflow'; got {encoder_type!r}"
        )
    decoder = GaussianDecoder(
        data_dim=data_dim, latent_dim=latent_dim, j=j, emb_time_dim=emb_time_dim,
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
