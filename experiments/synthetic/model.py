"""Composed model configs for the synthetic-data family.

One file holds every shape + architectural knob + composed
:class:`DDSSM` for this family. Reads top-to-bottom:

1. **Family-shared arch primitives** — mixers, residual blocks,
   context producer, Gaussian head, future summary, U-Net flavours,
   noise schedule. Reused by every shape below.
2. **Per-shape namespace classes** (``Small1D``, ``Robot2D``,
   ``ProbeMedium``). A class body is just an executable namespace
   with forward lookup — ``data_dim = 1`` at the top of the class
   propagates into every ``Encoder/Decoder/ZInit/...`` call beneath
   it. Tweak one constant and every downstream subconfig follows.
3. **Store registrations** at the bottom, exposing
   encoder/decoder/z_init/transition/unet/schedule/model handles for
   the Hydra CLI (``encoder=small_1d``, ``model=small_gauss``, …).

To change a knob, edit it once at the top of the relevant section
and the typed builders below pick it up.
"""

from __future__ import annotations

from ddssm.builders import (
    Combiner,
    ConcatLinearFusionB,
    Context,
    ContextAggregatorB,
    DDSSM,
    Decoder,
    DiffResidualBlock,
    DiffTransition,
    Encoder,
    FeatureMixer,
    GaussianDistHeadB,
    GaussTransition,
    GRUFutSum,
    Head,
    Hparams,
    IdentityAggregatorB,
    MLPUnet,
    ResidualBlock,
    Schedule,
    TimeMixer,
    Unet,
    ZInit,
)

from conf.registry import (
    decoder_store,
    encoder_store,
    model_store,
    schedule_store,
    transition_store,
    unet_store,
    z_init_store,
)


# ---------------------------------------------------------------------------
# Family-shared arch primitives.
#
# Every shape below uses these unless it explicitly overrides. Tweak a
# field here (e.g. ``SmallFeature.nheads``) and the change propagates
# into every encoder/decoder/transition built afterwards.
# ---------------------------------------------------------------------------

SmallTime = TimeMixer(type="conv", kernel_size=3, gru_layers=1)
SmallFeature = FeatureMixer(type="transformer", nheads=8, n_layers=1)

SmallResBlock = ResidualBlock(time=SmallTime, feature=SmallFeature)
SmallDiffResBlock = DiffResidualBlock(time=SmallTime, feature=SmallFeature)

SmallContext = Context(
    channels=8,
    num_layers=2,
    residual_block=SmallResBlock,
)

# Clamped head (encoder / z-init): bounded logvar floor keeps the
# posterior from collapsing.
SmallHead = Head(clamp_logvar_min=-10.0)
# Unclamped head (decoder / Gaussian transition): a logvar prior in the
# decoder + the KL term keep this well-behaved.
PlainHead = Head()

# Encoder distribution head (Gaussian, clamped logvar) — used by every
# shape in this family.
SmallDistHead = GaussianDistHeadB(clamp_logvar_min=-10.0)

# Encoder combiner for ``j=1`` shapes: identity history aggregator
# (no history to mix when j=1) + concat-linear fusion of h_fut and
# z_{t-1}. Override the fusion to ``DKSFusionB()`` for a DKS-style
# combiner, or swap the aggregator for ``GRUAggregatorB`` /
# ``MLPAggregatorB`` / ``AttentionAggregatorB`` when ``j>1``.
SmallIdentityCombiner = Combiner(
    aggregator=IdentityAggregatorB(),
    fusion=ConcatLinearFusionB(),
)

# Encoder combiner for ``j>1`` shapes: ContextProducer aggregator over
# the j-step history (matching the residual-block stack the encoder
# used before the aggregator/fusion split) + concat-linear fusion.
SmallContextCombiner = Combiner(
    aggregator=ContextAggregatorB(
        channels=8, num_layers=2, residual_block=SmallResBlock,
    ),
    fusion=ConcatLinearFusionB(),
)

# Tiny GRU summary — single layer, 16-dim hidden state. The future
# summary is the dominant per-step cost (sequential over T); overkill
# for toy synthetic data at this latent size.
TinyGRU = GRUFutSum(summary_dim=16, num_layers=1, gru_layers=1)

# Full CSDI U-Net at the default channel/layer count — used by Robot2D
# and KDD diffusion transitions where the extra capacity helps.
CSDIUnet = Unet()
# Default MLP ablation — drop-in replacement of the same interface.
MLP = MLPUnet(channels=64, n_layers=3)
# Tiny MLP score-net — used by the 1D synthetic Diff transition so its
# size matches Gauss (transition: ~10k vs 13k). Also used by the
# variance probes (the diffusion net is the thing being measured, not
# the thing that needs capacity).
MLPTiny = MLPUnet(channels=32, n_layers=2, embedding_dim=32)

DefaultSchedule = Schedule()


# ---------------------------------------------------------------------------
# Shape: Small 1D (harmonic, bimodal, LGSSM).
#
# D=1, latent_dim=4, j=1. ``Diff1D`` uses the tiny MLP score-net so
# Gauss-vs-Diff comparisons differ in the *modelling* story, not raw
# capacity. Swap to ``CSDIUnet`` for an apples-to-CSDI comparison via
# override.
# ---------------------------------------------------------------------------


class Small1D:
    data_dim = 1
    latent_dim = 4
    j = 1
    emb_time_dim = 16
    covariate_dim = 0
    hidden_dim = 64
    mask_emb_dim = 8

    encoder = Encoder(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        static_covariate_dim=0,
        use_mask=False,
        hidden_dim=hidden_dim,
        fut_mask_emb_dim=mask_emb_dim,
        pad_mask_emb_dim=mask_emb_dim,
        combiner=SmallIdentityCombiner,
        dist_head=SmallDistHead,
        fut_summary=TinyGRU,
    )

    decoder = Decoder(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        static_covariate_dim=0,
        hidden_dim=hidden_dim, mask_emb_dim=mask_emb_dim,
        context=SmallContext,
        gaussian_head=PlainHead,
    )

    z_init = ZInit(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        hidden_dim=hidden_dim, pad_mask_emb_dim=mask_emb_dim,
        context=SmallContext, aux_context=SmallContext,
        gaussian_head=SmallHead, aux_posterior_head=SmallHead,
    )

    gauss_transition = GaussTransition(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        hidden_dim=hidden_dim,
        context=SmallContext,
        gaussian_head=PlainHead,
    )

    diff_transition = DiffTransition(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        unet=MLPTiny, schedule=DefaultSchedule,
    )

    gauss_model = DDSSM(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        encoder=encoder, decoder=decoder, z_init=z_init,
        transition=gauss_transition,
        hyperparams=Hparams(),
        use_observation_mask=False,
    )

    diff_model = DDSSM(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        encoder=encoder, decoder=decoder, z_init=z_init,
        transition=diff_transition,
        hyperparams=Hparams(),
        use_observation_mask=False,
    )


# ---------------------------------------------------------------------------
# Shape: Robot 2D (robot-basis-pursuit). D=2, latent_dim=6, j=2.
#
# Real spatial structure benefits from the conv backbone, so the
# diffusion transition keeps the full CSDI U-Net.
# ---------------------------------------------------------------------------


class Robot2D:
    data_dim = 2
    latent_dim = 6
    j = 2
    emb_time_dim = 16
    covariate_dim = 0
    hidden_dim = 64
    mask_emb_dim = 8

    encoder = Encoder(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        static_covariate_dim=0,
        use_mask=False,
        hidden_dim=hidden_dim,
        fut_mask_emb_dim=mask_emb_dim,
        pad_mask_emb_dim=mask_emb_dim,
        combiner=SmallContextCombiner,
        dist_head=SmallDistHead,
        fut_summary=TinyGRU,
    )

    decoder = Decoder(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        static_covariate_dim=0,
        hidden_dim=hidden_dim, mask_emb_dim=mask_emb_dim,
        context=SmallContext,
        gaussian_head=PlainHead,
    )

    z_init = ZInit(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        hidden_dim=hidden_dim, pad_mask_emb_dim=mask_emb_dim,
        context=SmallContext, aux_context=SmallContext,
        gaussian_head=SmallHead, aux_posterior_head=SmallHead,
    )

    gauss_transition = GaussTransition(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        hidden_dim=hidden_dim,
        context=SmallContext,
        gaussian_head=PlainHead,
    )

    diff_transition = DiffTransition(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        unet=CSDIUnet, schedule=DefaultSchedule,
    )

    gauss_model = DDSSM(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        encoder=encoder, decoder=decoder, z_init=z_init,
        transition=gauss_transition,
        hyperparams=Hparams(),
        use_observation_mask=False,
    )

    diff_model = DDSSM(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        encoder=encoder, decoder=decoder, z_init=z_init,
        transition=diff_transition,
        hyperparams=Hparams(),
        use_observation_mask=False,
    )


# ---------------------------------------------------------------------------
# Shape: ProbeMedium (D=4, nonlinear-bimodal-lift).
#
# Encoder/decoder/z_init only — variance-probe runs build their own
# transition + DDSSM on top in :mod:`experiments.variance_probe.model`.
# ---------------------------------------------------------------------------


class ProbeMedium:
    data_dim = 4
    latent_dim = 8
    j = 1
    emb_time_dim = 16
    covariate_dim = 0
    hidden_dim = 64
    mask_emb_dim = 8

    encoder = Encoder(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        static_covariate_dim=0,
        use_mask=False,
        hidden_dim=hidden_dim,
        fut_mask_emb_dim=mask_emb_dim,
        pad_mask_emb_dim=mask_emb_dim,
        combiner=SmallIdentityCombiner,
        dist_head=SmallDistHead,
        fut_summary=TinyGRU,
    )

    decoder = Decoder(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        static_covariate_dim=0,
        hidden_dim=hidden_dim, mask_emb_dim=mask_emb_dim,
        context=SmallContext,
        gaussian_head=PlainHead,
    )

    z_init = ZInit(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        hidden_dim=hidden_dim, pad_mask_emb_dim=mask_emb_dim,
        context=SmallContext, aux_context=SmallContext,
        gaussian_head=SmallHead, aux_posterior_head=SmallHead,
    )


# ---------------------------------------------------------------------------
# Store registrations.
#
# Public CLI handles for encoder/decoder/z_init/transition/unet/
# schedule/model. Names match the pre-consolidation registry so any
# downstream ``python -m ddssm.app model=small_gauss`` continues to work.
# ---------------------------------------------------------------------------

encoder_store(Small1D.encoder, name="small_1d")
encoder_store(Robot2D.encoder, name="robot2d")
encoder_store(ProbeMedium.encoder, name="probe_medium")

decoder_store(Small1D.decoder, name="small_1d")
decoder_store(Robot2D.decoder, name="robot2d")
decoder_store(ProbeMedium.decoder, name="probe_medium")

z_init_store(Small1D.z_init, name="small_1d")
z_init_store(Robot2D.z_init, name="robot2d")
z_init_store(ProbeMedium.z_init, name="probe_medium")

transition_store(Small1D.gauss_transition, name="gauss_1d")
transition_store(Small1D.diff_transition, name="diff_1d")
transition_store(Robot2D.gauss_transition, name="gauss_robot2d")
transition_store(Robot2D.diff_transition, name="diff_robot2d")

unet_store(CSDIUnet, name="csdi")
unet_store(MLP, name="mlp")
unet_store(MLPTiny, name="mlp_tiny")

schedule_store(DefaultSchedule, name="default")

model_store(Small1D.gauss_model, name="small_gauss")
model_store(Small1D.diff_model, name="small_diff")
model_store(Robot2D.gauss_model, name="robot2d_gauss")
model_store(Robot2D.diff_model, name="robot2d_diff")
