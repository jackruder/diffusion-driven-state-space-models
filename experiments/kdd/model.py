"""Composed model configs for the KDD Cup 2018 PM2.5 family.

KDD uses a wider footprint than the synthetic family — D=6,
real-world covariates (3-d), and longer sequences justify the wider
context producer + GRU summary. All low-level knobs are spelled out
in one place so changing e.g. ``hidden_dim = 128`` propagates into
encoder + decoder + z_init + transition in one shot.
"""

from __future__ import annotations

from ddssm.builders import (
    Combiner,
    ConcatLinearFusionB,
    Context,
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
    transition_store,
    z_init_store,
)


# ---------------------------------------------------------------------------
# Arch primitives.
# ---------------------------------------------------------------------------

KDDTime = TimeMixer(type="conv", kernel_size=3, gru_layers=1)
KDDFeature = FeatureMixer(type="transformer", nheads=8, n_layers=1)

KDDResBlock = ResidualBlock(time=KDDTime, feature=KDDFeature)
KDDDiffResBlock = DiffResidualBlock(time=KDDTime, feature=KDDFeature)

KDDContext = Context(
    channels=8,
    num_layers=2,
    residual_block=KDDResBlock,
)

# Clamped (encoder / z-init) and unclamped (decoder / Gaussian transition).
KDDClampedHead = Head(clamp_logvar_min=-10.0)
KDDPlainHead = Head()

# 64-dim, 2-layer GRU summary — suitable for the 6-dim multivariate KDD series.
KDDFutSum = GRUFutSum(summary_dim=64, num_layers=2, gru_layers=1)

# Full CSDI U-Net for KDD's diffusion transition (64 channels, 4 layers).
KDDUnet = Unet(
    channels=64,
    n_layers=4,
    embedding_dim=128,
    residual_block=KDDDiffResBlock,
)
KDDSchedule = Schedule()

# Encoder distribution head + combiner. KDD has j=1, so the history
# aggregator is the identity (no z-history mixing needed); fusion is the
# default concat-linear of h_fut and z_{t-1}. Override with
# ``DKSFusionB()`` for a DKS-style combiner.
KDDDistHead = GaussianDistHeadB(clamp_logvar_min=-10.0)
KDDIdentityCombiner = Combiner(
    aggregator=IdentityAggregatorB(),
    fusion=ConcatLinearFusionB(),
)


# ---------------------------------------------------------------------------
# Shape: KDD (D=6, latent_dim=8, j=1, covariate_dim=3, emb_time_dim=32).
# ---------------------------------------------------------------------------


class KDD:
    data_dim = 6
    latent_dim = 8
    j = 1
    emb_time_dim = 32
    covariate_dim = 3
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
        combiner=KDDIdentityCombiner,
        dist_head=KDDDistHead,
        fut_summary=KDDFutSum,
    )

    decoder = Decoder(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        static_covariate_dim=0,
        hidden_dim=hidden_dim, mask_emb_dim=mask_emb_dim,
        context=KDDContext,
        gaussian_head=KDDPlainHead,
    )

    z_init = ZInit(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        hidden_dim=hidden_dim, pad_mask_emb_dim=mask_emb_dim,
        context=KDDContext, aux_context=KDDContext,
        gaussian_head=KDDClampedHead, aux_posterior_head=KDDClampedHead,
    )

    gauss_transition = GaussTransition(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        hidden_dim=hidden_dim,
        context=KDDContext,
        gaussian_head=KDDPlainHead,
    )

    diff_transition = DiffTransition(
        latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, covariate_dim=covariate_dim,
        unet=KDDUnet, schedule=KDDSchedule,
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
# Store registrations.
# ---------------------------------------------------------------------------

encoder_store(KDD.encoder, name="kdd")
decoder_store(KDD.decoder, name="kdd")
z_init_store(KDD.z_init, name="kdd")
transition_store(KDD.gauss_transition, name="kdd_gauss")
transition_store(KDD.diff_transition, name="kdd_diff")
model_store(KDD.gauss_model, name="kdd_gauss")
model_store(KDD.diff_model, name="kdd_diff")
