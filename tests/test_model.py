# tests/test_model.py
from types import SimpleNamespace
from functools import partial

import torch
import pytest

from ddssm.nn.futsum import GRUFutureSummary
from ddssm.model.dssd import DDSSM_base
from ddssm.nn.fusions import ConcatLinearFusion
from ddssm.nn.diffnets import (
    ContextProducer,
    FeatureMixerConfig,
    ResidualBlockConfig,
)
from ddssm.nn.combiners import CompoundCombiner
from ddssm.nn.gaussians import GaussianHead
from ddssm.nn.net_utils import get_side_info, time_embedding
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder
from ddssm.nn.dist_heads import GaussianDistHead
from ddssm.nn.aggregators import ContextProducerAggregator
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.model.transitions.transitions import GaussianTransition

# ---------------------------------------------------------------------------
# Shared tiny config (channels=16, nheads=2: head_dim=8 for SDPA compat)
# ---------------------------------------------------------------------------

J = 2
DATA_DIM = 3
LATENT_DIM = 4
EMB_TIME = 8
CHANNELS = 16
NHEADS = 2

# Small architectural configs reused across tests
_CTX = partial(
    ContextProducer,
    channels=CHANNELS,
    num_layers=1,
    residual_block=ResidualBlockConfig(
        feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
    ),
)
_GH = GaussianHead  # zen_partial-style: parents call _GH(in_features=..., out_features=...)
_FS = partial(GRUFutureSummary, summary_dim=CHANNELS, num_layers=1)

# Encoder-only slots: aggregator + fusion + dist head.
_AGG = partial(
    ContextProducerAggregator,
    channels=CHANNELS,
    num_layers=1,
    residual_block=ResidualBlockConfig(
        feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
    ),
)
_FUSION = partial(ConcatLinearFusion)
_DIST_HEAD = partial(GaussianDistHead)


def make_encoder():
    return GaussianEncoder(
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        use_mask=True,
        hidden_dim=CHANNELS,
        combiner=partial(CompoundCombiner, aggregator=_AGG, fusion=_FUSION),
        dist_head=_DIST_HEAD,
        fut_summary=_FS,
    )


def make_decoder():
    return GaussianDecoder(
        latent_dim=LATENT_DIM,
        data_dim=DATA_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX,
        gaussian_head=_GH,
    )


def make_aux():
    return AuxPosterior(latent_dim=LATENT_DIM, j=J, hidden_dim=CHANNELS, n_layers=1)


def make_gaussian_transition():
    return GaussianTransition(
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX,
        gaussian_head=_GH,
    )


def make_hyperparams():
    return SimpleNamespace(
        S=1,
        ema_decay=0.999,
        weight_decay=1e-2,
        batch_size=2,
        grad_accum_steps=1,
        t_chunk=4,
        clip_grad_norm=None,
        enc_lr=1e-3,
        dec_lr=1e-3,
        trans_lr=1e-3,
        logvar_min=-7.0,
        logvar_max=7.0,
    )


@pytest.fixture
def model():
    return DDSSM_base(
        encoder=make_encoder(),
        decoder=make_decoder(),
        transition=make_gaussian_transition(),
        aux_posterior=make_aux(),
        j=J,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        emb_time_dim=EMB_TIME,
    )


# ---------------------------------------------------------------------------
# net_utils utilities
# ---------------------------------------------------------------------------


def test_time_embedding_shapes():
    B, L, D = 2, 4, 6
    pos = torch.randint(0, 10, (B, L))
    emb = time_embedding(pos, d_model=D, device="cpu")
    assert emb.shape == (B, L, D)


def test_get_side_info_shapes():
    B, L, data_dim, emb_time, emb_feat = 2, 5, 3, 6, 4
    time_emb = torch.randn(B, L, emb_time)
    embed_layer = torch.nn.Embedding(data_dim, emb_feat)
    mask = torch.randint(0, 2, (B, data_dim, L))
    side = get_side_info(data_dim, time_emb, embed_layer, cond_mask=mask, device="cpu")
    # channels = emb_time + emb_feat + 1
    assert side.shape == (B, emb_time + emb_feat + 1, data_dim, L)


# ---------------------------------------------------------------------------
# Module shape tests
# ---------------------------------------------------------------------------


def test_encoder_sample_paths():
    enc = make_encoder()
    B, T, S = 2, 6, 2
    x = torch.randn(B, DATA_DIM, T)
    time_emb = torch.randn(B, T, EMB_TIME)
    mask = torch.ones(B, DATA_DIM, T)
    zs, logqs, stats = enc.sample_paths(x, time_emb, S=S, cond_mask=mask)
    assert zs.shape == (B, S, LATENT_DIM, T)
    assert logqs.shape == (B, S, T)


def test_decoder_forward():
    dec = make_decoder()
    B, T = 2, 4
    z_hist = torch.randn(B, LATENT_DIM, J)
    time_emb = torch.randn(B, T, EMB_TIME)
    time_idx = torch.randint(J, T, (B,))
    mu, logvar = dec.forward_unpadded(z=z_hist, time_embed=time_emb, time_idx=time_idx)
    assert mu.shape == (B, DATA_DIM)
    assert logvar.shape == (B, DATA_DIM)


def test_gaussian_transition_prior_params():
    trans = make_gaussian_transition()
    B = 3
    z_hist = torch.randn(B, LATENT_DIM, J)
    time_emb = torch.randn(B, J, EMB_TIME)
    mu, logvar = trans.prior_params(z_hist, ctx={"hist_time_emb": time_emb})
    assert mu.shape == (B, LATENT_DIM)
    assert logvar.shape == (B, LATENT_DIM)


# ---------------------------------------------------------------------------
# transition_kl: dict contract for the Gaussian transition
# ---------------------------------------------------------------------------


def _make_inputs(B=2, S=2, T=5):
    torch.manual_seed(0)
    zs = torch.randn(B, S, LATENT_DIM, T)
    logq = torch.randn(B, S, T)
    mus = torch.randn(B, S, LATENT_DIM, T)
    logvars = torch.randn(B, S, LATENT_DIM, T) * 0.1 - 1.0
    time_emb = torch.randn(B, T, EMB_TIME)
    return zs, logq, mus, logvars, time_emb


def test_gaussian_transition_kl_closed_form():
    trans = make_gaussian_transition()
    zs, logq, mus, logvars, time_emb = _make_inputs()
    enc_stats = {"mus": mus, "logvars": logvars}
    out = trans.transition_kl(
        enc_stats=enc_stats,
        zs=zs,
        logq_paths=logq,
        time_embed=time_emb,
    )
    # Closed-form path returns only "kl" (no L_p / L_q sub-components)
    assert set(out.keys()) == {"kl"}
    assert out["kl"].ndim == 0
    # KL of diagonal Gaussians is non-negative
    assert out["kl"].item() >= -1e-5


def test_gaussian_transition_kl_mc_fallback():
    trans = make_gaussian_transition()
    zs, logq, _mus, _lv, time_emb = _make_inputs()
    out = trans.transition_kl(
        enc_stats={},
        zs=zs,
        logq_paths=logq,
        time_embed=time_emb,
    )
    assert set(out.keys()) == {"kl", "L_p", "L_q"}
    for v in out.values():
        assert v.ndim == 0
    # kl == L_p - L_q
    assert torch.allclose(out["kl"], out["L_p"] - out["L_q"])


# ---------------------------------------------------------------------------
# DDSSM_base forward pass
# ---------------------------------------------------------------------------


def test_ddssm_forward(model):
    B, T = 2, 5
    x = torch.randn(B, DATA_DIM, T)
    mask = torch.ones(B, DATA_DIM, T)
    timepoints = torch.arange(T).unsqueeze(0).expand(B, -1)
    result = model(observed_data=x, observation_mask=mask, timepoints=timepoints)
    components, metrics, stats = result
    assert components.total().ndim == 0  # scalar
    assert components.recon.ndim == 0
    # New transition KL metric key (replaces former trans/total, trans/diff,
    # trans/entropy).  GaussianTransition + Gaussian encoder uses closed-form
    # KL so no L_p/L_q sub-component keys are emitted.
    assert "loss/rate/trans/kl" in metrics
    assert "loss/rate/trans/total" not in metrics
    assert "loss/rate/trans/diff" not in metrics
    assert "loss/rate/trans/entropy" not in metrics


# ---------------------------------------------------------------------------
# Encoder with each aggregator backbone — end-to-end ELBO smoke test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agg_name", ["identity", "gru", "mlp", "attention", "context"])
def test_ddssm_forward_with_each_aggregator(agg_name):
    """Every aggregator backbone produces a finite ELBO end-to-end."""
    from ddssm.nn.fusions import ConcatLinearFusion
    from ddssm.nn.combiners import CompoundCombiner
    from ddssm.nn.dist_heads import GaussianDistHead
    from ddssm.nn.aggregators import (
        GRUAggregator,
        MLPAggregator,
        IdentityAggregator,
        AttentionAggregator,
        ContextProducerAggregator,
    )

    # Identity requires j=1; others run with the test default j=2.
    j = 1 if agg_name == "identity" else J
    rb = ResidualBlockConfig(feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1))
    agg_map = {
        "identity": partial(IdentityAggregator),
        "gru": partial(GRUAggregator, num_gru_layers=1),
        "mlp": partial(MLPAggregator, num_layers=2),
        "attention": partial(AttentionAggregator, nheads=NHEADS, num_attn_layers=1),
        "context": partial(
            ContextProducerAggregator,
            channels=CHANNELS,
            num_layers=1,
            residual_block=rb,
        ),
    }
    combiner = partial(
        CompoundCombiner,
        aggregator=agg_map[agg_name],
        fusion=partial(ConcatLinearFusion),
    )

    enc = GaussianEncoder(
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        j=j,
        emb_time_dim=EMB_TIME,
        use_mask=True,
        hidden_dim=CHANNELS,
        combiner=combiner,
        dist_head=partial(GaussianDistHead),
        fut_summary=_FS,
    )
    # Decoder / transition keep their original ContextProducer.
    dec = GaussianDecoder(
        latent_dim=LATENT_DIM,
        data_dim=DATA_DIM,
        j=j,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX,
        gaussian_head=_GH,
    )
    aux = AuxPosterior(latent_dim=LATENT_DIM, j=j, hidden_dim=CHANNELS, n_layers=1)
    trans = GaussianTransition(
        latent_dim=LATENT_DIM,
        j=j,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX,
        gaussian_head=_GH,
    )

    model = DDSSM_base(
        encoder=enc,
        decoder=dec,
        transition=trans,
        aux_posterior=aux,
        j=j,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        emb_time_dim=EMB_TIME,
    )

    B, T = 2, 5
    x = torch.randn(B, DATA_DIM, T)
    mask = torch.ones(B, DATA_DIM, T)
    timepoints = torch.arange(T).unsqueeze(0).expand(B, -1)
    components, _metrics, _stats = model(
        observed_data=x,
        observation_mask=mask,
        timepoints=timepoints,
    )
    loss = components.total()
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# eval metrics that run a real model (regression guard for forward's arity)
# ---------------------------------------------------------------------------


def test_recon_mse_metric_runs_on_real_model(model):
    """recon_mse must unpack model.forward's 3-tuple, not a 5-tuple.

    Regression guard: ``eval_recon_mse`` previously did
    ``_l, _r, _d, _m, stats = model(...)`` which raised ``ValueError`` on
    every call because ``DDSSM_base.forward`` returns ``(components,
    metrics, stats)``.
    """
    from ddssm.eval.metrics import EvalContext, eval_recon_mse

    B, T = 2, 6
    batch = {
        "observed_data": torch.randn(B, DATA_DIM, T),
        "observation_mask": torch.ones(B, DATA_DIM, T),
        "timepoints": torch.arange(T).unsqueeze(0).expand(B, -1),
    }
    ctx = EvalContext(
        model=model,
        loader=[batch],
        device=torch.device("cpu"),
    )
    out = eval_recon_mse(ctx)
    assert "recon_mse" in out
    assert out["recon_mse"] >= 0.0
