# tests/test_model.py
from functools import partial
import torch
import pytest
from ddssm.net_utils import get_side_info, time_embedding
from ddssm.encoder import GaussianEncoder, GaussianInitPrior
from ddssm.decoder import Decoder
from ddssm.transitions.transitions import GaussianTransition
from ddssm.transitions.diffusion import DiffusionTransition
from ddssm.dssd import DDSSM_base
from ddssm.diffnets import (
    ContextProducer,
    CSDIUnet,
    DiffResidualBlockConfig,
    FeatureMixerConfig,
    ResidualBlockConfig,
)
from ddssm.gaussians import GaussianHead
from ddssm.futsum import GRUFutureSummary
from ddssm.transitions.diffusion import DiffusionScheduleConfig
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Shared tiny config (channels=8 to keep tests fast; nheads=4 divides 8)
# ---------------------------------------------------------------------------

J = 2
DATA_DIM = 3
LATENT_DIM = 4
EMB_TIME = 8
CHANNELS = 8
NHEADS = 4

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


def make_encoder():
    return GaussianEncoder(
        data_dim=DATA_DIM, latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        use_mask=True, hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH, fut_summary=_FS,
    )


def make_decoder():
    return Decoder(
        latent_dim=LATENT_DIM, data_dim=DATA_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH,
    )


def make_zinit():
    return GaussianInitPrior(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, aux_context=_CTX, gaussian_head=_GH, aux_posterior_head=_GH,
    )


def make_gaussian_transition():
    return GaussianTransition(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH,
    )


def make_hyperparams():
    return SimpleNamespace(
        S=1, ema_decay=0.999, weight_decay=1e-2, batch_size=2, grad_accum_steps=1,
        t_chunk=4, clip_grad_norm=None, lambda_schedule="none", lambda_start=0.001,
        lambda_end=1.0, lambda_warmup_steps=1, enc_lr=1e-3, dec_lr=1e-3,
        zinit_lr=1e-3, trans_lr=1e-3, logvar_min=-7.0, logvar_max=7.0,
        rewo=SimpleNamespace(D0=0.1, nu=1e-3, alpha=0.99, tau1=1.0, tau2=1.0),
    )


@pytest.fixture
def model():
    return DDSSM_base(
        encoder=make_encoder(), decoder=make_decoder(),
        z_init=make_zinit(), transition=make_gaussian_transition(),
        j=J, data_dim=DATA_DIM, latent_dim=LATENT_DIM, emb_time_dim=EMB_TIME,
        hyperparams=make_hyperparams(),
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


def test_diffusion_transition_builds():
    """DiffusionTransition should build with config objects and register expected buffers."""
    dt = DiffusionTransition(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        unet=partial(
            CSDIUnet,
            channels=CHANNELS,
            n_layers=1,
            embedding_dim=CHANNELS,
            residual_block=DiffResidualBlockConfig(
                feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
            ),
        ),
        schedule=DiffusionScheduleConfig(num_steps=10),
    )
    for buf in ("alpha_bar", "wtilde", "sigma", "c_noise", "p_k"):
        assert buf in dt._buffers, f"Missing buffer: {buf}"


# ---------------------------------------------------------------------------
# transition_kl: dict contract for both transitions
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
        enc_stats=enc_stats, zs=zs, logq_paths=logq, time_embed=time_emb,
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
        enc_stats={}, zs=zs, logq_paths=logq, time_embed=time_emb,
    )
    assert set(out.keys()) == {"kl", "L_p", "L_q"}
    for v in out.values():
        assert v.ndim == 0
    # kl == L_p - L_q
    assert torch.allclose(out["kl"], out["L_p"] - out["L_q"])


def _make_diffusion_transition():
    return DiffusionTransition(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        unet=partial(
            CSDIUnet,
            channels=CHANNELS,
            n_layers=1,
            embedding_dim=CHANNELS,
            residual_block=DiffResidualBlockConfig(
                feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
            ),
        ),
        schedule=DiffusionScheduleConfig(num_steps=4, S_k=1, k_chunk=1),
    )


def test_diffusion_transition_kl_closed_form_entropy():
    trans = _make_diffusion_transition()
    zs, logq, _mus, logvars, time_emb = _make_inputs()
    enc_stats = {"logvars": logvars}
    out = trans.transition_kl(
        enc_stats=enc_stats, zs=zs, logq_paths=logq, time_embed=time_emb,
    )
    assert set(out.keys()) == {"kl", "L_p", "L_q"}
    for v in out.values():
        assert v.ndim == 0
    assert torch.allclose(out["kl"], out["L_p"] - out["L_q"])


def test_diffusion_transition_kl_mc_entropy():
    trans = _make_diffusion_transition()
    zs, logq, _mus, _lv, time_emb = _make_inputs()
    out = trans.transition_kl(
        enc_stats={}, zs=zs, logq_paths=logq, time_embed=time_emb,
    )
    assert set(out.keys()) == {"kl", "L_p", "L_q"}
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
    loss, distortion, rate, metrics, stats = result
    assert loss.ndim == 0  # scalar
    assert distortion.ndim == 0
    # New transition KL metric key (replaces former trans/total, trans/diff,
    # trans/entropy).  GaussianTransition + Gaussian encoder uses closed-form
    # KL so no L_p/L_q sub-component keys are emitted.
    assert "loss/rate/trans/kl" in metrics
    assert "loss/rate/trans/total" not in metrics
    assert "loss/rate/trans/diff" not in metrics
    assert "loss/rate/trans/entropy" not in metrics
