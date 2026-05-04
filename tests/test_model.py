# tests/test_model.py
import torch
import pytest
from ddssm.dssd import (
    diff_DDSSM,
    decode_DDSSM,
    encode_DDSSM,
)
from ddssm.net_utils import get_side_info, time_embedding

from ddssm.config import (
    DDSSMConfig,
    UNetConfig,
    DecoderConfig,
    EncoderConfig,
    DDSSMHyperParams,
    ResidualBlockConfig,
    DiffusionScheduleConfig,
    DiffusionEmbeddingConfig,
)


def make_base_config():
    return DDSSMConfig(
        schedule=DiffusionScheduleConfig(num_steps=3),
        encoder=EncoderConfig(
            history_len=1, emb_feature_dim=4, hidden_dim=8, num_layers=1
        ),
        decoder=DecoderConfig(hidden_dim=8, num_layers=1),
        unet=UNetConfig(
            feature_emb_dim=4,
            embedding=DiffusionEmbeddingConfig(embedding_dim=8, projection_dim=8),
            block=ResidualBlockConfig(channels=8, layers=1, nheads=2),
        ),
        history_len=1,
        prediction_len=1,
        data_dim=2,
        latent_dim=3,
        emb_time_dim=5,
        latent_history_len=1,
        hyperparams=DDSSMHyperParams(S=2, loss_lambda=1.0, lr=1e-3, wd=1e-4),
    )


@pytest.fixture
def base_config():
    return make_base_config()


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


def test_encode_collect_shapes(base_config):
    enc = encode_DDSSM(
        config=base_config.encoder,
        history_len=base_config.history_len,
        data_dim=base_config.data_dim,
        latent_dim=base_config.latent_dim,
        emb_time_dim=base_config.emb_time_dim,
        device="cpu",
    )
    B, D, T = 2, base_config.data_dim, 3
    x = torch.randn(B, D, T)
    time_emb = torch.randn(B, T, base_config.emb_time_dim)
    # single sample
    mus, logvars, zs = enc.collect_stats_samples(x, time_emb, S=2, device="cpu")
    assert mus.shape == (B, base_config.latent_dim, T)
    assert logvars.shape == (B, base_config.latent_dim, T)
    assert zs.shape == (B, 2, base_config.latent_dim, T)


def test_decode_shapes(base_config):
    dec = decode_DDSSM(
        config=base_config.decoder,
        latent_dim=base_config.latent_dim,
        data_dim=base_config.data_dim,
    )
    B, Z = 3, base_config.latent_dim
    z = torch.randn(B, Z)
    x_hat = dec(z)
    assert x_hat.shape == (B, base_config.data_dim)


def test_diffusion_shapes(base_config):
    # instantiate diff
    diff = diff_DDSSM(
        config=base_config.unet,
        history_len=base_config.history_len,
        prediction_len=base_config.prediction_len,
        diffusion_steps=base_config.schedule.num_steps,
        latent_dim=base_config.latent_dim,
        latent_history_len=base_config.latent_history_len,
        side_dim=base_config.emb_time_dim + base_config.unet.feature_emb_dim,
    )
    B, d, L = (
        2,
        base_config.latent_dim,
        base_config.latent_history_len + base_config.prediction_len,
    )
    x = torch.randn(B, d, L)
    # side info must be (B, side_dim, d, L)
    time_emb = torch.randn(B, L, base_config.emb_time_dim)
    side = get_side_info(
        base_config.latent_dim,
        time_emb,
        torch.nn.Embedding(base_config.latent_dim, base_config.unet.feature_emb_dim),
    )
    side = side[:, : diff.side_dim, :, :]  # match side_dim
    steps = torch.randint(0, base_config.schedule.num_steps, (B,))
    out = diff(x, side, steps)
    # output shape (B, d, P)
    assert out.shape == (B, d, base_config.prediction_len)
