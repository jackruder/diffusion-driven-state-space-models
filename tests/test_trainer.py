# tests/test_trainer.py
import torch
import pytest
from ddssm.dssd import DDSSM_base
from ddssm.train import DDSSMTrainer
from ddssm.encoder import GaussianEncoder, GaussianInitPrior
from ddssm.decoder import Decoder
from ddssm.transitions.transitions import GaussianTransition
from ddssm.conf import DDSSMHyperParamsConf
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace

J = 1
DATA_DIM = 3
LATENT_DIM = 2
EMB_TIME = 8
CHANNELS = 8
NHEADS = 4


def make_small_model():
    enc = GaussianEncoder(
        data_dim=DATA_DIM, latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        use_mask=True, hidden_dim=CHANNELS, context_channels=CHANNELS,
        context_num_layers=1, context_nheads=NHEADS, context_feature_nheads=NHEADS,
        summary_dim=CHANNELS, summary_num_layers=1,
    )
    dec = Decoder(
        latent_dim=LATENT_DIM, data_dim=DATA_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS, context_channels=CHANNELS, context_num_layers=1,
        context_nheads=NHEADS, context_feature_nheads=NHEADS,
    )
    zinit = GaussianInitPrior(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS, context_channels=CHANNELS, context_num_layers=1,
        context_nheads=NHEADS, context_feature_nheads=NHEADS,
        aux_context_channels=CHANNELS, aux_context_num_layers=1,
        aux_context_nheads=NHEADS, aux_context_feature_nheads=NHEADS,
    )
    trans = GaussianTransition(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS, context_channels=CHANNELS, context_num_layers=1,
        context_nheads=NHEADS, context_feature_nheads=NHEADS,
    )
    hp = SimpleNamespace(
        S=1, ema_decay=0.999, weight_decay=1e-2, batch_size=1, grad_accum_steps=1,
        t_chunk=4, clip_grad_norm=None, lambda_schedule="none", lambda_start=0.001,
        lambda_end=1.0, lambda_warmup_steps=1, enc_lr=1e-3, dec_lr=1e-3,
        zinit_lr=1e-3, trans_lr=1e-3, logvar_min=-7.0, logvar_max=7.0,
        rewo=SimpleNamespace(D0=0.1, nu=1e-3, alpha=0.99, tau1=1.0, tau2=1.0),
    )
    return DDSSM_base(
        encoder=enc, decoder=dec, z_init=zinit, transition=trans,
        j=J, data_dim=DATA_DIM, latent_dim=LATENT_DIM, emb_time_dim=EMB_TIME,
        hyperparams=hp,
    )


@pytest.fixture
def small_model():
    return make_small_model()

