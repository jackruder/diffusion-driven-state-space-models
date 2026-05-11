# tests/test_trainer.py
from functools import partial
import torch
import pytest
from hydra_zen import instantiate
from ddssm.dssd import DDSSM_base
from ddssm.train import DDSSMTrainer
from ddssm.encoder import GaussianEncoder, GaussianInitPrior
from ddssm.decoder import GaussianDecoder
from ddssm.transitions.transitions import GaussianTransition
from ddssm.conf import DDSSMTrainerConf
from ddssm.diffnets import ContextProducer, FeatureMixerConfig, ResidualBlockConfig
from ddssm.gaussians import GaussianHead
from ddssm.futsum import GRUFutureSummary
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace

J = 1
DATA_DIM = 3
LATENT_DIM = 2
EMB_TIME = 8
CHANNELS = 8
NHEADS = 4

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


def make_small_model():
    enc = GaussianEncoder(
        data_dim=DATA_DIM, latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        use_mask=True, hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH, fut_summary=_FS,
    )
    dec = GaussianDecoder(
        latent_dim=LATENT_DIM, data_dim=DATA_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH,
    )
    zinit = GaussianInitPrior(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, aux_context=_CTX, gaussian_head=_GH, aux_posterior_head=_GH,
    )
    trans = GaussianTransition(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH,
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


def test_trainer_conf_builds_instantiates(small_model, tmp_path):
    trainer = instantiate(
        DDSSMTrainerConf(
            model=small_model,
            device=torch.device("cpu"),
            tensorboard_dir=str(tmp_path / "runs"),
            quiet=True,
        )
    )
    assert isinstance(trainer, DDSSMTrainer)
