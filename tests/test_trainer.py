# tests/test_trainer.py
import torch
import pytest
from ddssm.dssd import DDSSM_base
from ddssm.train import DDSSMTrainer
from torch.utils.data import Dataset, DataLoader

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

B = 1  # Batch size
D = 5  # Data dimension
H = 2  # History length
P = 1  # Prediction length
T = H + P + 1  # Number of timepoints
L = 2  # Latent dimension


class DummyDataset(Dataset):
    def __init__(self, B, D, H, P, T):
        self.B = B
        self.D = D
        self.H = H
        self.P = P
        self.T = T

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        x = torch.randn(self.D, self.H + 1)
        hx = torch.randn(self.D, self.P)
        return {
            "history": x,
            "history_mask": torch.ones(self.D, self.H + 1),
            "future": hx,
            "future_mask": torch.ones(self.D, self.P),
            "timepoints": torch.arange(self.T),
            "feature_id": torch.arange(self.D),
        }


@pytest.fixture
def small_model():
    cfg = DDSSMConfig(
        schedule=DiffusionScheduleConfig(num_steps=3),
        encoder=EncoderConfig(
            history_len=H, emb_feature_dim=6, hidden_dim=32, num_layers=1
        ),
        decoder=DecoderConfig(hidden_dim=8, num_layers=1),
        unet=UNetConfig(
            feature_emb_dim=4,
            embedding=DiffusionEmbeddingConfig(embedding_dim=8, projection_dim=8),
            block=ResidualBlockConfig(channels=8, layers=1, nheads=2),
        ),
        history_len=H,
        prediction_len=P,
        data_dim=D,
        latent_dim=2,
        latent_history_len=1,
        emb_time_dim=16,
        hyperparams=DDSSMHyperParams(S=2, loss_lambda=1.0, lr=1e-3, wd=1e-4),
    )
    return DDSSM_base(cfg, device="cpu")


def test_trainer_io(tmp_path, small_model):
    trainer = DDSSMTrainer(small_model, device="cpu")
    # config
    cfg_file = tmp_path / "cfg.yaml"
    trainer.save_config(str(cfg_file))
    loaded = DDSSMTrainer.load_from_yaml(str(cfg_file), device="cpu")
    assert isinstance(loaded, DDSSMTrainer)

    # checkpoint
    ckpt = tmp_path / "ckpt.pth"
    trainer.save_checkpoint(str(ckpt), epoch=5)
    trainer2, epoch = DDSSMTrainer.load_checkpoint(
        str(ckpt), device="cpu", optimizer=torch.optim.AdamW(small_model.parameters())
    )
    assert epoch == 5
    # step the optimizer
    trainer2.optimizer.step()


def test_train_epoch(small_model):
    ds = DummyDataset(B, D, H, P, T)
    dl = DataLoader(ds, batch_size=B)
    trainer = DDSSMTrainer(small_model, device="cpu")
    stats = trainer.train_one_epoch(dl)
    assert set(stats.keys()) == {"loss", "rec", "prior_kl", "diff_kl"}
    for v in stats.values():
        assert isinstance(v, float)
