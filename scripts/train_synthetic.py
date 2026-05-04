from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x

from ddssm.synthetic_serialize import (
    SeriesConfig,
    make_dataloaders,
    generate_and_serialize_dataset,
)

from ddssm.train import DDSSMTrainer
from ddssm.stages import StageOrchestrator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DS_ROOT = "./data/synthetic0"
DS_CONFIG = "./configs/synth0.yaml"
TEST_CONFIG = "./configs/base.yaml"

model = DDSSMTrainer.load_from_yaml(TEST_CONFIG, device)

# generate synthetic data, override necessary params as needed by the model
sconf = SeriesConfig.load(Path(DS_CONFIG))
sconf.D = model.model.data_dim

generate_and_serialize_dataset(
    Path(TEST_DS_ROOT),
    n_series=1,
    series_config=sconf,
    window_length=model.model.window_len,
    val_split=0.2,
    test_split=0.1,
    normalize=True,
    seed=42,
)

data = make_dataloaders(Path(TEST_DS_ROOT), model.get_batch_size(), device)
train_loader, val_loader = data["train_loader"], data["valid_loader"]

if model.model.config.stages is None:
    pass
else:
    StageOrchestrator(model, model.model.config).run(
        train_loader,
        val_loader,
        amp=False,
    )
