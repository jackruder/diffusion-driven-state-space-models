# tests/test_config_models.py
import pytest

from dkdm.config import (
    DKDMConfig,
)

EXAMPLE_YAML = """
schedule:
  num_steps: 500
  schedule: quad
  beta_start: 0.001
  beta_end: 0.01

encoder:
  emb_feature_dim: 32
  hidden_dim: 16
  num_layers: 2

decoder:
  hidden_dim: 16
  num_layers: 2

unet:
  feature_emb_dim: 8
  embedding:
    embedding_dim: 32
    projection_dim: 16
  block:
    channels: 16
    layers: 2
    nheads: 2

history_len: 2
prediction_len: 1
data_dim: 3
latent_dim: 4
latent_history_len: 1
emb_time_dim: 8

hyperparams:
  S: 2
  loss_lambda: 0.7
  lr: 5e-4
  wd: 1e-3
"""


def test_parse_yaml(tmp_path):
    fn = tmp_path / "cfg.yaml"
    fn.write_text(EXAMPLE_YAML)
    cfg = DKDMConfig.load_yaml(str(fn))
    # spot checks
    assert cfg.schedule.num_steps == 500
    assert cfg.schedule.schedule == "quad"
    assert pytest.approx(cfg.schedule.beta_end, rel=1e-6) == 0.01
    assert cfg.encoder.num_layers == 2
    assert cfg.unet.block.nheads == 2
    assert cfg.hyperparams.lr == 5e-4
    assert cfg.data_dim == 3
    assert cfg.latent_dim == 4
