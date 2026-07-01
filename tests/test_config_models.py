# tests/test_config_models.py
"""Tests for the hydra-zen configuration layer."""

import pytest
from omegaconf import OmegaConf

from ddssm.model.dssd import DDSSMHyperParamsConf
from ddssm.training.stages import StageSpecConf


def test_hyperparams_defaults():
    hp = DDSSMHyperParamsConf()
    assert hp.S == 1
    assert hp.enc_lr == pytest.approx(5e-4)
    assert hp.logvar_min == pytest.approx(-13.0)


def test_hyperparams_override():
    hp = DDSSMHyperParamsConf(S=4, enc_lr=1e-3, batch_size=32)
    assert hp.S == 4
    assert hp.enc_lr == pytest.approx(1e-3)
    assert hp.batch_size == 32


def test_stagespec_defaults():
    spec = StageSpecConf(steps=500)
    assert spec.steps == 500
    assert spec.log_every == 10
    assert spec.val_every == 100


def test_yaml_roundtrip(tmp_path):
    """Config should survive an OmegaConf YAML round-trip."""
    hp = DDSSMHyperParamsConf(S=2, dec_lr=1e-3)
    cfg = OmegaConf.structured(hp)
    yaml_str = OmegaConf.to_yaml(cfg)
    reloaded = OmegaConf.create(yaml_str)
    assert reloaded.S == 2
    assert float(reloaded.dec_lr) == pytest.approx(1e-3)
