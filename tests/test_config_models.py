# tests/test_config_models.py
"""Tests for the hydra-zen configuration layer."""

import pytest
from omegaconf import OmegaConf

from ddssm.model.dssd import DDSSMHyperParamsConf
from ddssm.model.ddssm_config import DDSSMModelKnobs


def test_hyperparams_defaults():
    hp = DDSSMHyperParamsConf()
    assert hp.enc_lr == pytest.approx(5e-4)
    # ``S`` / ``logvar_min`` / ``logvar_max`` moved to DDSSMModelKnobs
    # (model-side, not training-side) in the nested-config refactor.
    knobs = DDSSMModelKnobs()
    assert knobs.S == 1
    assert knobs.logvar_min == pytest.approx(-13.0)


def test_hyperparams_override():
    hp = DDSSMHyperParamsConf(enc_lr=1e-3, batch_size=32)
    assert hp.enc_lr == pytest.approx(1e-3)
    assert hp.batch_size == 32
    # Model-side scalar overrides live on DDSSMModelKnobs, not the training slice.
    knobs = DDSSMModelKnobs(S=4)
    assert knobs.S == 4


def test_yaml_roundtrip(tmp_path):
    """Config should survive an OmegaConf YAML round-trip."""
    hp = DDSSMHyperParamsConf(dec_lr=1e-3)
    cfg = OmegaConf.structured(hp)
    yaml_str = OmegaConf.to_yaml(cfg)
    reloaded = OmegaConf.create(yaml_str)
    assert float(reloaded.dec_lr) == pytest.approx(1e-3)
