"""Schema tests for ``DDSSMModelConfig`` and its sub-dataclasses."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest

from ddssm.model.config import ModelConfig
from ddssm.model.ddssm_config import (
    DDSSMModelConfig,
    DDSSMModelKnobs,
    DDSSMShape,
    DDSSMTrainingHparams,
)
from ddssm.nn.aux_posterior import AuxPosterior


def _minimal_aux():
    """Zero-arg AuxPosterior instance — no builds() wrap needed for schema tests."""
    return AuxPosterior(latent_dim=2, j=1, hidden_dim=8, n_layers=1)


def test_shape_defaults():
    s = DDSSMShape()
    assert s.j == 1 and s.data_dim == 1 and s.latent_dim == 1
    assert s.emb_time_dim == 16
    assert s.T_max == 32


def test_model_knobs_defaults():
    k = DDSSMModelKnobs()
    assert k.S == 1
    assert k.logvar_min == -13.0
    assert k.logvar_max == 13.0
    assert k.mask_emb_dim == 8
    assert k.recon_time_chunk is None
    assert not k.recon_grad_checkpoint


def test_training_hparams_subclasses_model_config():
    assert issubclass(DDSSMTrainingHparams, ModelConfig)
    hp = DDSSMTrainingHparams()
    assert isinstance(hp, ModelConfig)
    assert hp.batch_size == 16
    assert hp.enc_lr == 5e-4


def test_training_hparams_has_no_duplicated_model_fields():
    """The duplicated fields (S, logvar_min, logvar_max, t_chunk) that lived in
    ``DDSSMHyperParamsConf`` are gone from the training slice — they belong on
    ``DDSSMModelKnobs``."""
    names = {f.name for f in fields(DDSSMTrainingHparams)}
    for gone in ("S", "logvar_min", "logvar_max", "t_chunk"):
        assert gone not in names, f"{gone} should not be on DDSSMTrainingHparams"


def test_ddssm_model_config_is_dataclass_and_model_config():
    assert is_dataclass(DDSSMModelConfig)
    assert issubclass(DDSSMModelConfig, ModelConfig)


def test_ddssm_model_config_requires_aux_posterior():
    """``__post_init__`` rejects the missing-required-slot case at config time."""
    with pytest.raises(ValueError, match="aux_posterior is required"):
        DDSSMModelConfig()


def test_ddssm_model_config_batch_size_passthrough():
    """``batch_size`` property delegates to the training slice so
    ``Experiment.train``'s ``getattr(hparams, 'batch_size', None)`` sync
    keeps working when ``Experiment.hparams`` holds a full DDSSMModelConfig."""
    cfg = DDSSMModelConfig(aux_posterior=_minimal_aux())
    assert cfg.batch_size == 16
    cfg2 = DDSSMModelConfig(
        aux_posterior=_minimal_aux(),
        training=DDSSMTrainingHparams(batch_size=32),
    )
    assert cfg2.batch_size == 32


def test_missing_builders_are_importable():
    """The new encoder/decoder builders (commit 1 additions) exist and target
    the right runtime classes."""
    from hydra_zen import get_target

    from ddssm.experiment.builders import (
        ARFlowEncoderB,
        IdentityDecoderB,
        IdentityEncoderB,
    )
    from ddssm.model.decoder import IdentityDecoder
    from ddssm.model.encoder import ARFlowEncoder, IdentityEncoder

    assert get_target(ARFlowEncoderB) is ARFlowEncoder
    assert get_target(IdentityEncoderB) is IdentityEncoder
    assert get_target(IdentityDecoderB) is IdentityDecoder
