"""Tests for the ``ModelConfig`` base and its DDSSM subclass.

Module 1 of the ``ModelAdapter`` refactor: every model family will define
a ``ModelConfig`` subclass as the uniform config currency held on
``Experiment.hparams``. This module introduces the base and makes the
existing DDSSM hyperparams dataclass subclass it — no field changes.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from hydra_zen import builds, instantiate

from ddssm.model.config import ModelConfig
from ddssm.model.dssd import DDSSMHyperParamsConf


def test_model_config_is_empty_marker_dataclass():
    """The base is a marker dataclass — no field is universally required.

    Not every model family batches; ``batch_size`` lives on subclasses that
    need it (e.g. ``DDSSMHyperParamsConf``). ``Experiment`` reaches for
    optional knobs via ``getattr(hparams, "field", None)``.
    """
    assert is_dataclass(ModelConfig)
    ModelConfig()  # zero-arg instantiable
    assert {f.name for f in fields(ModelConfig)} == set()


def test_ddssm_hyperparams_subclasses_model_config():
    """``DDSSMHyperParamsConf`` inherits from and instantiates as ``ModelConfig``."""
    assert issubclass(DDSSMHyperParamsConf, ModelConfig)
    assert isinstance(DDSSMHyperParamsConf(), ModelConfig)


def test_ddssm_hyperparams_defaults_unchanged():
    """Subclassing must not perturb existing defaults."""
    cfg = DDSSMHyperParamsConf()
    assert cfg.batch_size == 16
    assert cfg.S == 1
    assert cfg.weight_decay == 1e-4
    # Additional spot-checks against silent default drift.
    assert cfg.ema_decay == 0.999
    assert cfg.grad_accum_steps == 4
    assert cfg.enc_lr == 5e-4


def test_ddssm_hyperparams_kwarg_construction_roundtrips():
    """All existing callers pass by keyword — that must still work."""
    cfg = DDSSMHyperParamsConf(batch_size=8, enc_lr=1e-3, weight_decay=2e-4)
    assert cfg.batch_size == 8
    assert cfg.enc_lr == 1e-3
    assert cfg.weight_decay == 2e-4


def test_hydra_zen_builds_ddssm_hyperparams():
    """hydra-zen's ``builds(..., populate_full_signature=True)`` still works."""
    Conf = builds(DDSSMHyperParamsConf, populate_full_signature=True, batch_size=4)
    cfg = instantiate(Conf)
    assert isinstance(cfg, DDSSMHyperParamsConf)
    assert cfg.batch_size == 4
