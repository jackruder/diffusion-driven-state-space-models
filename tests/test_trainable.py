"""Tests for trainable-mask wiring through Experiment.train."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from ddssm.experiment import Experiment, TrainingScalars
from ddssm.stages import StageTrainableConf


def test_trainable_modules_defaults_all_true():
    t = StageTrainableConf()
    assert t.encoder and t.decoder and t.z_init and t.transition and t.baseline


def test_recon_only_freezes_transition():
    t = StageTrainableConf(encoder=True, decoder=True, z_init=True, transition=False)
    assert not t.transition
    assert t.encoder and t.decoder and t.z_init


def test_train_calls_set_trainable_when_specified():
    """Experiment.train invokes trainer._set_trainable iff training.trainable is set."""
    expt = Experiment.__new__(Experiment)
    expt.seed = None
    expt.wandb_config = None
    expt.objective = None
    expt.training = TrainingScalars(
        steps=0, log_every=1,
        trainable=StageTrainableConf(encoder=True, decoder=True, z_init=True, transition=False),
    )

    fake_data = MagicMock()
    fake_data.train_loader.return_value = None  # short-circuit fit
    expt.data = fake_data

    fake_model = MagicMock()
    fake_model.parameters.return_value = []
    expt.model = fake_model

    fake_trainer = MagicMock()
    expt.build_trainer = MagicMock(return_value=fake_trainer)

    expt.train(device=torch.device("cpu"), run_dir="/tmp/_set_trainable_test")
    assert expt.trainer is fake_trainer
    fake_trainer._set_trainable.assert_called_once_with(expt.training.trainable)


def test_train_skips_set_trainable_when_none():
    expt = Experiment.__new__(Experiment)
    expt.seed = None
    expt.wandb_config = None
    expt.objective = None
    expt.training = TrainingScalars(steps=0, log_every=1, trainable=None)

    fake_data = MagicMock()
    fake_data.train_loader.return_value = None
    expt.data = fake_data
    expt.model = MagicMock(parameters=MagicMock(return_value=[]))
    fake_trainer = MagicMock()
    expt.build_trainer = MagicMock(return_value=fake_trainer)

    expt.train(device=torch.device("cpu"), run_dir="/tmp/_set_trainable_none")
    fake_trainer._set_trainable.assert_not_called()
