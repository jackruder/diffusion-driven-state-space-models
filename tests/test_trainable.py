"""Tests for the :class:`TrainingScalars.trainable` freeze-mask surface."""

from __future__ import annotations

import dataclasses

import torch

from ddssm.experiment import TrainingScalars
from ddssm.training.stages import TrainableConf


def test_training_scalars_carries_trainable_field():
    """``trainable`` is an optional field defaulting to None."""
    field_names = {f.name for f in dataclasses.fields(TrainingScalars)}
    assert "trainable" in field_names
    assert TrainingScalars().trainable is None


def test_training_scalars_accepts_trainable_conf():
    """``TrainingScalars(trainable=TrainableConf(...))`` stores the mask."""
    mask = TrainableConf(encoder=False, decoder=True, transition=True)
    ts = TrainingScalars(trainable=mask)
    assert ts.trainable is mask


def test_trainable_conf_defaults_all_trainable():
    """Zero-arg ``TrainableConf`` leaves every submodule trainable."""
    t = TrainableConf()
    assert t.encoder is True
    assert t.decoder is True
    assert t.transition is True


def test_fit_kwargs_keys():
    """``fit_kwargs`` returns exactly the runtime knobs forwarded to
    :meth:`DDSSMTrainer.fit`.

    ``trainable`` is intentionally NOT in ``fit_kwargs`` — it's consumed by
    the adapter (applied to the trainer before ``fit``), not by ``fit`` itself.
    """
    expected = {
        "total_steps",
        "log_every",
        "validate_every",
        "checkpoint_every",
        "checkpoint_prefix",
        "amp",
        "profile_steps",
        "resume_from",
    }
    assert set(TrainingScalars().fit_kwargs().keys()) == expected


def test_fit_kwargs_forwards_steps_as_total_steps():
    """``steps`` on the dataclass maps to ``total_steps`` in the fit kwargs."""
    assert TrainingScalars(steps=7).fit_kwargs()["total_steps"] == 7


def test_ddssm_adapter_applies_trainable_before_fit(tmp_path):
    """``TrainingScalars.trainable`` freezes flagged submodules before ``fit``.

    Wires a 1-step end-to-end fit through :class:`~ddssm.adapters.ddssm.DDSSMAdapter`
    with ``TrainableConf(encoder=False)``. After ``.fit``, encoder params must
    have ``requires_grad=False`` while decoder + transition stay ``True``.
    """
    from ddssm.adapters.ddssm import DDSSMAdapter
    from ddssm.data.datamodule import SyntheticDataModule
    from ddssm.model.dssd import DDSSMHyperParamsConf
    from tests.test_trainer import make_small_model, DATA_DIM

    module = make_small_model()
    adapter = DDSSMAdapter(config=DDSSMHyperParamsConf(batch_size=4), module=module)
    data = SyntheticDataModule(
        mode="lgssm", T=16, D=DATA_DIM, N_per_split=8, batch_size=4
    )
    training = TrainingScalars(
        steps=1,
        log_every=1,
        validate_every=0,
        trainable=TrainableConf(encoder=False, decoder=True, transition=True),
    )
    adapter.fit(
        data=data,
        training=training,
        device=torch.device("cpu"),
        csv_log_path=str(tmp_path / "metrics.csv"),
        tensorboard_dir=str(tmp_path / "tb"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert all(not p.requires_grad for p in module.encoder.parameters())
    assert any(p.requires_grad for p in module.decoder.parameters())
    assert any(p.requires_grad for p in module.transition.parameters())
