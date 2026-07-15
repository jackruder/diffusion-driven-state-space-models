"""Tests for the :class:`TrainingScalars` surface after trainable-mask removal."""

from __future__ import annotations

import dataclasses

from ddssm.experiment import TrainingScalars


def test_training_scalars_has_no_trainable_field():
    """The ``trainable`` field was removed along with staged training."""
    field_names = {f.name for f in dataclasses.fields(TrainingScalars)}
    assert "trainable" not in field_names


def test_fit_kwargs_keys():
    """``fit_kwargs`` returns exactly the runtime knobs forwarded to
    :meth:`DDSSMTrainer.fit`."""
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
