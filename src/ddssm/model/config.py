"""Base ``ModelConfig`` shared across every model family (leaf, stdlib-only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Uniform config currency held on ``Experiment.hparams``.

    Every model family (DDSSM, CSDI baseline, future adapters, …)
    subclasses this and adds its own fields. Only knobs the ``Experiment``
    composition root needs uniformly live here — presently just
    ``batch_size``, which ``Experiment`` syncs onto the data module.
    """

    batch_size: int = 16
