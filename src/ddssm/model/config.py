"""Base ``ModelConfig`` shared across every model family (leaf, stdlib-only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Marker base for each family's ``Experiment.hparams`` config.

    Deliberately empty: no field is universally required across every model
    family (e.g. not every family batches). Families subclass this and add
    their own fields — ``DDSSMHyperParamsConf`` for DDSSM, and one per
    baseline. ``Experiment`` reaches for optional knobs (``batch_size``,
    …) via ``getattr(hparams, "field", None)`` so families that don't
    carry them just skip that machinery.
    """
