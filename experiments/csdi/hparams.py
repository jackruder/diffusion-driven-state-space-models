"""Hparams (CSDI config) presets for the CSDI experiment family.

``CSDIHparams`` is the hydra-zen ``builds`` of :class:`ddssm.adapters.csdi.CSDIConfig`
(architecture + optimizer knobs all live in the config). Two named instances:

* ``CSDISmokeHparams`` — a tiny arch (layers=2, channels=32, num_steps=10,
  nheads=8) sized for a fast CPU fit. ``channels`` is divisible by ``nheads``
  (CSDI's transformer requires ``nheads | channels``); ``target_dim=2`` matches
  the smoke data's ``D``.
* ``CSDISolarHparams`` — paper-scale defaults for GluonTS Solar (``target_dim=137``).
"""

from __future__ import annotations

from hydra_zen import builds

from ddssm.adapters.csdi import CSDIConfig

CSDIHparams = builds(CSDIConfig, populate_full_signature=True)


# Tiny smoke arch: nheads=8 DIVIDES channels=32 (CSDI transformer constraint).
# target_dim=2 == SmokeData.d; num_steps=10 keeps diffusion sampling fast.
CSDISmokeHparams = CSDIHparams(
    target_dim=2,
    layers=2,
    channels=32,
    nheads=8,
    diffusion_embedding_dim=32,
    num_steps=10,
    timeemb=16,
    featureemb=8,
    batch_size=8,
)


# Paper-scale Solar preset (GluonTS solar has 137 series). Architecture fields
# carry the upstream CSDI defaults; only target_dim + batch_size are set here.
CSDISolarHparams = CSDIHparams(
    target_dim=137,
    batch_size=8,
)


__all__ = ["CSDIHparams", "CSDISmokeHparams", "CSDISolarHparams"]
