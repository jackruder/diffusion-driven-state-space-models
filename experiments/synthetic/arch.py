"""Explicit architectural defaults for the synthetic-data family.

This module spells out every low-level architecture knob so that
encoders/decoders/z-inits/transitions don't rely on silent module-level
defaults in :mod:`ddssm.builders`. Every preset reachable through the
experiment store is built from these named handles; overriding via the
CLI/zen still works because each handle is a ``builds(...)`` partial
(or a zen-built dataclass).

Knobs grouped by what they affect:

* :data:`SmallTime` / :data:`SmallFeature` — mixer types used inside
  context-producer residual blocks.
* :data:`SmallResBlock` — residual block config (= time mixer + feature
  mixer), used by :class:`~ddssm.diffnets.ContextProducer`.
* :data:`SmallDiffResBlock` — diffusion residual block (extra
  diffusion-step conditioning), used by :class:`~ddssm.diffnets.CSDIUnet`.
* :data:`SmallContext` — context-producer settings (channels, num_layers
  and the residual-block stack).
* :data:`SmallHead` — Gaussian head clamp.
* :data:`TinyGRU` — future-summary backbone (1-layer GRU, 16-dim).
"""

from __future__ import annotations

from ddssm.builders import (
    Context,
    DiffResidualBlock,
    FeatureMixer,
    GRUFutSum,
    Head,
    ResidualBlock,
    TimeMixer,
)


# ---- mixers -----------------------------------------------------------------

SmallTime = TimeMixer(type="conv", kernel_size=3, gru_layers=1)
SmallFeature = FeatureMixer(type="transformer", nheads=8, n_layers=1)


# ---- residual blocks --------------------------------------------------------

# ContextProducer's residual block: 1-layer feature mixer is enough at
# the toy 1D/2D shapes.
SmallResBlock = ResidualBlock(
    time=SmallTime,
    feature=SmallFeature,
)

# Diffusion residual block: same mixer footprint, used inside CSDIUnet.
SmallDiffResBlock = DiffResidualBlock(
    time=SmallTime,
    feature=SmallFeature,
)


# ---- context producer -------------------------------------------------------

SmallContext = Context(
    channels=8,
    num_layers=2,
    residual_block=SmallResBlock,
)


# ---- gaussian head ----------------------------------------------------------

SmallHead = Head(clamp_logvar_min=-10.0)


# ---- future summary ---------------------------------------------------------

# Tiny GRU summary — single layer, 16-dim hidden state. The future
# summary is the dominant per-step cost in training (sequential over T)
# and overkill for toy synthetic data at this latent size.
TinyGRU = GRUFutSum(summary_dim=16, num_layers=1, gru_layers=1)


__all__ = [
    "SmallTime", "SmallFeature",
    "SmallResBlock", "SmallDiffResBlock",
    "SmallContext", "SmallHead", "TinyGRU",
]
