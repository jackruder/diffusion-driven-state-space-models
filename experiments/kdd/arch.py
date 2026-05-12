"""Explicit architectural defaults for the KDD family.

KDD uses a larger hidden footprint than the synthetic family — D=6,
real-world covariates (3-d), and longer sequences justify the wider
context producer + GRU summary. All low-level knobs are spelled out here
so they're visible at the experiment site rather than living as silent
defaults in :mod:`ddssm.builders`.
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

KDDTime = TimeMixer(type="conv", kernel_size=3, gru_layers=1)
KDDFeature = FeatureMixer(type="transformer", nheads=8, n_layers=1)


# ---- residual blocks --------------------------------------------------------

KDDResBlock = ResidualBlock(time=KDDTime, feature=KDDFeature)
KDDDiffResBlock = DiffResidualBlock(time=KDDTime, feature=KDDFeature)


# ---- context producer -------------------------------------------------------

KDDContext = Context(
    channels=8,
    num_layers=2,
    residual_block=KDDResBlock,
)


# ---- gaussian heads ---------------------------------------------------------

KDDClampedHead = Head(clamp_logvar_min=-10.0)
KDDPlainHead = Head()


# ---- future summary ---------------------------------------------------------

# Default GRU summary footprint — 64-dim, 2 layers — suitable for the
# 6-dim multivariate KDD series.
KDDFutSum = GRUFutSum(summary_dim=64, num_layers=2, gru_layers=1)


__all__ = [
    "KDDTime", "KDDFeature",
    "KDDResBlock", "KDDDiffResBlock",
    "KDDContext", "KDDClampedHead", "KDDPlainHead", "KDDFutSum",
]
