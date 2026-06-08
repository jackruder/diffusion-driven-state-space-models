"""Lorenz attractor experiment family.

Validates whether the diffusion transition can learn the bimodal
lobe-switching residual on Lorenz 63 data (D=3, T=64).
"""

from . import data, hparams, experiments

__all__ = ["data", "hparams", "experiments"]
