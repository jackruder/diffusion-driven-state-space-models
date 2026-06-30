"""Lorenz attractor experiment family.

Validates whether the diffusion transition can learn the bimodal
lobe-switching residual on Lorenz 63 data (D=3, T=64).
"""

from . import cells, data, evals, experiments, hparams, study, sweeps

__all__ = ["cells", "data", "evals", "experiments", "hparams", "study", "sweeps"]
