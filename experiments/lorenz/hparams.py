"""Hparams and multi-stage config for the Lorenz experiment family.

Reuses the standard SmokeHparams and StagesB factory from init_centering,
but increases the stage budgets to accommodate 3D nonlinear dynamics:

- n_pretrain=800: ~4× the smoke run (200), giving the MLP baseline
  time to converge on the coupled quadratic Lorenz dynamics.
- n_stage2=2000: ~2× the smoke run (1000), giving the diffusion score
  net time to learn the bimodal lobe-switching residual in 3D.
"""

from __future__ import annotations

from experiments.init_centering.hparams import SmokeHparams, StagesB, Training800

LorenzHparams = SmokeHparams

LorenzStages = StagesB(n_pretrain=800, n_stage2=2000)

__all__ = ["LorenzHparams", "LorenzStages", "Training800"]
