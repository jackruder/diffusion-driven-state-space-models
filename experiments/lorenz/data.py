"""Data-module config for the Lorenz experiment family.

Lorenz 63 direct observation: D=3 (x, y, z), T=64.
At dt=0.05 each sequence spans ~3 Lyapunov times, giving 1-3
lobe-switching events per trajectory on average.
"""

from __future__ import annotations

from ddssm.data.presets import LorenzDirect

__all__ = ["LorenzDirect"]
