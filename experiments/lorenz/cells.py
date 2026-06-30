"""Structural cell grid for the Lorenz sweep study.

Two axes:
    latent_dim  ∈ {4, 8}   — does extra latent capacity help capture the
                              bimodal lobe-switching structure in 3D Lorenz?
    frozen_enc  ∈ {False, True} — does freezing the encoder in stage 2
                                  stabilise the diffusion score net?

4 cells total; each is paired with a Lorenz Optuna sweep so HPO finds the
best training hyperparameters for that structural configuration.
"""

from __future__ import annotations

from typing import Iterator, NamedTuple

LATENT_DIMS: tuple[int, ...] = (4, 8)


class LorenzCell(NamedTuple):
    """One point of the Lorenz structural grid: (latent_dim, frozen_enc)."""

    latent_dim: int
    frozen_enc: bool

    @property
    def name(self) -> str:
        enc_tag = "frozen_enc" if self.frozen_enc else "open_enc"
        return f"lorenz_{self.latent_dim}d_{enc_tag}"


def iter_cells() -> Iterator[LorenzCell]:
    """Yield a :class:`LorenzCell` for every point of the grid."""
    for latent_dim in LATENT_DIMS:
        for frozen_enc in (False, True):
            yield LorenzCell(latent_dim, frozen_enc)


__all__ = ["LATENT_DIMS", "LorenzCell", "iter_cells"]
