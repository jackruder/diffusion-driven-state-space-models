"""The init-centering ablation's dataset axis — single source of truth.

Each :class:`AblationDataset` names the underlying ``SyntheticDataModule``
preset (from :mod:`ddssm.data.presets`) plus the model shape at the *tiny*
size. The paper-headline size doubles ``latent_dim`` (CONTEXT.md § Size
axis); ``data_dim`` is unchanged. Add a dataset here to extend the study's
dataset axis — nothing else needs editing.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from ddssm.data.presets import NonlinBimodalLift1D, NonlinBimodalLiftMV


@dataclass(frozen=True)
class AblationDataset:
    """One point on the study's dataset axis."""

    label: str  # short axis label used in preset names, e.g. "1d" / "mv"
    data_preset: Any  # the hydra-zen Synthetic config (D + expose_gt baked in)
    data_dim: int  # observation dim D (must match ``data_preset``)
    latent_dim: int  # model latent dim at the *tiny* size


ABLATION_DATASETS: tuple[AblationDataset, ...] = (
    AblationDataset("1d", NonlinBimodalLift1D, data_dim=1, latent_dim=1),
    AblationDataset("mv", NonlinBimodalLiftMV, data_dim=8, latent_dim=4),
)

# Paper-headline size: latent_dim is scaled by this factor; data_dim is held.
PAPER_LATENT_MULT = 2


def paper_latent(ds: AblationDataset) -> int:
    """The paper-headline ``latent_dim`` for a dataset (2× the tiny dim)."""
    return ds.latent_dim * PAPER_LATENT_MULT


def dataset_by_label(label: str) -> AblationDataset:
    """Look up an :class:`AblationDataset` by its axis label.

    Args:
        label: Short label such as ``"1d"`` or ``"mv"``.

    Returns:
        The matching :class:`AblationDataset`.

    Raises:
        KeyError: If no registered dataset has the given label.
    """
    for ds in ABLATION_DATASETS:
        if ds.label == label:
            return ds
    known = ", ".join(d.label for d in ABLATION_DATASETS)
    raise KeyError(f"unknown dataset label {label!r}; known: {known}")


__all__ = [
    "ABLATION_DATASETS",
    "PAPER_LATENT_MULT",
    "AblationDataset",
    "dataset_by_label",
    "paper_latent",
]
