"""Single source of truth for the init-centering ablation grid.

The library Study (:mod:`.study`) registers one preset per
``(cell, dataset)`` combination via :func:`iter_cells`; the parametric
factory test in :mod:`tests.test_init_centering_factory` consumes the
same enumerator, so the grid definition lives in exactly one place.

The grid is the product

    baseline_form  ∈ {zero, persistence}
    tracking_mode  ∈ {fixed, per_t}

for a total of 4 cells. Both baselines are parameter-free (``σ_p² = 1``
constant); the diffusion transition consumes them by reference. The
``global_ema`` σ_data mode remains a supported
:class:`ddssm.model.centering.sigma_data.SigmaDataBuffer` value; it just
isn't a cell in this grid.
"""

from __future__ import annotations

from typing import NamedTuple
from collections.abc import Iterator

BASELINE_FORMS: tuple[str, ...] = ("zero", "persistence")
TRACKING_MODES: tuple[str, ...] = ("fixed", "per_t")

# The canonical cell — also the default in ``_build_init_centering_model``.
CANONICAL_CELL: tuple[str, str] = ("persistence", "per_t")


class Cell(NamedTuple):
    """One point of the grid: ``(baseline_form, tracking_mode)``."""

    baseline_form: str
    tracking_mode: str

    @property
    def name(self) -> str:
        return cell_name(self.baseline_form, self.tracking_mode)


def iter_cells() -> Iterator[Cell]:
    """Yield a :class:`Cell` for every point of the grid."""
    for form in BASELINE_FORMS:
        for tracking in TRACKING_MODES:
            yield Cell(form, tracking)


def cell_name(form: str, tracking: str) -> str:
    """Hydra-friendly preset name for a cell, e.g. ``init_persistence_per_t``."""
    return f"init_{form}_{tracking}"


__all__ = [
    "BASELINE_FORMS",
    "CANONICAL_CELL",
    "TRACKING_MODES",
    "Cell",
    "cell_name",
    "iter_cells",
]
