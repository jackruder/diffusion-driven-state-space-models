"""Single source of truth for the init-centering ablation grid.

The library Study (:mod:`.study`) registers one preset per
``(cell, dataset)`` combination via :func:`iter_cells`; the parametric
factory test in :mod:`tests.test_init_centering_factory` and the
Phase-B per-cell integration test consume the same enumerator, so the
grid definition lives in exactly one place.

The grid is the product

    baseline_form  ∈ {zero, persistence, linear, mlp}
    baseline_mode  ∈ {pinned, learnable}
    tracking_mode  ∈ {fixed, per_t}

with the auto-degenerate clamp from
``experiments/init_centering/model.py:_PARAM_FREE_FORMS``: parameter-
free baselines (``zero``, ``persistence``) drop the ``learnable`` mode
because they have no μ_p parameters to learn.  That removes 4 cells
(2 forms × 1 mode × 2 tracking) and yields 12 distinct triples.

The ``global_ema`` tracking mode (single scalar EMA-tracked σ_data²)
was dropped from the ablation — only ``fixed`` (σ_data² = 1) and
``per_t`` (time-varying buffer) are studied.  The underlying
:class:`ddssm.model.centering.sigma_data.SigmaDataBuffer` still supports
``global_ema`` as a valid mode; it just isn't a cell in this grid.
"""

from __future__ import annotations

from typing import Iterator, NamedTuple

BASELINE_FORMS: tuple[str, ...] = ("zero", "persistence", "linear", "mlp")
BASELINE_MODES: tuple[str, ...] = ("pinned", "learnable")
TRACKING_MODES: tuple[str, ...] = ("fixed", "per_t")

# Forms with no learnable μ_p parameters — they drop the ``learnable`` mode
# both here (cell enumeration) and in ``model.py`` (the auto-clamp), which
# imports this single definition rather than mirroring it.
_PARAM_FREE_FORMS: frozenset[str] = frozenset({"zero", "persistence"})

# The canonical cell — also the default in ``_build_init_centering_model``.
# Reused by the V2-reduction test.
CANONICAL_CELL: tuple[str, str, str] = ("mlp", "pinned", "per_t")


class Cell(NamedTuple):
    """One point of the grid: ``(baseline_form, baseline_mode, tracking_mode)``.

    A ``NamedTuple`` so it still unpacks like the original triple
    (``for form, mode, tracking in iter_cells()``) and compares equal to a plain
    tuple, while carrying a self-describing ``.name``.
    """

    baseline_form: str
    baseline_mode: str
    tracking_mode: str

    @property
    def name(self) -> str:
        return cell_name(self.baseline_form, self.baseline_mode, self.tracking_mode)


def iter_cells() -> Iterator[Cell]:
    """Yield a :class:`Cell` for every point of the post-auto-clamp grid."""
    for form in BASELINE_FORMS:
        modes: tuple[str, ...] = (
            ("pinned",) if form in _PARAM_FREE_FORMS else BASELINE_MODES
        )
        for mode in modes:
            for tracking in TRACKING_MODES:
                yield Cell(form, mode, tracking)


def cell_name(form: str, mode: str, tracking: str) -> str:
    """Hydra-friendly preset name for a cell, e.g. ``init_mlp_pinned_per_t``."""
    return f"init_{form}_{mode}_{tracking}"


__all__ = [
    "BASELINE_FORMS",
    "BASELINE_MODES",
    "CANONICAL_CELL",
    "Cell",
    "TRACKING_MODES",
    "cell_name",
    "iter_cells",
]
