"""Tests for the init-centering ablation grid (the study's cell axis).

The grid is enumerated by
:func:`experiments.init_centering.cells.iter_cells`; every triple it
yields must round-trip through

* the experiment store (registered under
  :func:`experiments.init_centering.cells.cell_name`),
* the Hydra CLI bridge (``experiment=<cell-name>``), and
* :func:`hydra_zen.instantiate` (so the resulting model's
  ``baseline_form / baseline_mode / tracking_mode`` slots agree with
  the cell's triple).

The expensive ``instantiate(...)`` checks run on three representative
cells only — the parametric factory test in
:mod:`tests.test_init_centering_factory` already covers every cell at
the factory level.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
from hydra_zen import instantiate
from hydra.core.global_hydra import GlobalHydra

from ddssm.experiment.stores import store
from ddssm.experiment.registry import register_experiments
from ddssm.model.centering.baselines import (
    MLPBaseline,
    ZeroBaseline,
    LinearBaseline,
    PersistenceBaseline,
)
from experiments.init_centering.cells import (
    BASELINE_FORMS,
    BASELINE_MODES,
    CANONICAL_CELL,
    TRACKING_MODES,
    cell_name,
    iter_cells,
)

CONF_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "ddssm" / "conf"
).as_posix()


@pytest.fixture(autouse=True)
def _clear_global_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    register_experiments()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# Enumerator
# ---------------------------------------------------------------------------


def test_iter_cells_yields_exactly_12_distinct_triples() -> None:
    """The post-auto-clamp grid has 12 unique cells."""
    cells = list(iter_cells())
    assert len(cells) == 12
    assert len(set(cells)) == 12
    # Spot-check the math: 2 param-free baselines × pinned-only × 2 tracking
    # + 2 parametric baselines × 2 modes × 2 tracking = 4 + 8.
    pinned_only = [c for c in cells if c[0] in {"zero", "persistence"}]
    assert len(pinned_only) == 4
    assert all(c[1] == "pinned" for c in pinned_only)


def test_iter_cells_axes_match_advertised_sets() -> None:
    """Every cell's axes are drawn from the public axis tuples."""
    for form, mode, tracking in iter_cells():
        assert form in BASELINE_FORMS
        assert mode in BASELINE_MODES
        assert tracking in TRACKING_MODES


def test_canonical_cell_is_in_grid() -> None:
    """The canonical cell (MLP / Pinned / per-t) sits inside the grid."""
    assert CANONICAL_CELL in set(iter_cells())


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_all_cells_registered() -> None:
    """Every cell × dataset preset (``init_<cell>__<ds>``) is registered."""
    register_experiments()
    names = {name for _, name in store["experiment"]}
    for form, mode, tracking in iter_cells():
        cell = cell_name(form, mode, tracking)
        assert f"{cell}__1d" in names
        assert f"{cell}__mv" in names


def test_control_cells_no_longer_registered() -> None:
    """The control presets were dropped per ADR-0002.

    Defensive guard against accidental reintroduction: future code that
    needs handoff-protocol ablations should register them as plain
    ``ablation_canonical_*`` presets, not under the ``ctrl`` name.
    """
    register_experiments()
    names = {name for _, name in store["experiment"]}
    assert "init_canonical_ctrl_sigma0" not in names
    assert "init_canonical_ctrl_npretrain0" not in names


# ---------------------------------------------------------------------------
# Composition + axis propagation (spot-checked on three representative cells)
# ---------------------------------------------------------------------------


_REPRESENTATIVE_CELLS = [
    ("zero", "pinned", "fixed"),       # simplest; reduces to V2 (see other test)
    ("mlp", "pinned", "per_t"),        # canonical cell (== Phase-C pilot)
    ("linear", "learnable", "per_t"),  # all-three-axes-distinct exemplar
]


@pytest.mark.parametrize("form,mode,tracking", _REPRESENTATIVE_CELLS)
def test_cell_axes_propagate_through_hydra(form, mode, tracking) -> None:
    """``experiment=<cell>__1d`` resolves to a model with the right axes."""
    name = f"{cell_name(form, mode, tracking)}__1d"
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.experiment.model.baseline_form == form
    assert cfg.experiment.model.baseline_mode == mode
    assert cfg.experiment.model.tracking_mode == tracking


@pytest.mark.parametrize(
    "form,expected_baseline_cls",
    [
        ("zero", ZeroBaseline),
        ("persistence", PersistenceBaseline),
        ("linear", LinearBaseline),
        ("mlp", MLPBaseline),
    ],
)
def test_cell_instantiates_with_correct_baseline_class(form, expected_baseline_cls) -> None:
    """Instantiating a cell preset materialises the right baseline class."""
    # Use the (form, pinned, fixed) cell on the 1d dataset for each form
    # (always valid post-clamp).
    cfg = store["experiment"]["experiment", f"{cell_name(form, 'pinned', 'fixed')}__1d"]
    exp = instantiate(cfg)
    assert isinstance(exp.model.baseline, expected_baseline_cls)


# ``test_canonical_cell_preset_matches_pilot_axes`` was removed when the
# ``init_centering_pilot`` preset was retired during the smoke
# restructure (CONTEXT.md § "pilot cell" is deliberately not used).
# The canonical cell still exists as a triple in ``cells.py`` and as
# named presets ``init_mlp_pinned_per_t__{1d,mv}`` in the study grid.


# Control-cell tests were removed when the controls themselves were
# dropped per ``docs/adr/0002-drop-canonical-controls.md``.
# ``test_control_cells_no_longer_registered`` above guards against
# accidental reintroduction.
