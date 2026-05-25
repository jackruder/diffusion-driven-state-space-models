"""Phase-D tests for the 18-cell ablation grid and the two control cells.

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
:mod:`tests.test_init_centering_factory` already covers all 18 at the
factory level.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate

from conf.registry import store
from ddssm._experiment_registry import register_experiments
from ddssm.centering.baselines import (
    IdentityBaseline,
    LinearBaseline,
    MLPBaseline,
    ZeroBaseline,
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


def test_iter_cells_yields_exactly_18_distinct_triples() -> None:
    """The post-auto-clamp grid has 18 unique cells."""
    cells = list(iter_cells())
    assert len(cells) == 18
    assert len(set(cells)) == 18
    # Spot-check the math: 2 param-free baselines × pinned-only × 3 tracking
    # + 2 parametric baselines × 2 modes × 3 tracking = 6 + 12.
    pinned_only = [c for c in cells if c[0] in {"zero", "identity"}]
    assert len(pinned_only) == 6
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


def test_all_18_cells_registered() -> None:
    """Every cell's preset name is registered in the experiment store."""
    register_experiments()
    names = {name for _, name in store["experiment"]}
    for form, mode, tracking in iter_cells():
        assert cell_name(form, mode, tracking) in names


def test_control_cells_registered() -> None:
    """Both control presets (sigma_pert=0, n_pretrain=0) are registered."""
    register_experiments()
    names = {name for _, name in store["experiment"]}
    assert "init_canonical_ctrl_sigma0" in names
    assert "init_canonical_ctrl_npretrain0" in names


# ---------------------------------------------------------------------------
# Composition + axis propagation (spot-checked on three representative cells)
# ---------------------------------------------------------------------------


_REPRESENTATIVE_CELLS = [
    ("zero", "pinned", "fixed"),       # simplest; reduces to V2 (see other test)
    ("mlp", "pinned", "per_t"),        # canonical cell (== Phase-C pilot)
    ("linear", "learnable", "global_ema"),  # all-three-axes-distinct exemplar
]


@pytest.mark.parametrize("form,mode,tracking", _REPRESENTATIVE_CELLS)
def test_cell_axes_propagate_through_hydra(form, mode, tracking) -> None:
    """``experiment=<cell-name>`` resolves to a model with the right axes."""
    name = cell_name(form, mode, tracking)
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.experiment.model.baseline_form == form
    assert cfg.experiment.model.baseline_mode == mode
    assert cfg.experiment.model.tracking_mode == tracking


@pytest.mark.parametrize(
    "form,expected_baseline_cls",
    [
        ("zero", ZeroBaseline),
        ("identity", IdentityBaseline),
        ("linear", LinearBaseline),
        ("mlp", MLPBaseline),
    ],
)
def test_cell_instantiates_with_correct_baseline_class(form, expected_baseline_cls) -> None:
    """Instantiating a cell preset materialises the right baseline class."""
    # Use the (form, pinned, fixed) cell for each form (always valid post-clamp).
    cfg = store["experiment"]["experiment", cell_name(form, "pinned", "fixed")]
    exp = instantiate(cfg)
    assert isinstance(exp.model.baseline, expected_baseline_cls)


def test_canonical_cell_preset_matches_pilot_axes() -> None:
    """The canonical cell preset and ``init_centering_pilot`` share their axes."""
    canonical = instantiate(
        store["experiment"]["experiment", cell_name(*CANONICAL_CELL)]
    )
    pilot = instantiate(store["experiment"]["experiment", "init_centering_pilot"])
    # Regression guard: if Phase C and Phase D drift, fail loudly.
    assert canonical.model.baseline_mode == pilot.model.baseline_mode
    assert canonical.model.sigma_data.tracking_mode == pilot.model.sigma_data.tracking_mode
    assert type(canonical.model.baseline) is type(pilot.model.baseline)


# ---------------------------------------------------------------------------
# Control cells
# ---------------------------------------------------------------------------


def test_sigma0_control_zeros_sigma_pert() -> None:
    """``init_canonical_ctrl_sigma0`` instantiates with ``sigma_pert == 0``."""
    cfg = store["experiment"]["experiment", "init_canonical_ctrl_sigma0"]
    exp = instantiate(cfg)
    stage2_handoff = exp.model.config.stages.stage_2.centering_handoff
    assert stage2_handoff is not None
    assert stage2_handoff.sigma_pert == 0.0


def test_npretrain0_control_zeros_stage1_steps() -> None:
    """``init_canonical_ctrl_npretrain0`` instantiates with stage-1 steps == 0."""
    cfg = store["experiment"]["experiment", "init_canonical_ctrl_npretrain0"]
    exp = instantiate(cfg)
    assert exp.model.config.stages.stage_1.steps == 0
