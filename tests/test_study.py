"""Tests for the library Study abstraction + the init-centering instance."""

from __future__ import annotations

import pytest
import torch
from hydra_zen import instantiate

from ddssm.study import Axis, Study, StudyPoint
from ddssm.launch import register_study, STUDY_REGISTRY
from experiments.init_centering.study import INIT_CENTERING_STUDY


def test_study_rejects_duplicate_point_names() -> None:
    """A name_point/axis-key collision raises instead of silently overwriting."""
    with pytest.raises(ValueError, match="duplicate point names"):
        Study.from_axes(
            "collide",
            axes=[Axis("x", ["a", "b"], key=lambda v: v)],
            build=lambda c: {"cfg": c["x"]},
            name_point=lambda tags: "same",  # both coords -> same name
            launch=lambda p: None,
        )


def test_register_study_into_publishes_points() -> None:
    """register_study(study, into=store) publishes every point in one call."""
    published: dict[str, object] = {}
    study = Study.from_axes(
        "study_into_test",
        axes=[Axis("x", ["a", "b"], key=lambda v: v)],
        build=lambda c: {"cfg": c["x"]},
        launch=lambda p: None,
    )
    register_study(study, into=lambda config, name: published.__setitem__(name, config))
    assert set(published) == {"a", "b"}
    assert STUDY_REGISTRY["study_into_test"] is study


# ---------------------------------------------------------------------------
# Library primitives (a tiny synthetic study)
# ---------------------------------------------------------------------------


def test_from_axes_cross_product_tags_and_select() -> None:
    s = Study.from_axes(
        "t",
        axes=[
            Axis("x", [1, 2], key=str),
            Axis("y", ["a", "b"], key=str, tags=lambda v: {"y_upper": v.upper()}),
        ],
        build=lambda coords: dict(coords),
        name_point=lambda tags: f"{tags['x']}_{tags['y']}",
        launch=lambda p: None,
    )
    assert len(s.points) == 4
    p = s.point("1_a")
    assert p.coords == {"x": 1, "y": "a"}
    assert p.config == {"x": 1, "y": "a"}            # build received raw values
    assert p.tags == {"x": "1", "y": "a", "y_upper": "A"}  # axis.tags merged in
    assert {pt.name for pt in s.select(x="1")} == {"1_a", "1_b"}
    assert {pt.name for pt in s.select(y_upper="B")} == {"1_b", "2_b"}


def test_from_axes_filter_drops_combos() -> None:
    s = Study.from_axes(
        "t",
        axes=[Axis("x", [1, 2, 3], key=str)],
        build=lambda c: c,
        launch=lambda p: None,
        filter=lambda c: c["x"] != 2,
    )
    assert sorted(p.coords["x"] for p in s.points) == [1, 3]


def test_from_points_escape_hatch_and_register() -> None:
    pts = [StudyPoint("a", {"cfg": 1}, {"k": "v"}, {"k": 1})]
    s = Study.from_points("t", pts, launch=lambda p: None)
    assert s.names() == ["a"]
    collected: list = []
    s.register(lambda config, *, name: collected.append((config, name)))
    assert collected == [({"cfg": 1}, "a")]


# ---------------------------------------------------------------------------
# The init-centering study instance
# ---------------------------------------------------------------------------


def test_init_study_has_24_points_with_full_tags() -> None:
    assert len(INIT_CENTERING_STUDY.points) == 24
    assert len(set(INIT_CENTERING_STUDY.names())) == 24
    for p in INIT_CENTERING_STUDY.points:
        assert p.name == f"{p.tags['cell']}__{p.tags['dataset']}"
        assert {"cell", "baseline_form", "baseline_mode", "tracking_mode", "dataset"} <= set(p.tags)


def test_init_study_select_filters() -> None:
    assert len(INIT_CENTERING_STUDY.select(baseline_form="mlp")) == 8   # 2 modes × 2 tracking × 2 ds
    assert len(INIT_CENTERING_STUDY.select(dataset="mv")) == 12
    assert len(INIT_CENTERING_STUDY.select(baseline_form="zero", dataset="1d")) == 2  # 2 tracking


def test_init_points_bake_real_dataset_and_dims() -> None:
    expect = {"1d": ("nonlinear-bimodal-lift", 1, 1), "mv": ("nonlinear-bimodal-lift-mv", 8, 4)}
    for p in INIT_CENTERING_STUDY.points:
        mode, data_dim, latent = expect[p.tags["dataset"]]
        assert p.config.data.mode == mode
        assert p.config.data.D == data_dim
        assert p.config.model.data_dim == data_dim
        assert p.config.model.latent_dim == latent


def test_init_variants() -> None:
    v = INIT_CENTERING_STUDY.variants
    assert "tiny" in v and "paper" in v and "smoke" in v
    p1d = INIT_CENTERING_STUDY.point("init_mlp_pinned_per_t__1d")
    pmv = INIT_CENTERING_STUDY.point("init_mlp_pinned_per_t__mv")
    assert v["tiny"](p1d) == []
    assert v["paper"](p1d) == ["experiment.model.latent_dim=2"]
    assert v["paper"](pmv) == ["experiment.model.latent_dim=8"]
    assert any("n_pretrain=5" in o for o in v["smoke"](p1d))


def test_one_point_forward_backward_is_finite_with_grads() -> None:
    exp = instantiate(INIT_CENTERING_STUDY.point("init_zero_pinned_fixed__1d").config)
    model = exp.model
    model.stage_selector = "stage_1"
    B, D, T = 2, model.data_dim, 8
    torch.manual_seed(0)
    components, _m, _ = model(
        torch.randn(B, D, T), torch.ones(B, D, T),
        torch.arange(T).expand(B, T).clone().long(),
    )
    loss = components.total()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
        for q in model.parameters() if q.requires_grad
    )
