"""Tests for the Study abstraction + the init-centering study instance."""

from __future__ import annotations

import torch
from hydra_zen import instantiate

from experiments.init_centering.study import INIT_CENTERING_STUDY


def test_study_has_24_unique_points() -> None:
    names = INIT_CENTERING_STUDY.names()
    assert len(INIT_CENTERING_STUDY.points) == 24
    assert len(names) == 24
    assert len(set(names)) == 24


def test_point_names_are_cell_then_dataset() -> None:
    for p in INIT_CENTERING_STUDY.points:
        assert p.name == f"{p.tags['cell']}__{p.tags['dataset']}"
        assert p.tags["dataset"] in {"1d", "mv"}


def test_select_filters_by_tags() -> None:
    mlp = INIT_CENTERING_STUDY.select(baseline_form="mlp")
    assert mlp and all(p.tags["baseline_form"] == "mlp" for p in mlp)

    mv = INIT_CENTERING_STUDY.select(dataset="mv")
    assert len(mv) == 12  # 12 cells, one per cell on the mv dataset

    combo = INIT_CENTERING_STUDY.select(baseline_form="mlp", dataset="1d")
    assert all(
        p.tags["baseline_form"] == "mlp" and p.tags["dataset"] == "1d"
        for p in combo
    )

    # Collection-valued filter (membership).
    param_free = INIT_CENTERING_STUDY.select(baseline_form={"zero", "identity"})
    assert param_free and all(
        p.tags["baseline_form"] in {"zero", "identity"} for p in param_free
    )


def test_register_calls_store_once_per_point() -> None:
    collected: list[str] = []

    def fake_store(config, *, name: str) -> None:  # noqa: ANN001
        collected.append(name)

    INIT_CENTERING_STUDY.register(fake_store)
    assert sorted(collected) == sorted(INIT_CENTERING_STUDY.names())


def test_points_bake_the_real_dataset_and_matching_dims() -> None:
    """Each point's config carries the real lift dataset (not harmonic) +
    consistent ``data_dim`` between the data module and the model."""
    expect = {
        "1d": ("nonlinear-bimodal-lift", 1, 1),
        "mv": ("nonlinear-bimodal-lift-mv", 8, 4),
    }
    for p in INIT_CENTERING_STUDY.points:
        mode, data_dim, latent = expect[p.tags["dataset"]]
        # Read straight off the hydra-zen config (no instantiation needed).
        assert p.config.data.mode == mode
        assert p.config.data.D == data_dim
        assert p.config.model.data_dim == data_dim
        assert p.config.model.latent_dim == latent


def test_paper_size_override_doubles_latent_dim() -> None:
    for p in INIT_CENTERING_STUDY.points:
        assert p.size_overrides("tiny") == []
        paper = p.size_overrides("paper")
        assert len(paper) == 1
        assert paper[0].startswith("experiment.model.latent_dim=")
    # 1d tiny latent 1 -> paper 2; mv tiny latent 4 -> paper 8.
    assert INIT_CENTERING_STUDY.point("init_mlp_pinned_per_t__1d").size_overrides(
        "paper"
    ) == ["experiment.model.latent_dim=2"]
    assert INIT_CENTERING_STUDY.point("init_mlp_pinned_per_t__mv").size_overrides(
        "paper"
    ) == ["experiment.model.latent_dim=8"]


def test_one_point_forward_backward_is_finite_with_grads() -> None:
    """A built point trains for one step without NaN/shape breakage.

    Guards against presets that *build* but produce a broken loss (the
    catalogue count/instantiate tests only check params > 0).
    """
    exp = instantiate(
        INIT_CENTERING_STUDY.point("init_zero_pinned_fixed__1d").config
    )
    model = exp.model
    model.stage_selector = "stage_1"  # closed-form transition; cheap
    B, D, T = 2, model.data_dim, 8
    torch.manual_seed(0)
    components, _metrics, _ = model(
        torch.randn(B, D, T),
        torch.ones(B, D, T),
        torch.arange(T).expand(B, T).clone().long(),
    )
    loss = components.total()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
        for q in model.parameters()
        if q.requires_grad
    )
