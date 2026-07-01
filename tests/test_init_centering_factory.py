"""Phase-B factory tests for the parametric init-centering model builder.

Verifies that :func:`_build_init_centering_model` honours the ablation
grid axes from ``init-experiment.org`` § Composition with the ablation
grid:

* the three cell axes round-trip into the resulting :class:`DDSSM_base`,
* the auto-degenerate clamp turns ``baseline_mode="learnable"`` into
  ``"pinned"`` for parameter-free baseline forms (zero / persistence),
* the default ``anchor_lambda`` depends on ``baseline_mode``.

All cells must build without error and the assembled stages factory
must produce a baseline-trainable mask consistent with the chosen
``baseline_mode``.
"""

from __future__ import annotations

import logging

import pytest

from ddssm.model.centering.baselines import (
    MLPBaseline,
    ZeroBaseline,
    LinearBaseline,
    PersistenceBaseline,
)
from experiments.init_centering.cells import iter_cells
from experiments.init_centering.model import _build_init_centering_model
from experiments.init_centering.hparams import _build_init_centering_stages


def _cells():
    """Every cell of the post-auto-clamp ablation grid."""
    return list(iter_cells())


@pytest.mark.parametrize("baseline_form,baseline_mode,tracking_mode", _cells())
def test_all_cells_build(baseline_form, baseline_mode, tracking_mode) -> None:
    """Every cell of the ablation grid builds without error."""
    model = _build_init_centering_model(
        baseline_form=baseline_form,
        baseline_mode=baseline_mode,
        tracking_mode=tracking_mode,
    )
    assert model.baseline_mode == baseline_mode
    assert model.sigma_data is not None
    assert model.sigma_data.tracking_mode == tracking_mode


@pytest.mark.parametrize(
    "form,expected_cls",
    [
        ("zero", ZeroBaseline),
        ("persistence", PersistenceBaseline),
        ("linear", LinearBaseline),
        ("mlp", MLPBaseline),
    ],
)
def test_baseline_form_dispatch(form, expected_cls) -> None:
    """The factory builds the right baseline class for each form."""
    model = _build_init_centering_model(
        baseline_form=form,
        baseline_mode="pinned",
        tracking_mode="per_t",
    )
    assert isinstance(model.baseline, expected_cls)


@pytest.mark.parametrize("form", ["zero", "persistence"])
def test_param_free_forms_autoclamp_learnable_to_pinned(form, caplog) -> None:
    """``zero`` and ``persistence`` clamp ``baseline_mode='learnable'`` to ``'pinned'``."""
    with caplog.at_level(logging.WARNING, logger="experiments.init_centering.model"):
        model = _build_init_centering_model(
            baseline_form=form,
            baseline_mode="learnable",
            tracking_mode="per_t",
        )
    assert model.baseline_mode == "pinned"
    assert "auto-degenerate" in caplog.text, (
        f"expected an auto-degenerate clamp log; got {caplog.text!r}"
    )


# anchor_lambda (stage-2 R_μp strength λ_μp) now lives on the stage-2
# loss object, not the model (ADR-0004 follow-up). Its default still
# depends on baseline_mode; the stage builder owns that logic.


def test_pinned_default_anchor_lambda_is_zero() -> None:
    """Default stage-2 ``λ_μp`` is 0.0 under Pinned mode (R_μp inactive)."""
    stages = _build_init_centering_stages(baseline_mode="pinned")
    assert stages.stage_2.loss.lambda_mu_p == 0.0


def test_learnable_default_anchor_lambda_is_nonzero() -> None:
    """Under Learnable mode the default stage-2 ``λ_μp`` is the doc-recommended 1e-2."""
    stages = _build_init_centering_stages(baseline_mode="learnable")
    assert stages.stage_2.loss.lambda_mu_p == pytest.approx(1e-2)


def test_explicit_anchor_lambda_overrides_default() -> None:
    """A user-supplied ``anchor_lambda`` survives even under Pinned mode."""
    stages = _build_init_centering_stages(baseline_mode="pinned", anchor_lambda=0.7)
    assert stages.stage_2.loss.lambda_mu_p == pytest.approx(0.7)


def test_invalid_baseline_form_raises() -> None:
    """Unknown ``baseline_form`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="baseline_form must be one of"):
        _build_init_centering_model(
            baseline_form="quadratic",  # not a registered form
        )


@pytest.mark.parametrize(
    "baseline_mode,expected_stage2_baseline",
    [
        ("pinned", False),
        ("learnable", True),
    ],
)
def test_stages_factory_baseline_mask_matches_mode(
    baseline_mode,
    expected_stage2_baseline,
) -> None:
    """The stages factory wires stage-2's baseline trainable in lockstep with the mode."""
    stages = _build_init_centering_stages(
        baseline_mode=baseline_mode,
        n_pretrain=10,
        n_stage2=10,
        sigma_pert=1e-3,
    )
    # Stage 1 always trains the baseline.
    assert stages.stage_1.trainable.baseline is True
    # Stage 2 mirrors the mode.
    assert stages.stage_2.trainable.baseline is expected_stage2_baseline
    # The handoff is declared on stage 1 (fires *after* it, before stage 2),
    # so a stage-2-only run carries no handoff artifact.
    assert stages.stage_1.centering_handoff is not None
    assert stages.stage_1.centering_handoff.sigma_pert == pytest.approx(1e-3)
    assert stages.stage_2.centering_handoff is None


def test_stages_factory_early_stop_disabled_by_default() -> None:
    """``early_stop_enabled=False`` leaves both stages without an early-stop spec."""
    stages = _build_init_centering_stages(n_pretrain=10, n_stage2=10)
    assert stages.stage_1.early_stop is None
    assert stages.stage_2.early_stop is None


def test_stages_factory_early_stop_enables_on_stage1() -> None:
    """``early_stop_enabled=True`` populates stage 1 only (Phase-C sweep target)."""
    stages = _build_init_centering_stages(
        n_pretrain=10,
        n_stage2=10,
        early_stop_enabled=True,
        early_stop_window=20,
        early_stop_min_improvement=1e-3,
        early_stop_warmup_steps=30,
    )
    assert stages.stage_1.early_stop is not None
    assert stages.stage_1.early_stop.enabled is True
    assert stages.stage_1.early_stop.window == 20
    assert stages.stage_1.early_stop.min_improvement == pytest.approx(1e-3)
    assert stages.stage_1.early_stop.warmup_steps == 30
    # Stage 2 still unset.
    assert stages.stage_2.early_stop is None
