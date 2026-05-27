"""Phase-B factory tests for the parametric init-centering model builder.

Verifies that :func:`_build_init_centering_model` honours the ablation
grid axes from ``init-experiment.org`` § Composition with the ablation
grid:

* the three cell axes round-trip into the resulting :class:`DDSSM_base`,
* the auto-degenerate clamp turns ``baseline_mode="learnable"`` into
  ``"pinned"`` for parameter-free baseline forms (zero / identity),
* the default ``anchor_lambda`` depends on ``baseline_mode``.

All cells must build without error and the assembled stages factory
must produce a baseline-trainable mask consistent with the chosen
``baseline_mode``.
"""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import pytest

from ddssm.centering.baselines import (
    MLPBaseline,
    ZeroBaseline,
    LinearBaseline,
    IdentityBaseline,
)
from experiments.init_centering.cells import iter_cells
from experiments.init_centering.model import _build_init_centering_model
from experiments.init_centering.hparams import _build_init_centering_stages


def _cells():
    """Every cell of the post-auto-clamp ablation grid."""
    return list(iter_cells())


def _minimal_hparams() -> SimpleNamespace:
    return SimpleNamespace(
        S=1, batch_size=4, grad_accum_steps=1, ema_decay=0.999, weight_decay=1e-2,
        lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=10, enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
        lambda_sigma_p=1e-2,
    )


@pytest.mark.parametrize("baseline_form,baseline_mode,tracking_mode", _cells())
def test_all_cells_build(baseline_form, baseline_mode, tracking_mode) -> None:
    """Every cell of the ablation grid builds without error."""
    model = _build_init_centering_model(
        baseline_form=baseline_form,
        baseline_mode=baseline_mode,
        tracking_mode=tracking_mode,
        hyperparams=_minimal_hparams(),
        stages=None,
    )
    assert model.baseline_mode == baseline_mode
    assert model.sigma_data is not None
    assert model.sigma_data.tracking_mode == tracking_mode


@pytest.mark.parametrize(
    "form,expected_cls",
    [
        ("zero", ZeroBaseline),
        ("identity", IdentityBaseline),
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
        hyperparams=_minimal_hparams(),
        stages=None,
    )
    assert isinstance(model.baseline, expected_cls)


@pytest.mark.parametrize("form", ["zero", "identity"])
def test_param_free_forms_autoclamp_learnable_to_pinned(form) -> None:
    """``zero`` and ``identity`` clamp ``baseline_mode='learnable'`` to ``'pinned'``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _build_init_centering_model(
            baseline_form=form,
            baseline_mode="learnable",
            tracking_mode="per_t",
            hyperparams=_minimal_hparams(),
            stages=None,
        )
    assert model.baseline_mode == "pinned"
    assert any("auto-degenerate" in str(w.message) for w in caught), (
        f"expected an auto-degenerate warning; got {[str(w.message) for w in caught]}"
    )


def test_pinned_default_anchor_lambda_is_zero() -> None:
    """Default ``anchor_lambda`` is 0.0 under Pinned mode (R_μp inactive)."""
    model = _build_init_centering_model(
        baseline_form="mlp",
        baseline_mode="pinned",
        tracking_mode="per_t",
        hyperparams=_minimal_hparams(),
        stages=None,
    )
    assert model.anchor_lambda == 0.0


def test_learnable_default_anchor_lambda_is_nonzero() -> None:
    """Under Learnable mode the default anchor_lambda is the doc-recommended 1e-2."""
    model = _build_init_centering_model(
        baseline_form="mlp",
        baseline_mode="learnable",
        tracking_mode="per_t",
        hyperparams=_minimal_hparams(),
        stages=None,
    )
    assert model.anchor_lambda == pytest.approx(1e-2)


def test_explicit_anchor_lambda_overrides_default() -> None:
    """A user-supplied ``anchor_lambda`` survives even under Pinned mode."""
    model = _build_init_centering_model(
        baseline_form="mlp",
        baseline_mode="pinned",
        tracking_mode="per_t",
        anchor_lambda=0.7,
        hyperparams=_minimal_hparams(),
        stages=None,
    )
    assert model.anchor_lambda == pytest.approx(0.7)


def test_invalid_baseline_form_raises() -> None:
    """Unknown ``baseline_form`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="baseline_form must be one of"):
        _build_init_centering_model(
            baseline_form="quadratic",   # not a registered form
            hyperparams=_minimal_hparams(),
            stages=None,
        )


@pytest.mark.parametrize("baseline_mode,expected_stage2_baseline", [
    ("pinned", False),
    ("learnable", True),
])
def test_stages_factory_baseline_mask_matches_mode(
    baseline_mode, expected_stage2_baseline,
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
    # The handoff is always set on stage 2.
    assert stages.stage_2.centering_handoff is not None
    assert stages.stage_2.centering_handoff.sigma_pert == pytest.approx(1e-3)


def test_stages_factory_early_stop_disabled_by_default() -> None:
    """``early_stop_enabled=False`` leaves both stages without an early-stop spec."""
    stages = _build_init_centering_stages(n_pretrain=10, n_stage2=10)
    assert stages.stage_1.early_stop is None
    assert stages.stage_2.early_stop is None


def test_stages_factory_early_stop_enables_on_stage1() -> None:
    """``early_stop_enabled=True`` populates stage 1 only (Phase-C sweep target)."""
    stages = _build_init_centering_stages(
        n_pretrain=10, n_stage2=10,
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
