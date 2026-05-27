"""Phase-B integration tests: every cell of the 18-cell grid round-trips.

For each ``(baseline_form, baseline_mode, tracking_mode)`` cell from
``init-experiment.org`` § Composition with the ablation grid we:

1. Build the model via :func:`_build_init_centering_model`.
2. Run the centering handoff (the stage-1 → stage-2 boundary).
3. Apply the stage-2 trainable mask from
   :func:`_build_init_centering_stages`.
4. Assert that the resulting ``requires_grad`` flags on
   ``model.baseline.parameters()`` agree with the cell's
   ``baseline_mode`` (Pinned ⇒ frozen, Learnable ⇒ trainable).
5. Assert ``model.sigma_data`` has the requested ``tracking_mode``
   and that ``reset_schedule()``'s effect matches the mode (``fixed``
   ⇒ ``frozen=True``; otherwise ``frozen=False``).

The test does not run an optimiser step — it verifies that the
parametric factories produce *self-consistent* models, which is the
core deliverable of Phase B.  Math-level convergence per cell is
covered by ``test_integration/test_baseline_form_learns.py`` and
``test_tracking_modes.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from ddssm.train import DDSSMTrainer
from ddssm.stages import StageLrsConf
from ddssm.centering.handoff import CenteringHandoffConf, perform_centering_handoff
from experiments.init_centering.model import _build_init_centering_model
from experiments.init_centering.hparams import _build_init_centering_stages

_BASELINE_FORMS = ("zero", "identity", "linear", "mlp")
_TRACKING_MODES = ("fixed", "global_ema", "per_t")


def _cells():
    cells = []
    for form in _BASELINE_FORMS:
        modes = ("pinned",) if form in ("zero", "identity") else ("pinned", "learnable")
        for mode in modes:
            for tm in _TRACKING_MODES:
                cells.append((form, mode, tm))
    return cells


def _hparams() -> SimpleNamespace:
    return SimpleNamespace(
        S=1, batch_size=4, grad_accum_steps=1, ema_decay=0.999, weight_decay=1e-2,
        lambda_schedule="cosine", lambda_start=0.001, lambda_end=1.0,
        lambda_warmup_steps=10, enc_lr=5e-4, dec_lr=5e-4, zinit_lr=5e-4, trans_lr=5e-4,
        lambda_sigma_p=1e-2,
    )


@pytest.mark.parametrize("baseline_form,baseline_mode,tracking_mode", _cells())
def test_cell_handoff_produces_consistent_baseline_freeze(
    baseline_form, baseline_mode, tracking_mode, tmp_path,
) -> None:
    """Per cell, post-handoff baseline ``requires_grad`` matches ``baseline_mode``."""
    stages = _build_init_centering_stages(
        baseline_mode=baseline_mode, n_pretrain=2, n_stage2=2, sigma_pert=1e-3,
    )
    model = _build_init_centering_model(
        baseline_form=baseline_form,
        baseline_mode=baseline_mode,
        tracking_mode=tracking_mode,
        hyperparams=_hparams(),
        stages=stages,
    )

    # The factory's auto-degenerate clamp may have rewritten the mode.
    actual_mode = model.baseline_mode

    # Trainer is needed by the handoff (rebuilds the optimiser).
    trainer = DDSSMTrainer(
        model=model, device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"), quiet=True,
    )
    perform_centering_handoff(
        trainer, CenteringHandoffConf(sigma_pert=1e-3), new_lrs=StageLrsConf(),
    )

    # The orchestrator would then apply the stage-2 trainable mask.  When
    # the user-requested mode is "learnable" but the factory clamped it to
    # "pinned" (zero / identity forms), the stages factory still emits the
    # *user's* mode and the declarative mask sets ``baseline=True``.  In
    # that case the handoff's imperative freeze (run above) takes priority
    # and keeps baseline params frozen, demonstrating the "belt-and-
    # suspenders" design noted in init-experiment-implementation.org.
    trainer._set_trainable(stages.stage_2.trainable)

    baseline_params = list(model.baseline.parameters())
    if not baseline_params:
        # Pure zero/identity have no learnable μ_p params; nothing to check.
        return

    if actual_mode == "pinned":
        assert all(not p.requires_grad for p in baseline_params), (
            f"cell ({baseline_form}, {baseline_mode}, {tracking_mode}): "
            "baseline params should be frozen post-handoff under Pinned mode"
        )
    else:
        assert all(p.requires_grad for p in baseline_params), (
            f"cell ({baseline_form}, {baseline_mode}, {tracking_mode}): "
            "baseline params should be trainable under Learnable mode"
        )


@pytest.mark.parametrize("tracking_mode", _TRACKING_MODES)
def test_cell_tracking_mode_freeze_semantics(tracking_mode, tmp_path) -> None:
    """``reset_schedule()`` freezes only the ``fixed`` mode."""
    stages = _build_init_centering_stages(
        baseline_mode="pinned", n_pretrain=2, n_stage2=2, sigma_pert=1e-3,
    )
    model = _build_init_centering_model(
        baseline_form="mlp",
        baseline_mode="pinned",
        tracking_mode=tracking_mode,
        hyperparams=_hparams(),
        stages=stages,
    )
    assert model.sigma_data.tracking_mode == tracking_mode

    trainer = DDSSMTrainer(
        model=model, device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"), quiet=True,
    )
    perform_centering_handoff(
        trainer, CenteringHandoffConf(sigma_pert=1e-3), new_lrs=StageLrsConf(),
    )

    if tracking_mode == "fixed":
        assert model.sigma_data.frozen is True
    else:
        assert model.sigma_data.frozen is False
