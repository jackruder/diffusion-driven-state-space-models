"""Integration tests for :class:`ddssm.training.stages.StageOrchestrator`'s handoff hook.

Verify that when a stage has ``centering_handoff`` set, the orchestrator
calls :func:`perform_centering_handoff` *after* that stage's fit loop and
only when a later stage will run (so stage 1's handoff fires between the
two stages' fits), flips ``stage_selector`` correctly, and does not
double-rebuild the optimizer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from ddssm.training.stages import (
    StagesConf,
    StageLrsConf,
    StageSpecConf,
    StageOrchestrator,
    StageTrainableConf,
)
from ddssm.model.centering.handoff import CenteringHandoffConf


class _DummyTrainer:
    """Stub trainer that records the orchestrator's calls in order."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(stage_selector="stage_0")
        self.calls: list[tuple] = []
        self.global_step: int = 0

    def _rebuild_optimizer(self, lrs) -> None:
        self.calls.append(("rebuild", lrs))

    def _set_trainable(self, t) -> None:
        self.calls.append(("trainable", t))

    def fit(self, **kw) -> None:
        self.calls.append(("fit", kw.get("total_steps"), kw.get("checkpoint_prefix")))


def _make_config(with_handoff: bool) -> StagesConf:
    handoff = CenteringHandoffConf(sigma_pert=0.0) if with_handoff else None
    # The handoff is declared on stage 1: it fires *after* stage 1's loop and
    # before stage 2's, rebuilding the optimizer with stage 2's LRs.
    stage_1 = StageSpecConf(
        steps=10,
        trainable=StageTrainableConf(),
        lrs=StageLrsConf(enc_lr=1e-3),
        log_every=5,
        val_every=10,
        checkpoint_every=10,
        centering_handoff=handoff,
    )
    stage_2 = StageSpecConf(
        steps=20,
        trainable=StageTrainableConf(),
        lrs=StageLrsConf(enc_lr=2e-3),
        log_every=5,
        val_every=10,
        checkpoint_every=10,
    )
    return StagesConf(stage_1=stage_1, stage_2=stage_2, run=["stage_1", "stage_2"])


def test_orchestrator_flips_stage_selector() -> None:
    """The orchestrator sets ``stage_selector`` to the stage key before fit."""
    trainer = _DummyTrainer()
    cfg = _make_config(with_handoff=False)
    orch = StageOrchestrator(trainer, cfg)
    selectors_at_fit = []

    real_fit = trainer.fit

    def _spy_fit(**kw) -> None:
        selectors_at_fit.append(trainer.model.stage_selector)
        real_fit(**kw)

    trainer.fit = _spy_fit  # type: ignore[assignment]
    orch.run(train_loader=object(), amp=False)
    assert selectors_at_fit == ["stage_1", "stage_2"]


def test_orchestrator_calls_handoff_after_stage1_when_configured() -> None:
    """Stage 1's ``centering_handoff`` fires once, with the *next* stage's LRs."""
    trainer = _DummyTrainer()
    cfg = _make_config(with_handoff=True)
    orch = StageOrchestrator(trainer, cfg)

    handoff_called: list[tuple] = []

    def _spy_handoff(trainer_arg, spec, *, new_lrs) -> None:  # noqa: ANN001
        handoff_called.append((spec.sigma_pert, new_lrs.enc_lr))

    with patch(
        "ddssm.training.stages.perform_centering_handoff", side_effect=_spy_handoff
    ):
        orch.run(train_loader=object(), amp=False)

    # Called exactly once, rebuilding for stage_2 (its σ_pert + LRs).
    assert handoff_called == [(0.0, 2e-3)]


def test_orchestrator_skips_rebuild_when_handoff_set() -> None:
    """When centering_handoff fires, the orchestrator skips ``_rebuild_optimizer``."""
    trainer = _DummyTrainer()
    cfg = _make_config(with_handoff=True)
    orch = StageOrchestrator(trainer, cfg)

    with patch("ddssm.training.stages.perform_centering_handoff"):
        orch.run(train_loader=object(), amp=False)

    # Stage 1 rebuilds once at its start.  Its post-loop handoff rebuilds for
    # stage 2 (counted as the handoff's own, not orchestrator-driven), so the
    # orchestrator skips stage 2's rebuild → one orchestrator-driven rebuild.
    rebuild_calls = [c for c in trainer.calls if c[0] == "rebuild"]
    assert len(rebuild_calls) == 1


def test_orchestrator_rebuilds_when_no_handoff() -> None:
    """Without a handoff, each stage triggers one optimizer rebuild."""
    trainer = _DummyTrainer()
    cfg = _make_config(with_handoff=False)
    orch = StageOrchestrator(trainer, cfg)
    orch.run(train_loader=object(), amp=False)
    rebuild_calls = [c for c in trainer.calls if c[0] == "rebuild"]
    assert len(rebuild_calls) == 2


def test_orchestrator_stage2_only_fires_no_handoff() -> None:
    """``run=["stage_2"]`` (stage 1 disabled) leaves no handoff artifact.

    The handoff is owned by stage 1 and fires only after it runs, so a
    stage-2-only run never snapshots/pins μ_p or perturbs the encoder. Stage 2
    just rebuilds its own optimizer and trains.
    """
    trainer = _DummyTrainer()
    cfg = _make_config(with_handoff=True)
    cfg.run = ["stage_2"]
    orch = StageOrchestrator(trainer, cfg)

    with patch("ddssm.training.stages.perform_centering_handoff") as mock_handoff:
        orch.run(train_loader=object(), amp=False)

    assert mock_handoff.call_count == 0
    fits = [c for c in trainer.calls if c[0] == "fit"]
    assert len(fits) == 1 and fits[0][2] == "stage_2"
    rebuild_calls = [c for c in trainer.calls if c[0] == "rebuild"]
    assert len(rebuild_calls) == 1
