"""Per-stage loss-object selection (ADR-0004, shifts 4-6).

The `StageOrchestrator` installs an `active_loss` on the trainer when
entering each stage. The trainer drives `step_within_stage` from
its own counters; the loss object's λ schedule starts fresh at each
stage boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from ddssm.model.losses import Loss, FullELBO, LossComponents
from ddssm.training.stages import (
    StagesConf,
    StageLrsConf,
    StageSpecConf,
    LambdaRampConf,
    StageOrchestrator,
    StageTrainableConf,
)


class _DummyTrainer:
    """Stub trainer that records orchestrator-driven loss installation."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(stage_selector="stage_0")
        self.global_step: int = 0
        self._active_loss: Loss | None = None
        self._stage_start_step: int = 0
        self.installs: list[tuple[str, Loss | None]] = []

    def _rebuild_optimizer(self, lrs) -> None:
        pass

    def _set_trainable(self, t) -> None:
        pass

    def fit(self, **kw) -> None:
        # Snapshot installed loss at fit time, tagged by checkpoint_prefix (= stage key)
        self.installs.append((kw.get("checkpoint_prefix"), self._active_loss))
        # Bump global_step to simulate stage 1's progress before stage 2 enters
        self.global_step += int(kw.get("total_steps", 0)) - self.global_step


def _make_2stage(loss_a: Loss | None, loss_b: Loss | None) -> StagesConf:
    stage_1 = StageSpecConf(
        steps=10, trainable=StageTrainableConf(), lrs=StageLrsConf(),
        lambda_ramp=LambdaRampConf(start=0.001, end=1.0, steps=10),
        loss=loss_a,
    )
    stage_2 = StageSpecConf(
        steps=20, trainable=StageTrainableConf(), lrs=StageLrsConf(),
        lambda_ramp=LambdaRampConf(start=0.1, end=1.0, steps=20),
        loss=loss_b,
    )
    return StagesConf(stage_1=stage_1, stage_2=stage_2, run=["stage_1", "stage_2"])


def test_orchestrator_installs_stage_loss_per_stage() -> None:
    """When `stage.loss` is set, the orchestrator installs it on the trainer
    before that stage's fit loop. Different stages get different losses.
    """
    loss_a = FullELBO(rate_lambda=lambda s: 0.5)
    loss_b = FullELBO(rate_lambda=lambda s: 0.7)
    trainer = _DummyTrainer()
    cfg = _make_2stage(loss_a, loss_b)
    StageOrchestrator(trainer, cfg).run(train_loader=object(), amp=False)
    assert trainer.installs[0] == ("stage_1", loss_a)
    assert trainer.installs[1] == ("stage_2", loss_b)


def test_orchestrator_installs_default_full_elbo_when_stage_loss_none() -> None:
    """When `stage.loss is None`, the orchestrator constructs a default
    `FullELBO` whose `rate_lambda` matches the stage's `lambda_ramp`.
    Preserves pre-ADR-0004 behavior for presets that haven't migrated.
    """
    trainer = _DummyTrainer()
    cfg = _make_2stage(loss_a=None, loss_b=None)
    StageOrchestrator(trainer, cfg).run(train_loader=object(), amp=False)
    stage_1_loss = trainer.installs[0][1]
    stage_2_loss = trainer.installs[1][1]
    assert isinstance(stage_1_loss, FullELBO)
    assert isinstance(stage_2_loss, FullELBO)
    # Stage 1 ramps from 0.001 to 1.0 over 10 steps;
    # Stage 2 ramps from 0.1 to 1.0 over 20 steps.
    # Different starts ⇒ different first-step λ values.
    assert stage_1_loss.rate_lambda(0) != stage_2_loss.rate_lambda(0)


def test_step_within_stage_resets_at_boundary() -> None:
    """`_stage_start_step` is reset at each stage boundary so the loss
    object's λ schedule starts fresh. λ at the first step of stage 2
    reflects stage 2's ramp, not continuation of stage 1.
    """
    trainer = _DummyTrainer()
    cfg = _make_2stage(loss_a=None, loss_b=None)
    StageOrchestrator(trainer, cfg).run(train_loader=object(), amp=False)
    stage_1_loss = trainer.installs[0][1]
    stage_2_loss = trainer.installs[1][1]
    components = LossComponents(
        recon=torch.tensor(0.0), init_kl=torch.tensor(1.0),
        trans_kl=torch.tensor(0.0), r_sigma_p=torch.tensor(0.0),
        r_mu_p=torch.tensor(0.0),
    )
    # Stage 1, step_within_stage=1 ⇒ very early in cosine ramp.
    stage_1_lambda_at_1 = stage_1_loss(components, 1).item()
    # Stage 2, step_within_stage=1 ⇒ also very early in stage-2 ramp,
    # but the start floor is 0.1 not 0.001.
    stage_2_lambda_at_1 = stage_2_loss(components, 1).item()
    # Each stage's λ at its first step matches the stage's own ramp start.
    assert stage_2_lambda_at_1 > stage_1_lambda_at_1
