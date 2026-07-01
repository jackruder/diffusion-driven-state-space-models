"""End-to-end-ish test for the preempt-retry-resume chain (ADR-0009).

This test composes the three independent surfaces from Phases 2c / 5 / 6:
- the trainer raises ``PreemptError(resume_from=<ckpt>)`` (Phase 2c),
- ``ddssm.app._enqueue_preempt_retry`` calls ``study.add_trial`` carrying
  the failed trial's params + ``user_attrs["resume_from"]`` (Phase 6),
- the next ``study.ask()`` returns the retry with the inherited attrs,
  ``_get_resume_from_user_attrs`` reads them back, and a fresh trainer
  resumes from the saved checkpoint and walks ``global_step`` forward
  (Phase 2c resume tests).

It exercises the entire in-process retry chain without spinning up a
SLURM job or a ``python -m ddssm.app`` subprocess — the orchestrator
sbatch path is unit-tested in ``tests/test_launch.py`` and there's no
value in re-asserting that text here. What we DO want to lock down is
that the three surfaces above wire together correctly.
"""

from __future__ import annotations

from pathlib import Path

import torch
import optuna
from optuna.trial import TrialState

from ddssm.app import (
    _enqueue_preempt_retry,
    _get_resume_from_user_attrs,
    _find_current_running_trial_by_params,
)


def _tiny_storage(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'e2e.db'}"


def _new_trial_with_params(
    study: optuna.Study, params: dict[str, float]
) -> optuna.Trial:
    trial = study.ask()
    for name, value in params.items():
        trial.suggest_float(name, value, value)
    return trial


def test_preempt_retry_chain_carries_resume_from_across_trials(tmp_path) -> None:
    """A preempted trial's ``resume_from`` reaches the retry via ``user_attrs``."""
    study = optuna.create_study(
        study_name="chain",
        storage=_tiny_storage(tmp_path),
        direction="minimize",
    )
    params = {"x": 0.42, "y": -1.5}
    original = _new_trial_with_params(study, params)
    original_number = original.number
    ckpt = str(tmp_path / "ckpt_latest.pth")
    Path(ckpt).write_bytes(b"")

    # Simulate the trainer's preempt path: ddssm.app catches PreemptError
    # and enqueues the retry; the original is marked FAILED separately by
    # the sweeper. Here we drive the same call by hand.
    _enqueue_preempt_retry(study, original, resume_from=ckpt)
    study.tell(original, state=TrialState.FAIL)

    waiting = [
        t for t in study.get_trials(deepcopy=False) if t.state == TrialState.WAITING
    ]
    assert len(waiting) == 1, f"expected exactly one WAITING retry; got {len(waiting)}"
    retry = waiting[0]
    assert retry.params == original.params, "retry inherits hparams"
    assert _get_resume_from_user_attrs(retry) == ckpt, "retry carries the ckpt path"
    assert retry.user_attrs.get("retried_from") == original_number, (
        "retry tracks lineage"
    )


def test_preempt_retry_chain_walks_forward_across_two_preempts(tmp_path) -> None:
    """A retry that itself preempts produces a second retry whose resume_from steps forward."""
    study = optuna.create_study(
        study_name="walk",
        storage=_tiny_storage(tmp_path),
        direction="minimize",
    )
    params = {"x": 0.1}
    trial_a = _new_trial_with_params(study, params)
    ckpt_a = str(tmp_path / "a.pth")
    Path(ckpt_a).write_bytes(b"")
    _enqueue_preempt_retry(study, trial_a, resume_from=ckpt_a)
    study.tell(trial_a, state=TrialState.FAIL)

    # The retry is picked up by the next ask(); simulate it preempting too.
    trial_b = study.ask()
    assert trial_b.params == params, "retry params match"
    assert _get_resume_from_user_attrs(_freeze(trial_b, study)) == ckpt_a, (
        "the retry that was just dequeued knows where to resume from"
    )

    ckpt_b = str(tmp_path / "b.pth")
    Path(ckpt_b).write_bytes(b"")
    _enqueue_preempt_retry(study, _freeze(trial_b, study), resume_from=ckpt_b)
    study.tell(trial_b, state=TrialState.FAIL)

    waiting = [
        t for t in study.get_trials(deepcopy=False) if t.state == TrialState.WAITING
    ]
    assert len(waiting) == 1, (
        "exactly one new WAITING retry chained off the second preempt"
    )
    assert _get_resume_from_user_attrs(waiting[0]) == ckpt_b, (
        "the chain's resume_from advances to the latest preempted trial's ckpt"
    )


def test_preempt_chain_trainer_resume_continues_global_step(tmp_path) -> None:
    """The end-to-end shape: trainer raises → retry enqueued → trainer resumes from ckpt.

    This stitches the trainer's PreemptError + the app's retry enqueue into a
    chain that mirrors what ddssm.app would do across two invocations: the
    first writes a ckpt and raises; the second reads ``resume_from`` from the
    retry's ``user_attrs`` and would pass it to ``trainer.fit(resume_from=...)``.

    We don't drive a real ``DDSSMTrainer`` here — that path is exhaustively
    tested in ``tests/test_train_preempt.py``. We assert the resume_from
    *value* that would land in the fresh trainer's config.
    """
    storage = _tiny_storage(tmp_path)
    study = optuna.create_study(
        study_name="resume_chain",
        storage=storage,
        direction="minimize",
    )

    # Invocation 1: a trial starts, the trainer's signal handler fires
    # mid-step and the trainer raises PreemptError(resume_from=<step 10>).
    params = {"lr": 5e-4}
    t1 = _new_trial_with_params(study, params)
    ckpt_step10 = tmp_path / "run_dir_1" / "checkpoints" / "ckpt_latest.pth"
    ckpt_step10.parent.mkdir(parents=True)
    torch.save({"global_step": 10, "model_state": {}}, ckpt_step10)
    _enqueue_preempt_retry(study, t1, resume_from=str(ckpt_step10))
    study.tell(t1, state=TrialState.FAIL)

    # Invocation 2: a fresh app.py task function loads the study and
    # picks up the next trial.
    t2 = study.ask()
    t2_frozen = _freeze(t2, study)
    resume_from = _get_resume_from_user_attrs(t2_frozen)
    assert resume_from == str(ckpt_step10), "fresh trial's config gets the step-10 ckpt"

    # The trainer (if we ran it) would resume from that path and continue.
    # Verify the ckpt is loadable and carries the expected global_step.
    payload = torch.load(resume_from, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 10


def test_find_current_running_trial_picks_unique_match(tmp_path) -> None:
    """The orchestrator's app-side lookup finds the right trial among siblings."""
    study = optuna.create_study(
        study_name="lookup",
        storage=_tiny_storage(tmp_path),
        direction="minimize",
    )
    a = _new_trial_with_params(study, {"x": 0.1})
    b = _new_trial_with_params(study, {"x": 0.5})

    found = _find_current_running_trial_by_params(study, {"x": 0.5})
    assert found is not None
    assert found.number == b.number

    # Ambiguous (no match) is a None, not a raise.
    assert _find_current_running_trial_by_params(study, {"x": 99.0}) is None


def test_apply_preempt_hooks_is_noop_without_env_var(tmp_path, monkeypatch) -> None:
    """The whole preempt path is gated on ``DDSSM_PREEMPTIVE=1``."""
    from omegaconf import OmegaConf

    from ddssm.app import apply_preempt_hooks

    monkeypatch.delenv("DDSSM_PREEMPTIVE", raising=False)
    cfg = OmegaConf.create({"experiment": {"training": {}}})
    trial, study = apply_preempt_hooks(cfg)
    assert trial is None and study is None
    assert "resume_from" not in cfg.experiment.training


# --- helpers ----------------------------------------------------------------


def _freeze(trial: optuna.Trial, study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Return the FrozenTrial view of an in-flight Trial via the storage."""
    return study._storage.get_trial(trial._trial_id)
