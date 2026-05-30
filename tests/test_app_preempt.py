"""Tests for preempt-aware trial-resume hand-off in :mod:`ddssm.app`.

Covers ADR-0009 §5 — the app-level retry path. After the gate test in
Phase 1 ruled out ``RetryFailedTrialCallback`` (Outcome B), the retry
trial is enqueued *explicitly* by ``app.py`` on ``PreemptError`` via
``study.add_trial(...)`` — no monkey-patch, no callback machinery.

The Hydra-Optuna sweeper does NOT expose the current Optuna trial to the
task function (Phase 0 finding I1), so the only way to find the current
trial in ``app.py`` is by **param-match**: load the study, filter
``RUNNING`` trials, find the unique one whose ``params`` match the cfg's
sampled hparam values. These tests pin that behaviour in isolation.
"""

from __future__ import annotations

import logging
import os

import optuna
import pytest
from optuna.trial import TrialState
from omegaconf import OmegaConf

from ddssm.app import (
    _enqueue_preempt_retry,
    _find_current_running_trial_by_params,
    _get_resume_from_user_attrs,
    apply_preempt_hooks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _storage_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'study.db'}"


def _make_study(tmp_path, name: str = "s") -> tuple[optuna.Study, str]:
    url = _storage_url(tmp_path)
    study = optuna.create_study(
        study_name=name, storage=url, direction="minimize",
    )
    return study, url


# ---------------------------------------------------------------------------
# _find_current_running_trial_by_params
# ---------------------------------------------------------------------------


def test_find_current_trial_by_param_match(tmp_path) -> None:
    """Param-match against the unique RUNNING trial returns it; otherwise None."""
    study, _ = _make_study(tmp_path)
    # Categorical distribution gives us deterministic, exact-representable
    # param values without needing to monkey around with float-log encoding.
    dist = optuna.distributions.CategoricalDistribution([0.1, 0.5, 0.99])

    # Trial A: COMPLETE. Enqueue a specific param value so we know what to skip
    # past in the match.
    study.enqueue_trial({"lr": 0.1})
    t_complete = study.ask({"lr": dist})
    study.tell(t_complete, 0.5)

    # Trial B: RUNNING with params {lr=0.5}. Enqueue → ask leaves it RUNNING.
    study.enqueue_trial({"lr": 0.5})
    t_running = study.ask({"lr": dist})
    assert t_running.params == {"lr": 0.5}

    # Match on the RUNNING trial's params.
    reloaded = optuna.load_study(study_name=study.study_name, storage=study._storage)
    found = _find_current_running_trial_by_params(reloaded, {"lr": 0.5})
    assert found is not None
    assert found.number == t_running.number
    assert found.state == TrialState.RUNNING

    # No-match returns None.
    assert _find_current_running_trial_by_params(reloaded, {"lr": 0.99}) is None

    # Empty hparams should return None — ambiguous / no-match path.
    assert _find_current_running_trial_by_params(reloaded, {}) is None


def test_find_current_trial_returns_none_on_ambiguous_match(tmp_path, caplog) -> None:
    """Two RUNNING trials with identical params → None + warning logged."""
    study, _ = _make_study(tmp_path)
    dist = optuna.distributions.CategoricalDistribution([0.5])  # single value → deterministic

    # Two RUNNING trials, both forced to the same param value via enqueue.
    for _ in range(2):
        study.enqueue_trial({"lr": 0.5})
        study.ask({"lr": dist})

    reloaded = optuna.load_study(study_name=study.study_name, storage=study._storage)
    with caplog.at_level(logging.WARNING):
        found = _find_current_running_trial_by_params(reloaded, {"lr": 0.5})
    assert found is None
    # Some warning was emitted; don't pin the exact text.
    assert any("trial" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# _get_resume_from_user_attrs
# ---------------------------------------------------------------------------


def test_extract_resume_from_user_attrs_on_retry(tmp_path) -> None:
    """A trial carrying ``user_attrs["resume_from"]`` returns the path; else None."""
    study, _ = _make_study(tmp_path)

    # Trial with resume_from + retried_from set.
    t1 = optuna.trial.create_trial(
        params={"lr": 0.5},
        distributions={
            "lr": optuna.distributions.FloatDistribution(1e-5, 1.0, log=True),
        },
        state=TrialState.WAITING,
        user_attrs={"resume_from": "/tmp/x.pth", "retried_from": 0},
    )
    study.add_trial(t1)

    # Trial WITHOUT user_attrs.
    t2 = optuna.trial.create_trial(
        params={"lr": 0.1},
        distributions={
            "lr": optuna.distributions.FloatDistribution(1e-5, 1.0, log=True),
        },
        value=0.1,
    )
    study.add_trial(t2)

    trials = study.get_trials(deepcopy=False)
    # Find trials by params.
    with_attr = next(t for t in trials if t.params.get("lr") == 0.5)
    without_attr = next(t for t in trials if t.params.get("lr") == 0.1)

    assert _get_resume_from_user_attrs(with_attr) == "/tmp/x.pth"
    assert _get_resume_from_user_attrs(without_attr) is None


# ---------------------------------------------------------------------------
# _enqueue_preempt_retry
# ---------------------------------------------------------------------------


def test_enqueue_retry_on_preempt_error(tmp_path) -> None:
    """``_enqueue_preempt_retry`` adds a WAITING trial with the right metadata."""
    study, _ = _make_study(tmp_path)
    dist = optuna.distributions.CategoricalDistribution([0.1, 0.5, 0.99])

    # Build a "current" running trial with known params via enqueue.
    study.enqueue_trial({"lr": 0.5})
    current_live = study.ask({"lr": dist})
    assert current_live.params == {"lr": 0.5}
    # Reload the FrozenTrial snapshot the helper will receive.
    current = study._storage.get_trial(current_live._trial_id)
    assert current.state == TrialState.RUNNING

    _enqueue_preempt_retry(study, current, resume_from="/tmp/y.pth")

    # Now the study should have a NEW WAITING trial.
    trials = study.get_trials(deepcopy=False)
    waiting = [t for t in trials if t.state == TrialState.WAITING]
    assert len(waiting) == 1
    retry = waiting[0]

    # (a) Same params.
    assert retry.params == current.params
    # (b) resume_from in user_attrs.
    assert retry.user_attrs.get("resume_from") == "/tmp/y.pth"
    # (c) retried_from points at the original.
    assert retry.user_attrs.get("retried_from") == current.number


# ---------------------------------------------------------------------------
# apply_preempt_hooks — the public top-level entry point
# ---------------------------------------------------------------------------


def test_preempt_logic_skipped_when_env_unset(tmp_path, monkeypatch) -> None:
    """With ``DDSSM_PREEMPTIVE`` unset, ``apply_preempt_hooks`` is a no-op."""
    monkeypatch.delenv("DDSSM_PREEMPTIVE", raising=False)

    # Cfg with a bogus storage/study — apply_preempt_hooks must NOT touch it.
    cfg = OmegaConf.create(
        {
            "experiment": {"training": {"resume_from": None}},
            "hydra": {
                "sweeper": {
                    "study_name": "nonexistent",
                    "storage": "sqlite:////tmp/does-not-exist-xyz.db",
                },
            },
        },
    )

    trial, study = apply_preempt_hooks(cfg)
    assert trial is None
    assert study is None
    # cfg.experiment.training.resume_from MUST remain unset (no injection).
    assert cfg.experiment.training.resume_from is None


def test_skip_resume_when_no_param_match(tmp_path, monkeypatch, caplog) -> None:
    """No matching RUNNING trial → warning + no resume_from injected."""
    monkeypatch.setenv("DDSSM_PREEMPTIVE", "1")
    _, url = _make_study(tmp_path)
    # The study exists but has no RUNNING trials.

    cfg = OmegaConf.create(
        {
            "experiment": {
                "training": {"resume_from": None, "n_pretrain": 123},
            },
            "hydra": {
                "sweeper": {"study_name": "s", "storage": url},
            },
        },
    )

    with caplog.at_level(logging.WARNING):
        trial, study = apply_preempt_hooks(cfg)

    # The study should have been loaded but no trial matched.
    assert trial is None
    # The caller still gets the study handle back for downstream use, OR None
    # if the helper bails entirely; either is acceptable. We only pin that no
    # resume_from was injected.
    assert cfg.experiment.training.resume_from is None
    # A warning was logged.
    assert any(
        "match" in rec.message.lower() or "trial" in rec.message.lower()
        for rec in caplog.records
    )
