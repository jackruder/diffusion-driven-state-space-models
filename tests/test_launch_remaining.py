"""Tests for ``ddssm.launch_remaining`` — the preemptive-sweep budget CLI.

Covers ``compute_remaining(storage, study_name, target, cleanup_older_than)``
and the argparse-driven ``main`` entrypoint baked into the sbatch preamble by
ADR-0009. Each test spins up a tmp SQLite-backed Optuna study, drives it via
the public Optuna API where possible, and pokes ``datetime_start`` via raw
SQLite for the cleanup-timeout cases (no ``freezegun`` dependency in this
project — see ``pyproject.toml``).
"""

from __future__ import annotations

import os
import sqlite3
import datetime

import optuna
from optuna.trial import TrialState

from ddssm.launch_remaining import main, compute_remaining


def _storage_url(tmp_path) -> tuple[str, str]:
    """Return ``(sqlite_url, db_path)`` rooted under ``tmp_path``."""
    db = os.path.join(tmp_path, "study.db")
    return f"sqlite:///{db}", db


def _age_trial(db_path: str, trial_id: int, seconds_ago: float) -> None:
    """Backdate a trial's ``datetime_start`` directly in the SQLite file.

    Optuna's storage API does not expose a writer for ``datetime_start``;
    raw SQL is the project-stable lever and matches what the cleanup path
    has to handle in production (clocks drifting across SLURM nodes / NFS).
    """
    aged = (
        datetime.datetime.now() - datetime.timedelta(seconds=seconds_ago)
    ).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE trials SET datetime_start = ? WHERE trial_id = ?",
            (aged, trial_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# compute_remaining
# ---------------------------------------------------------------------------


def test_returns_target_when_study_absent(tmp_path) -> None:
    url, db = _storage_url(tmp_path)
    # DB file should not even exist yet.
    assert not os.path.exists(db)
    assert compute_remaining(url, "missing_study", target=12) == 12
    # And compute_remaining MUST NOT side-effect a study into existence.
    if os.path.exists(db):
        # Even if SQLite created the file, no study row should be there.
        studies = optuna.get_all_study_summaries(storage=url)
        assert all(s.study_name != "missing_study" for s in studies)


def test_subtracts_complete_and_pruned(tmp_path) -> None:
    url, _ = _storage_url(tmp_path)
    study = optuna.create_study(study_name="s", storage=url, direction="minimize")
    # 3 COMPLETE
    for v in (0.1, 0.2, 0.3):
        t = study.ask()
        study.tell(t, v)
    # 2 PRUNED
    for _ in range(2):
        t = study.ask()
        study.tell(t, state=TrialState.PRUNED)
    # 1 FAILED (must NOT subtract from target)
    t = study.ask()
    study.tell(t, state=TrialState.FAIL)

    assert compute_remaining(url, "s", target=10) == 5


def test_clamps_at_zero(tmp_path) -> None:
    url, _ = _storage_url(tmp_path)
    study = optuna.create_study(study_name="s", storage=url, direction="minimize")
    for v in range(7):
        t = study.ask()
        study.tell(t, float(v))
    assert compute_remaining(url, "s", target=5) == 0


def test_cleanup_is_noop_without_heartbeats(tmp_path) -> None:
    """An aged RUNNING trial must NOT be reaped without heartbeat config.

    Reaping by ``datetime_start`` (the prior implementation) falsely killed
    healthy workers that were still running their trial. The corrected
    implementation delegates to ``optuna.storages.fail_stale_trials``,
    which uses heartbeat metadata; without a heartbeat-configured storage
    it is a safe no-op. Orphan RUNNING rows persist in that mode but do
    not affect the budget count (only COMPLETE + PRUNED count).
    """
    url, db = _storage_url(tmp_path)
    study = optuna.create_study(study_name="s", storage=url, direction="minimize")
    t = study.ask()
    trial_id = t._trial_id
    _age_trial(db, trial_id, seconds_ago=300)  # 5 minutes ago

    compute_remaining(url, "s", target=10, cleanup_older_than=60)

    reloaded = optuna.load_study(study_name="s", storage=url)
    ft = reloaded._storage.get_trial(trial_id)
    assert ft.state == TrialState.RUNNING


def test_cleanup_does_not_touch_fresh_running(tmp_path) -> None:
    """Sanity check that fresh RUNNING trials are also safe (same no-op path)."""
    url, db = _storage_url(tmp_path)
    study = optuna.create_study(study_name="s", storage=url, direction="minimize")
    t = study.ask()
    trial_id = t._trial_id
    _age_trial(db, trial_id, seconds_ago=5)

    compute_remaining(url, "s", target=10, cleanup_older_than=60)

    reloaded = optuna.load_study(study_name="s", storage=url)
    ft = reloaded._storage.get_trial(trial_id)
    assert ft.state == TrialState.RUNNING


def test_orphan_running_trials_do_not_affect_budget(tmp_path) -> None:
    """RUNNING orphans must NOT subtract from the remaining target.

    Without heartbeats the reaper is a no-op, so orphan RUNNING rows
    persist. The budget invariant — only COMPLETE + PRUNED count — must
    hold regardless of how many orphans accumulate.
    """
    url, db = _storage_url(tmp_path)
    study = optuna.create_study(study_name="s", storage=url, direction="minimize")
    # 3 COMPLETE; should leave target - 3 = 7 remaining.
    for v in (0.1, 0.2, 0.3):
        t = study.ask()
        study.tell(t, v)
    # 4 RUNNING orphans aged out: must not reduce remaining.
    for _ in range(4):
        t = study.ask()
        _age_trial(db, t._trial_id, seconds_ago=900)

    assert compute_remaining(url, "s", target=10, cleanup_older_than=60) == 7


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_main_prints_remaining_count_to_stdout(tmp_path, capsys) -> None:
    url, _ = _storage_url(tmp_path)
    study = optuna.create_study(study_name="s", storage=url, direction="minimize")
    for v in (0.1, 0.2):
        t = study.ask()
        study.tell(t, v)

    rc = main(["--storage", url, "--study", "s", "--target", "10"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    # Must parse as an int — the sbatch preamble does ``$(python -m ...)`` math on it.
    assert int(out) == 8
