r"""Compute how many trials a preemptive sweep still owes its target.

Invoked from the bash preamble that ``ddssm.cluster.sbatch`` emits under
``PointLaunch.preemptive=True`` (ADR-0009):

    N_REMAINING=$(python -m ddssm.launch_remaining \\
        --storage <storage> --study <name> --target <target>)

The CLI prints a single integer to stdout (the count of COMPLETE+PRUNED
trials subtracted from ``--target``, clamped at zero); FAILED trials do NOT
count against the budget so retries are free. If the study does not yet
exist, prints ``--target`` *without* creating a stub — first-run config is
left to ``ddssm.app``'s sweeper so the right sampler / directions stick.

``--cleanup-running-older-than`` is now an advisory flag (kept for back-
compat with sbatch preambles emitted before this change): orphaned
RUNNING trials are reaped via ``optuna.storages.fail_stale_trials``,
which uses Optuna's heartbeat mechanism — a healthy trial still updates
its heartbeat and is therefore safe from reaping no matter how long it
has been running.  When the storage is NOT heartbeat-configured (the
current default), ``fail_stale_trials`` is a no-op; in that mode
orphaned RUNNING rows persist in the DB but do not affect the budget
count (only COMPLETE + PRUNED count toward target).  Enable heartbeats
by configuring the storage with ``RDBStorage(..., heartbeat_interval=...,
grace_period=..., failed_trial_callback=...)``.
"""

from __future__ import annotations

import sys
import argparse

import optuna
from optuna.trial import TrialState

# Budget states (per locked decision #3): COMPLETE + PRUNED count toward target.
_BUDGET_STATES = (TrialState.COMPLETE, TrialState.PRUNED)


def compute_remaining(
    storage: str,
    study_name: str,
    target: int,
    cleanup_older_than: int | None = None,
) -> int:
    """Return how many trials the study still owes to reach ``target``.

    Tries ``optuna.load_study(...)``. If the study does not exist
    (``KeyError`` from the RDB backend — ``load_study`` already raises when
    the study is missing and never creates one) we return ``target`` without
    side-effecting a stub into the storage. The sweeper's first invocation
    is what should create the study with the right sampler/directions.

    If ``cleanup_older_than`` is non-None, attempts a heartbeat-based
    reap of orphaned RUNNING trials via ``optuna.storages.fail_stale_trials``.
    The actual stale-detection window is governed by Optuna's storage-side
    ``heartbeat_interval`` + ``grace_period`` (not by ``cleanup_older_than``,
    which is now advisory); a healthy trial whose worker is updating its
    heartbeat is never reaped, no matter how long it has been running.

    Returns ``max(0, target - count(COMPLETE+PRUNED))``.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except (KeyError, ValueError):
        # ``load_study`` raises (and does not create) when the study is
        # missing — RDB backends raise KeyError; some other backends raise
        # ValueError. Either way: no study, no count, no side effects.
        return target

    if cleanup_older_than is not None:
        _reap_stale_running(study)

    count = len(study.get_trials(states=_BUDGET_STATES, deepcopy=False))
    return max(0, target - count)


def _reap_stale_running(study: optuna.Study) -> None:
    """Flip orphaned RUNNING trials to FAILED via Optuna's heartbeat reaper.

    Delegates to ``optuna.storages.fail_stale_trials`` which uses the
    storage's heartbeat mechanism to identify trials whose worker has
    stopped sending heartbeats.  Crucially, a healthy worker still
    sending heartbeats is NEVER reaped regardless of how long the trial
    has been running — fixing the prior ``datetime_start``-based reaper
    that flipped healthy trials older than 60 s to FAILED, burning their
    compute and double-spending budget slots.

    When the storage is not heartbeat-configured (no
    ``heartbeat_interval`` on ``RDBStorage``) this is a no-op; the
    failure mode in that case is that genuinely orphaned RUNNING rows
    persist in the DB, which is benign (they do not affect the budget
    count). To enable real reaping, configure the storage with
    ``RDBStorage(..., heartbeat_interval=N, grace_period=M)``.
    """
    optuna.storages.fail_stale_trials(study)


def main(argv: list[str] | None = None) -> int:
    """Argparse entrypoint — prints the remaining count to stdout.

    Flag names are baked into the sbatch preamble (see ``ddssm.cluster.sbatch``);
    do not rename without updating that emitter.
    """
    p = argparse.ArgumentParser(prog="python -m ddssm.launch_remaining")
    p.add_argument(
        "--storage", required=True, help="Optuna storage URL (e.g. sqlite:///path.db)"
    )
    p.add_argument("--study", required=True, help="Optuna study name")
    p.add_argument("--target", type=int, required=True, help="Total trial budget")
    p.add_argument(
        "--cleanup-running-older-than",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Mark RUNNING trials older than this as FAILED before counting",
    )
    args = p.parse_args(argv)

    remaining = compute_remaining(
        storage=args.storage,
        study_name=args.study,
        target=args.target,
        cleanup_older_than=args.cleanup_running_older_than,
    )
    print(remaining)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["compute_remaining", "main"]
