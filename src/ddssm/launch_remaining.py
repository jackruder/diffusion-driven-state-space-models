"""Compute how many trials a preemptive sweep still owes its target.

Invoked from the bash preamble that ``ddssm.sbatch`` emits under
``PointLaunch.preemptive=True`` (ADR-0009):

    N_REMAINING=$(python -m ddssm.launch_remaining \\
        --storage <storage> --study <name> --target <target> \\
        --cleanup-running-older-than 60)

The CLI prints a single integer to stdout (the count of COMPLETE+PRUNED
trials subtracted from ``--target``, clamped at zero); FAILED trials do NOT
count against the budget so retries are free. If the study does not yet
exist, prints ``--target`` *without* creating a stub — first-run config is
left to ``ddssm.app``'s sweeper so the right sampler / directions stick.

``--cleanup-running-older-than N`` reaps RUNNING trials whose
``datetime_start`` is older than ``N`` seconds (orphans from a previous
preempt cycle), flipping them to FAILED. With ``DDSSM_PREEMPTIVE=1`` set
on the cleanup invocation, the storage's installed
``failed_trial_callback`` fires for each and enqueues a retry.
"""

from __future__ import annotations

import argparse
import datetime
import sys

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

    If ``cleanup_older_than`` is set, RUNNING trials whose ``datetime_start``
    is older than ``cleanup_older_than`` seconds are flipped to FAILED first
    (so the next worker doesn't deadlock waiting for orphaned slots and so
    the retry callback — if installed via ``DDSSM_PREEMPTIVE=1`` — fires).

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
        _reap_stale_running(study, older_than_seconds=cleanup_older_than)

    count = len(study.get_trials(states=_BUDGET_STATES, deepcopy=False))
    return max(0, target - count)


def _reap_stale_running(study: optuna.Study, *, older_than_seconds: float) -> None:
    """Flip RUNNING trials older than ``older_than_seconds`` to FAILED.

    ``study.get_trials`` returns ``FrozenTrial`` snapshots (not live ``Trial``
    objects), so ``study.tell(frozen, ...)`` rejects them with
    ``TypeError: Trial must be a trial object or trial number``. We pass
    ``.number`` instead — same FAILED transition, and any installed
    ``failed_trial_callback`` fires identically.
    """
    cutoff = datetime.datetime.now() - datetime.timedelta(seconds=older_than_seconds)
    for trial in study.get_trials(states=[TrialState.RUNNING], deepcopy=False):
        if trial.datetime_start is not None and trial.datetime_start < cutoff:
            study.tell(trial.number, state=TrialState.FAIL)


def main(argv: list[str] | None = None) -> int:
    """Argparse entrypoint — prints the remaining count to stdout.

    Flag names are baked into the sbatch preamble (see ``ddssm.sbatch``);
    do not rename without updating that emitter.
    """
    p = argparse.ArgumentParser(prog="python -m ddssm.launch_remaining")
    p.add_argument("--storage", required=True, help="Optuna storage URL (e.g. sqlite:///path.db)")
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
