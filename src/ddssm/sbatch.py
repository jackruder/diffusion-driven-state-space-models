"""Slurm submit-script rendering + submission for named experiments.

``render_sbatch`` emits a single-job ``.sbatch`` that launches
``python -m ddssm.app experiment=<name> "$@"`` under the requested resources;
``submit_sbatch`` shells out to ``sbatch``. Used by the standalone
``python -m experiments sbatch <name>`` CLI and by
:class:`ddssm.launch.StudyOrchestrator`.

Resource resolution (highest precedence wins): CLI overrides → the experiment's
``SBatch`` field (or the study point's :class:`ddssm.launch.PointLaunch`
resources) → :data:`DEFAULT_SBATCH`.

Preempt-aware rendering (ADR-0009): when a :class:`PreemptSpec` is passed via
``render_sbatch(..., preempt=...)``, the script gains three ``#SBATCH``
directives (``--requeue``, ``--signal=B:USR1@<grace>``, ``--open-mode=append``)
injected BEFORE the user's ``extra_flags`` (SLURM's last-line-wins keeps the
user's overrides authoritative), and a bash preamble that calls
``ddssm.launch_remaining`` to compute the remaining trial budget, divides it
across ``n_workers``, exports the env vars the trainer's signal handler keys
off, and runs the python child in the background under a ``trap`` that
forwards ``SIGUSR1``/``SIGTERM`` to it (so SLURM-driven preempt grace flows
all the way through to the in-flight trial).
"""

from __future__ import annotations

from typing import Iterable
import subprocess
import dataclasses
from dataclasses import dataclass

from ddssm.experiment import SBatch

# Project default Slurm resources for a single-job render.
DEFAULT_SBATCH = SBatch(
    partition="gpu",
    time="04:00:00",
    gpus=1,
    cpus=4,
    mem="32G",
    nodes=1,
)


# Placeholder token that the preempt-aware strategies emit for ``n_trials``;
# substituted in this file with the bash-side ``$N_PER_WORKER`` shell var.
_N_PER_WORKER_PLACEHOLDER = "__N_PER_WORKER__"


@dataclass(frozen=True)
class PreemptSpec:
    """Parameters the orchestrator passes to ``render_sbatch`` for preempt mode.

    ``grace_seconds`` is the ``--signal=B:USR1@<grace>`` lead-time; the trainer
    saves a checkpoint and raises ``PreemptError`` when it sees the signal.
    ``storage_url`` + ``study_name`` + ``target`` drive the preamble's
    ``ddssm.launch_remaining`` invocation (which computes the still-pending
    budget; it does NOT reap RUNNING trials — see the preamble for why).
    ``n_workers`` divides the remaining budget across siblings (ceiling
    division). ``worker_idx`` is
    baked into the ``DDSSM_WORKER_ID`` env export so each worker subprocess
    knows its slot.
    """

    grace_seconds: int
    storage_url: str
    study_name: str
    target: int
    n_workers: int
    worker_idx: int = 0


def _resolve(
    *,
    name: str,
    exp_sbatch: SBatch | None,
    overrides: dict[str, object],
) -> SBatch:
    """Merge project default → experiment-level spec → CLI overrides."""
    base = exp_sbatch if exp_sbatch is not None else DEFAULT_SBATCH
    merged = dataclasses.replace(
        base, **{k: v for k, v in overrides.items() if v is not None}
    )
    if merged.job_name is None:
        merged = dataclasses.replace(merged, job_name=f"ddssm-{name}")
    return merged


def render_sbatch(
    name: str,
    *,
    exp_sbatch: SBatch | None,
    hydra_overrides: Iterable[str] = (),
    cli_overrides: dict[str, object] | None = None,
    output_pattern: str | None = None,
    preempt: PreemptSpec | None = None,
) -> str:
    """Render an sbatch script for ``experiment=<name>``.

    ``exp_sbatch`` is the resolved resource spec (an ``SBatch``; may be
    ``None`` → project default). ``hydra_overrides`` are baked into the
    ``python -m ddssm.app`` invocation; ``cli_overrides`` are per-resource
    overrides; ``output_pattern`` defaults to ``runs/<name>/slurm-%j.out``.

    When ``preempt`` is passed, three additional ``#SBATCH`` directives are
    injected BEFORE the resolved spec's ``extra_flags`` (so the user's flags
    win under SLURM's last-line-wins semantics), and the ``exec python``
    one-liner is replaced with a preamble that computes the per-worker
    n_trials, exports preempt-aware env vars, and runs the python child in
    the background under a ``trap`` that forwards ``SIGUSR1``/``SIGTERM``.
    """
    spec = _resolve(name=name, exp_sbatch=exp_sbatch, overrides=cli_overrides or {})
    log_pattern = output_pattern or f"runs/{name}/slurm-%j.out"

    lines: list[str] = [
        "#!/bin/bash",
        f"#SBATCH --job-name={spec.job_name}",
        f"#SBATCH --partition={spec.partition}",
        f"#SBATCH --time={spec.time}",
        f"#SBATCH --gres=gpu:{spec.gpus}",
        f"#SBATCH --cpus-per-task={spec.cpus}",
        f"#SBATCH --mem={spec.mem}",
        f"#SBATCH --nodes={spec.nodes}",
        f"#SBATCH --output={log_pattern}",
    ]
    # Preempt directives go BEFORE the user's ``extra_flags`` so a user-supplied
    # ``--signal=...`` line (in ``extra_flags``) wins on SLURM's last-line
    # semantics.
    if preempt is not None:
        lines += [
            "#SBATCH --requeue",
            f"#SBATCH --signal=B:USR1@{preempt.grace_seconds}",
            "#SBATCH --open-mode=append",
        ]
    for flag in spec.extra_flags:
        lines.append(f"#SBATCH {flag}")

    lines += [
        "set -euo pipefail",
        'cd "$SLURM_SUBMIT_DIR"',
    ]
    lines += list(spec.setup)

    # Hydra's argparse expects ``--``-prefixed flags (e.g. ``--multirun``)
    # BEFORE positional overrides. Splitting + reordering avoids the
    # "unrecognized arguments" error that surfaces when ``--multirun``
    # sits in the middle of the override list.
    overrides = list(hydra_overrides)
    flag_args = [o for o in overrides if o.startswith("--")]
    kv_args = [o for o in overrides if not o.startswith("--")]
    flags_blob = " ".join(_shell_quote(o) for o in flag_args)
    kvs_blob = " ".join(_shell_quote(o) for o in kv_args)
    if preempt is not None:
        # The strategies emit ``hydra.sweeper.n_trials=__N_PER_WORKER__`` under
        # preemptive runs; bridge it to the shell-computed ``$N_PER_WORKER``.
        # CRITICAL: substitute AFTER ``_shell_quote`` — the placeholder has no
        # shell-special chars so it stays unquoted, and the resulting bare
        # ``$N_PER_WORKER`` expands at runtime. Substituting first would let
        # ``_shell_quote`` wrap the ``$`` in single quotes, so the literal
        # string ``$N_PER_WORKER`` reaches Hydra (n_trials becomes a str → the
        # Optuna sweeper's ``n_trials_to_go > 0`` raises a str/int TypeError).
        flags_blob = flags_blob.replace(_N_PER_WORKER_PLACEHOLDER, "$N_PER_WORKER")
        kvs_blob = kvs_blob.replace(_N_PER_WORKER_PLACEHOLDER, "$N_PER_WORKER")

    if preempt is None:
        parts = ["exec python -m ddssm.app"]
        if flags_blob:
            parts.append(flags_blob)
        parts.append(f"experiment={name}")
        if kvs_blob:
            parts.append(kvs_blob)
        parts.append('"$@"')
        lines.append(" ".join(parts))
    else:
        lines.extend(_render_preempt_preamble(preempt))
        child_parts = ["python -m ddssm.app"]
        if flags_blob:
            child_parts.append(flags_blob)
        child_parts.append(f"experiment={name}")
        if kvs_blob:
            child_parts.append(kvs_blob)
        child_parts.append('"$@" &')
        lines.append(" ".join(child_parts))
        lines += [
            "PID=$!",
            'wait "$PID"',
        ]

    return "\n".join(lines) + "\n"


def _render_preempt_preamble(ps: PreemptSpec) -> list[str]:
    """Bash preamble for preempt-aware rendering.

    Order matters: the launch_remaining call runs FIRST (under ``DDSSM_PREEMPTIVE=1``
    via the surrounding env exports it's still safe — the cleanup callback path
    is the explicit-enqueue model from app.py per the Phase 0/1 gate-test
    outcome). The early-exit guard short-circuits when the study has already
    hit its target; otherwise the worker computes its slice via ceiling
    division and dispatches the python child under a USR1/TERM trap so the
    trainer's signal handler sees the preempt signal.
    """
    return [
        # Budget count ONLY — no ``--cleanup-running-older-than``. The age-based
        # reap can't tell a live sibling's in-flight trial from a dead worker's
        # orphan, so it would fail the live trials of any other workers sharing
        # this study (catastrophic when groups join staggered or requeue). The
        # real resume path is independent: a SIGUSR1 worker checkpoints, raises
        # PreemptError, and app.py enqueues the retry. Hard-killed orphans just
        # linger as RUNNING (inert — they don't count against the budget).
        "N_REMAINING=$(python -m ddssm.launch_remaining \\",
        f"    --storage {ps.storage_url} --study {ps.study_name} \\",
        f"    --target {ps.target})",
        'if [ "$N_REMAINING" -le 0 ]; then',
        '    echo "[preempt] target reached, exiting cleanly"',
        "    exit 0",
        "fi",
        f"N_PER_WORKER=$(( (N_REMAINING + {ps.n_workers} - 1) / {ps.n_workers} ))",
        "export DDSSM_INVOC=$(date +%s)",
        f"export DDSSM_PREEMPTIVE=1 DDSSM_WORKER_ID={ps.worker_idx}",
        'trap \'kill -USR1 "$PID"; wait "$PID"\' USR1 TERM',
    ]


def render_packed_sbatch(
    name: str,
    *,
    exp_sbatch: SBatch | None,
    worker_overrides: list[tuple[int, list[str]]],
    cli_overrides: dict[str, object] | None = None,
    output_pattern: str | None = None,
    preempt: PreemptSpec | None = None,
) -> str:
    """Render ONE sbatch that runs K workers packed on the job's GPU(s).

    ``worker_overrides`` is a list of ``(worker_idx, hydra_overrides)`` — one
    entry per packed worker. All K workers share the job's single GPU and the
    cell's Optuna DB (same ``study_name`` / ``storage`` / ``sweep.dir``); each is
    pinned to ``spec.cpus // K`` CPU threads via ``OMP_NUM_THREADS`` /
    ``MKL_NUM_THREADS`` so K procs × that-many threads = ``spec.cpus`` with no
    oversubscription (the actual round-1 fix — round-1 starved 6 procs on 4 CPUs).

    Under ``preempt`` the shared preamble runs ``launch_remaining`` once and binds
    ``$N_PER_WORKER``; each worker exports its own ``DDSSM_WORKER_ID`` inline, and
    a single ``trap`` fans the preempt signal out to all packed PIDs.
    """
    if not worker_overrides:
        raise ValueError("render_packed_sbatch needs at least one worker")
    spec = _resolve(name=name, exp_sbatch=exp_sbatch, overrides=cli_overrides or {})
    log_pattern = output_pattern or f"runs/{name}/slurm-%j.out"
    k = len(worker_overrides)
    cpus_per_worker = max(1, spec.cpus // k)

    lines: list[str] = [
        "#!/bin/bash",
        f"#SBATCH --job-name={spec.job_name}",
        f"#SBATCH --partition={spec.partition}",
        f"#SBATCH --time={spec.time}",
        f"#SBATCH --gres=gpu:{spec.gpus}",
        f"#SBATCH --cpus-per-task={spec.cpus}",
        f"#SBATCH --mem={spec.mem}",
        f"#SBATCH --nodes={spec.nodes}",
        f"#SBATCH --output={log_pattern}",
    ]
    if preempt is not None:
        lines += [
            "#SBATCH --requeue",
            f"#SBATCH --signal=B:USR1@{preempt.grace_seconds}",
            "#SBATCH --open-mode=append",
        ]
    for flag in spec.extra_flags:
        lines.append(f"#SBATCH {flag}")

    lines += [
        "set -euo pipefail",
        'cd "$SLURM_SUBMIT_DIR"',
    ]
    lines += list(spec.setup)
    if preempt is not None:
        lines += _render_packed_preempt_preamble(preempt)
    else:
        # Non-preempt packed runs have no ``launch_remaining`` preamble, so the
        # Optuna schema is never initialized before the K workers spawn. Without
        # this, all K race on ``CREATE TABLE`` against the fresh SQLite file and
        # all-but-one die with ``sqlite3.OperationalError: table studies already
        # exists`` (observed: 15/16 A100 workers lost). Touch the storage once
        # here to create the tables — workers then only contend on the study-row
        # INSERT, which ``create_study(load_if_exists=True)`` handles. Mirrors
        # the schema init that ``launch_remaining`` performs for the preempt path.
        lines += _render_packed_storage_init(worker_overrides)

    lines.append("PIDS=()")
    if preempt is not None:
        # Fan the preempt signal out to every packed worker so each trainer's
        # handler checkpoints its in-flight trial and raises PreemptError.
        lines.append(
            'trap \'for _p in "${PIDS[@]}"; do kill -USR1 "$_p" 2>/dev/null; done; wait\' USR1 TERM'
        )

    for worker_idx, overrides in worker_overrides:
        flag_args = [o for o in overrides if o.startswith("--")]
        kv_args = [o for o in overrides if not o.startswith("--")]
        flags_blob = " ".join(_shell_quote(o) for o in flag_args)
        kvs_blob = " ".join(_shell_quote(o) for o in kv_args)
        if preempt is not None:
            # Substitute AFTER quoting so ``$N_PER_WORKER`` stays unquoted and
            # expands at runtime (see render_sbatch for the failure mode).
            flags_blob = flags_blob.replace(_N_PER_WORKER_PLACEHOLDER, "$N_PER_WORKER")
            kvs_blob = kvs_blob.replace(_N_PER_WORKER_PLACEHOLDER, "$N_PER_WORKER")
        parts = [
            f"DDSSM_WORKER_ID={worker_idx}",
            f"OMP_NUM_THREADS={cpus_per_worker}",
            f"MKL_NUM_THREADS={cpus_per_worker}",
            "python -m ddssm.app",
        ]
        if flags_blob:
            parts.append(flags_blob)
        parts.append(f"experiment={name}")
        if kvs_blob:
            parts.append(kvs_blob)
        parts.append('"$@" &')
        lines.append(" ".join(parts))
        lines.append("PIDS+=($!)")

    lines += [
        "STATUS=0",
        'for _p in "${PIDS[@]}"; do wait "$_p" || STATUS=$?; done',
        "exit $STATUS",
    ]
    return "\n".join(lines) + "\n"


def _render_packed_preempt_preamble(ps: PreemptSpec) -> list[str]:
    """Shared preempt preamble for a packed (K-workers-one-GPU) job.

    Like :func:`_render_preempt_preamble` but for the packed shape: it computes
    the cell-wide remaining budget ONCE (``ps.n_workers`` is the cell total, so
    ``$N_PER_WORKER`` is the per-worker slice across all packed workers), exports
    the shared ``DDSSM_INVOC`` + ``DDSSM_PREEMPTIVE`` env, and leaves the
    per-worker ``DDSSM_WORKER_ID`` export and the signal ``trap`` to
    :func:`render_packed_sbatch` (they are per-process / fan-out, not global).
    """
    return [
        # Budget count ONLY (no orphan reap) — see _render_preempt_preamble.
        "N_REMAINING=$(python -m ddssm.launch_remaining \\",
        f"    --storage {ps.storage_url} --study {ps.study_name} \\",
        f"    --target {ps.target})",
        'if [ "$N_REMAINING" -le 0 ]; then',
        '    echo "[preempt] target reached, exiting cleanly"',
        "    exit 0",
        "fi",
        f"N_PER_WORKER=$(( (N_REMAINING + {ps.n_workers} - 1) / {ps.n_workers} ))",
        "export DDSSM_INVOC=$(date +%s)",
        "export DDSSM_PREEMPTIVE=1",
    ]


def _render_packed_storage_init(
    worker_overrides: list[tuple[int, list[str]]],
) -> list[str]:
    """One-time Optuna schema init for the non-preempt packed shape.

    Extracts the shared ``hydra.sweeper.storage`` URL from the workers' override
    lists (they all carry the same value) and emits a single ``RDBStorage(url)``
    construction, which runs Optuna's ``_init_tables`` and creates the schema
    before the K workers start. Returns ``[]`` if no storage override is found
    (e.g. an in-memory sweeper), in which case there is no DDL race to avoid.
    """
    storage = None
    for _, overrides in worker_overrides:
        for o in overrides:
            if o.startswith("hydra.sweeper.storage="):
                storage = o.split("=", 1)[1]
                break
        if storage is not None:
            break
    if not storage:
        return []
    code = f'from optuna.storages import RDBStorage; RDBStorage("{storage}")'
    return [f"python -c {_shell_quote(code)}"]


def _shell_quote(s: str) -> str:
    """Quote ``s`` for safe embedding in a generated bash script."""
    if not s or any(ch in s for ch in " \t\n\"'\\$`!*?(){}[]<>|&;#"):
        escaped = s.replace("'", "'\"'\"'")
        return f"'{escaped}'"
    return s


def submit_sbatch(path: str) -> str:
    """Submit a rendered sbatch script via ``sbatch <path>``.

    Shells out to ``sbatch`` with the path as a single argv element (list form,
    no shell) so there is no injection surface. Returns ``sbatch``'s stripped
    stdout. Propagates ``FileNotFoundError`` if ``sbatch`` is missing and
    ``CalledProcessError`` on non-zero exit so the caller fails loudly.
    """
    result = subprocess.run(
        ["sbatch", path], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


__all__ = [
    "DEFAULT_SBATCH",
    "PreemptSpec",
    "render_packed_sbatch",
    "render_sbatch",
    "submit_sbatch",
]
