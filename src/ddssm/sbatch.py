"""Slurm submit-script rendering + submission for named experiments.

``render_sbatch`` emits a single-job ``.sbatch`` that launches
``python -m ddssm.app experiment=<name> "$@"`` under the requested resources;
``submit_sbatch`` shells out to ``sbatch``. Used by the standalone
``python -m experiments sbatch <name>`` CLI and by
:class:`ddssm.launch.StudyOrchestrator`.

Resource resolution (highest precedence wins): CLI overrides → the experiment's
``SBatch`` field (or the study point's :class:`ddssm.launch.PointLaunch`
resources) → :data:`DEFAULT_SBATCH`.
"""

from __future__ import annotations

import dataclasses
import subprocess
from typing import Iterable

from ddssm.experiment import SBatch


# Project default — mirrors `submitit_slurm.yaml`.
DEFAULT_SBATCH = SBatch(
    partition="gpu",
    time="04:00:00",
    gpus=1,
    cpus=4,
    mem="32G",
    nodes=1,
)


def _resolve(
    *,
    name: str,
    exp_sbatch: SBatch | None,
    overrides: dict[str, object],
) -> SBatch:
    """Merge project default → experiment-level spec → CLI overrides."""
    base = exp_sbatch if exp_sbatch is not None else DEFAULT_SBATCH
    merged = dataclasses.replace(base, **{k: v for k, v in overrides.items() if v is not None})
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
) -> str:
    """Render an sbatch script for ``experiment=<name>``.

    ``exp_sbatch`` is the resolved resource spec (an ``SBatch``; may be
    ``None`` → project default). ``hydra_overrides`` are baked into the
    ``python -m ddssm.app`` invocation; ``cli_overrides`` are per-resource
    overrides; ``output_pattern`` defaults to ``runs/<name>/slurm-%j.out``.
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
    for flag in spec.extra_flags:
        lines.append(f"#SBATCH {flag}")

    lines += [
        "set -euo pipefail",
        'cd "$SLURM_SUBMIT_DIR"',
    ]

    # Hydra's argparse expects ``--``-prefixed flags (e.g. ``--multirun``)
    # BEFORE positional overrides. Splitting + reordering avoids the
    # "unrecognized arguments" error that surfaces when ``--multirun``
    # sits in the middle of the override list.
    overrides = list(hydra_overrides)
    flag_args = [o for o in overrides if o.startswith("--")]
    kv_args = [o for o in overrides if not o.startswith("--")]
    flags_blob = " ".join(_shell_quote(o) for o in flag_args)
    kvs_blob = " ".join(_shell_quote(o) for o in kv_args)
    parts = ["exec python -m ddssm.app"]
    if flags_blob:
        parts.append(flags_blob)
    parts.append(f"experiment={name}")
    if kvs_blob:
        parts.append(kvs_blob)
    parts.append('"$@"')
    lines.append(" ".join(parts))

    return "\n".join(lines) + "\n"


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
    result = subprocess.run(["sbatch", path], check=True, capture_output=True, text=True)
    return result.stdout.strip()


__all__ = ["DEFAULT_SBATCH", "render_sbatch", "submit_sbatch"]
