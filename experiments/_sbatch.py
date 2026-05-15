"""Slurm submit-script rendering for named experiments.

``python -m experiments sbatch <name>`` calls :func:`render_sbatch`
to emit a single-job ``.sbatch`` script that launches ``python -m
ddssm.app experiment=<name> "$@"`` under the requested resources.

Resource resolution (highest precedence wins):

1. ``--partition=``, ``--time=``, ``--gpus=``, ``--cpus=``, ``--mem=``,
   ``--job-name=``, ``--out=`` flags on the CLI.
2. ``Experiment.sbatch`` field on the named experiment, if set in
   ``experiments/<family>/experiments.py``.
3. The project default :data:`DEFAULT_SBATCH` below, which mirrors
   :file:`src/ddssm/conf/hydra/launcher/submitit_slurm.yaml`.

The generator stops short of submitting the script — that's
``sbatch <path>`` and intentionally left to the user. Extra positional
overrides on the CLI are forwarded to the script's ``"$@"`` so
``sbatch runs/foo.sbatch experiment.training.steps=200`` works.
"""

from __future__ import annotations

import dataclasses
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

    Parameters
    ----------
    name
        Name of a registered experiment.
    exp_sbatch
        The instantiated ``SBatch`` field on the experiment (may be
        ``None`` — falls back to the project default).
    hydra_overrides
        Hydra-style positional overrides to bake into the
        ``python -m ddssm.app`` invocation
        (e.g. ``["experiment.training.steps=4000"]``).
    cli_overrides
        Per-resource overrides from the CLI (``partition``, ``time``,
        ``gpus``, ``cpus``, ``mem``, ``nodes``, ``job_name``).
    output_pattern
        Slurm ``--output=`` log pattern. Defaults to
        ``runs/<name>/slurm-%j.out``.
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

    bake = " ".join(_shell_quote(o) for o in hydra_overrides)
    if bake:
        lines.append(f'exec python -m ddssm.app experiment={name} {bake} "$@"')
    else:
        lines.append(f'exec python -m ddssm.app experiment={name} "$@"')

    return "\n".join(lines) + "\n"


def _shell_quote(s: str) -> str:
    """Quote ``s`` for safe embedding in a generated bash script."""
    if not s or any(ch in s for ch in " \t\n\"'\\$`!*?(){}[]<>|&;#"):
        escaped = s.replace("'", "'\"'\"'")
        return f"'{escaped}'"
    return s


__all__ = ["DEFAULT_SBATCH", "render_sbatch"]
