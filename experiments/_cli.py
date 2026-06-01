"""Argparse-based CLI for the experiments registry.

Three subcommands::

    python -m experiments list
    python -m experiments run    [--run-dir=PATH]      <name> [hydra overrides...]
    python -m experiments sbatch [--out=PATH] [resource flags...]  <name> [hydra overrides...]

* ``list`` walks the hydra-zen ``experiment`` store and prints every
  registered name.
* ``run`` instantiates the named experiment via
  :func:`experiments._make.run` and trains it locally (writing
  outputs under ``runs/<name>`` unless ``--run-dir`` is given).
* ``sbatch`` renders a one-job Slurm submit script via
  :func:`ddssm.sbatch.render_sbatch`; writes to ``--out=`` or stdout. Does
  not call ``sbatch`` — submission is left to the user. (For launching a whole
  *study*, use ``python -m ddssm.launch <study>``.)

All flags must come **before** ``<name>``; everything after ``<name>`` is
forwarded as Hydra overrides (e.g. ``training.steps=200``).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from hydra_zen import instantiate, store

from ddssm._experiment_registry import register_experiments

from experiments._make import override, run
from ddssm.sbatch import render_sbatch


def _registered_names() -> list[str]:
    register_experiments()
    if "experiment" not in store:
        return []
    return sorted(n for _, n in store["experiment"])


def _get_experiment_node(name: str) -> Any:
    register_experiments()
    if "experiment" not in store or (("experiment", name)) not in dict(store["experiment"]):
        names = _registered_names()
        raise SystemExit(
            f"Unknown experiment {name!r}. Registered: {', '.join(names) or '<none>'}"
        )
    return store["experiment"][("experiment", name)]


def _cmd_list(_args: argparse.Namespace) -> int:
    for n in _registered_names():
        print(n)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    node = _get_experiment_node(args.name)
    if args.overrides:
        node = override(node, *args.overrides)
    run_dir = args.run_dir or f"runs/{args.name}"
    os.makedirs(run_dir, exist_ok=True)
    run(node, run_dir=run_dir)
    return 0


def _cmd_sbatch(args: argparse.Namespace) -> int:
    node = _get_experiment_node(args.name)
    exp = instantiate(node)
    cli_overrides = {
        "partition": args.partition,
        "time": args.time,
        "gpus": args.gpus,
        "cpus": args.cpus,
        "mem": args.mem,
        "nodes": args.nodes,
        "job_name": args.job_name,
    }
    text = render_sbatch(
        args.name,
        exp_sbatch=exp.sbatch,
        hydra_overrides=args.overrides or (),
        cli_overrides=cli_overrides,
        output_pattern=args.output,
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments",
        description="List, run, and emit sbatch scripts for named DDSSM experiments.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Print every registered experiment name.")

    pr = sub.add_parser(
        "run", help="Run a named experiment locally (writes to runs/<name>)."
    )
    pr.add_argument("name")
    pr.add_argument("--run-dir", default=None, help="Output directory (default runs/<name>).")
    pr.add_argument(
        "overrides", nargs=argparse.REMAINDER,
        help="Hydra-style overrides, e.g. 'training.steps=200'.",
    )

    ps = sub.add_parser(
        "sbatch", help="Render an sbatch submit script for an experiment."
    )
    ps.add_argument("name")
    ps.add_argument("--out", default=None, help="Write the script here (default stdout).")
    ps.add_argument("--partition", default=None)
    ps.add_argument("--time", default=None)
    ps.add_argument("--gpus", type=int, default=None)
    ps.add_argument("--cpus", type=int, default=None)
    ps.add_argument("--mem", default=None)
    ps.add_argument("--nodes", type=int, default=None)
    ps.add_argument("--job-name", default=None)
    ps.add_argument("--output", default=None, help="Slurm --output= log pattern.")
    ps.add_argument(
        "overrides", nargs=argparse.REMAINDER,
        help="Hydra overrides baked into the generated script.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the matching subcommand handler.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns:
        The subcommand's process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    cmd = {"list": _cmd_list, "run": _cmd_run, "sbatch": _cmd_sbatch}[args.cmd]
    return cmd(args)


__all__ = ["main"]
