"""Notebook-first DDSSM experiments.

This package holds one experiment per file as plain Python.  Each file
constructs an :class:`Experiment` config via :func:`make_experiment`
using the builders in :mod:`ddssm.builders`, and exposes the result as
the module-level ``exp`` variable.

Run an experiment from the shell::

    python -m experiments.harmonic_gauss
    python -m experiments.harmonic_diffusion run_dir=runs/harm_diff

…or from a notebook / org src block::

    from experiments.harmonic_diffusion import exp
    from experiments._make import run, to_yaml, save_yaml
    print(to_yaml(exp))
    run(exp, run_dir="runs/harm_diff_lownoise")

The Hydra CLI (``python -m ddssm.app experiment=NAME``) discovers
modules in this package by name; see :mod:`ddssm.app` for the bridge.
"""

from ._make import experiment, from_yaml, override, run, save_yaml, to_yaml

__all__ = [
    "experiment", "run", "to_yaml", "save_yaml", "from_yaml", "override",
]
