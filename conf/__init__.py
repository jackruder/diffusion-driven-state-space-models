"""Configuration package — exposes hydra-zen stores keyed by group.

The single concrete export is :mod:`conf.registry`, which defines one
``store(group="X")`` handle per axis-of-variation we want to register
named configs against. Every experiment family (``experiments.synthetic``,
``experiments.variance_probe``, ``experiments.kdd``) imports the
appropriate store from there and registers its named instances on import.
"""
