"""Lean Optuna sweep for the CSDI experiment family.

Both the arch (``channels``/``layers``) and optim (``lr``/``batch_size``) knobs
live on ``CSDIConfig`` (the adapter config == ``experiment.hparams``), so a
single :class:`~experiments._sweep.SweepSpace` with ``prefix="experiment.hparams"``
covers everything. The path is validated against ``CSDIHparams`` at import, so a
renamed field surfaces on ``python -m experiments list`` instead of at trial time.

DIVISIBILITY: CSDI's transformer requires ``nheads | channels``. ``nheads`` is
fixed (8), so ``channels`` is swept over a ``choice`` of multiples of 8
(32/64/96/128) rather than ``log_int(32, 128)`` — a log-int draw could yield a
non-divisible width and crash every such trial. Single objective: minimise the
validation loss (read from ``metrics.csv``; no per-trial forecast sampling).
"""

from __future__ import annotations

from experiments._sweep import SweepSpace
from ddssm.experiment.stores import sweep_store
from experiments.csdi.hparams import CSDIHparams

_lean = SweepSpace(target=CSDIHparams, prefix="experiment.hparams")
_lean.log("lr", 1e-4, 5e-3)
# channels swept over multiples of the fixed nheads=8 (see module docstring).
_lean.raw("channels", "choice(32, 64, 96, 128)")
_lean.log_int("layers", 2, 6)
_lean.raw("batch_size", "choice(8, 16, 32, 64)")

CSDILeanSweep = _lean.build(sweeper="ddssm_optuna", direction="minimize")
sweep_store(CSDILeanSweep, name="csdi_lean")


__all__ = ["CSDILeanSweep"]
