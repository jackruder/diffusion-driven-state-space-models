"""Objective + variance-probe specs (the variance-probe family's eval surface).

Variance-probe runs don't compute forecast metrics — the trainer's
loss tail is the Optuna objective, and the post-training
:class:`ProbeSpec` measures score-net variance against the trained
checkpoint. So this family's "evals" file holds those two specs
instead of an :class:`Eval`/:class:`Viz` pair.
"""

from __future__ import annotations

from ddssm.builders import Objective, Probe


# Optuna objective: mean of the last 10% of train-loss rows.
LossTail = Objective(metric="loss/total", split="train", tail_frac=0.1)

# Variance probe spec — runs against the checkpoint emitted by training.
VarianceProbe = Probe()
