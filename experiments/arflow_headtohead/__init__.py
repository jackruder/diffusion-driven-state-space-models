"""Head-to-head family: gaussian vs IAF vs deterministic-causal encoders.

Compares the three encoders behind the ``ARFlowEncoder`` shell against the
sequential ``GaussianEncoder`` on an easy (``lgssm``) and a hard
(``nonlin_bimodal_lift_mv``) dataset, in two phases: a pure-AE capacity probe
(``h2h_cap__<enc>__<ds>``, ``+sweep=h2h_lr_only``) and the full two-stage ELBO
selected on forecast CRPS-sum (``h2h__<enc>__<ds>``, ``+sweep=h2h_full``). 12
cells + 2 sweeps. See :mod:`experiments.arflow_headtohead.experiments`.
"""

from . import (
    sweeps,  # noqa: F401  -- registers h2h_lr_only + h2h_full
    experiments,  # noqa: F401  -- registers the 12 cells
)

__all__ = ["sweeps", "experiments"]
