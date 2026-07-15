"""Model conf for the CSDI experiment family.

``CSDIModel`` targets :class:`~ddssm.adapters.csdi.CSDIAdapter` directly (it is a
``ModelAdapter`` subclass), so :func:`experiments._make.experiment` does NOT
wrap it — it curries the winning ``hparams`` on via ``dataclasses.replace(model,
config=hparams)``. Consequence: ``CSDIModel(...)`` must be a *constructable*
``builds()`` instance BEFORE the factory replaces ``config``, so it is
constructed in ``experiments.py`` with a placeholder ``config=<CSDIHparams>``
(the factory overrides it, so the placeholder is harmless).
"""

from __future__ import annotations

from hydra_zen import builds

from ddssm.adapters.csdi import CSDIAdapter
from experiments.csdi.hparams import CSDIHparams

# ``config`` is curried on by the experiment factory (dataclasses.replace); a
# CSDIHparams instance is passed as a harmless placeholder at construction.
#
# The explicit ``config=`` kwarg is load-bearing: without it,
# ``populate_full_signature=True`` bakes the strict ``config: CSDIConfig``
# annotation from ``CSDIAdapter.__init__``, and OmegaConf then rejects the
# factory's ``dataclasses.replace(model, config=<Builds_CSDIConfig>)`` (a builds
# config is not a ``CSDIConfig`` *subclass*). Passing ``config`` explicitly
# widens the field annotation to ``Any``, so the curried hparams config assigns
# cleanly while the full signature is still populated.
CSDIModel = builds(
    CSDIAdapter,
    populate_full_signature=True,
    config=CSDIHparams(),
)


__all__ = ["CSDIModel"]
