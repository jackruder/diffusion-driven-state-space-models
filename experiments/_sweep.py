"""Typed builder for Optuna search spaces with import-time validation.

A sweep search space is otherwise a ``dict[str, str]`` of free-form Hydra
dotted paths → Optuna distribution strings. Nothing checks that a path resolves
to a real config field, so a typo or a renamed factory argument crashes every
trial *after* the sweep launches (the override fails at trial time, on the
cluster). :class:`SweepSpace` validates each field against a target hydra-zen
``builds(...)`` config at registration time — so the mistake surfaces on import
(``python -m experiments list``) instead — and asserts the multi-objective
``direction`` length matches the paired objective. It emits the same
``hydra.sweeper.params`` dict the raw-string form did, so sweeps run identically.
"""

from __future__ import annotations

from typing import Any
import dataclasses

from hydra_zen import make_config


class SweepSpace:
    """Build a validated Optuna search space against a target config.

    Args:
        target: a hydra-zen ``builds(...)`` config (or any dataclass) whose
            fields the swept knobs must exist on — e.g. the stage-builder
            config ``StagesB``.
        prefix: dotted path prepended to each field to form the full Hydra
            override key, e.g. ``"experiment.training.stages"``.
    """

    def __init__(self, *, target: Any, prefix: str) -> None:
        self._valid = {f.name for f in dataclasses.fields(target)}
        self._target_name = getattr(target, "__name__", repr(target))
        self._prefix = prefix.rstrip(".")
        self._params: dict[str, str] = {}

    def _path(self, field: str) -> str:
        if field not in self._valid:
            raise ValueError(
                f"SweepSpace: {field!r} is not a field of {self._target_name} "
                f"(valid: {sorted(self._valid)}). Did the factory signature change?"
            )
        path = f"{self._prefix}.{field}"
        if path in self._params:
            raise ValueError(f"SweepSpace: duplicate sweep field {field!r}.")
        return path

    @staticmethod
    def _n(x: float | int) -> str:
        return repr(x)

    # --- distribution helpers (return self for chaining) ---
    def log(self, field: str, lo: float, hi: float) -> SweepSpace:
        """Log-uniform float over ``[lo, hi]``."""
        self._params[self._path(field)] = (
            f"tag(log, interval({self._n(lo)}, {self._n(hi)}))"
        )
        return self

    def log_int(self, field: str, lo: int, hi: int) -> SweepSpace:
        """Log-uniform integer over ``[lo, hi]``."""
        self._params[self._path(field)] = (
            f"tag(log, int(interval({self._n(lo)}, {self._n(hi)})))"
        )
        return self

    def uniform(self, field: str, lo: float, hi: float) -> SweepSpace:
        """Uniform float over ``[lo, hi]``."""
        self._params[self._path(field)] = f"interval({self._n(lo)}, {self._n(hi)})"
        return self

    def raw(self, field: str, distribution: str) -> SweepSpace:
        """Escape hatch: a raw Optuna distribution string (path still validated)."""
        self._params[self._path(field)] = distribution
        return self

    def params(self) -> dict[str, str]:
        """Return a copy of the accumulated ``hydra.sweeper.params`` dict."""
        return dict(self._params)

    def build(
        self,
        *,
        sweeper: str,
        direction: str | list[str],
        objectives: Any = None,
    ) -> Any:
        """Return the ``make_config`` sweep preset.

        ``sweeper`` is the sweeper group to override into (e.g.
        ``"ddssm_optuna"`` / ``"ddssm_optuna_moo"``). For multi-objective runs
        pass a ``direction`` list and the paired ``objectives`` config; their
        lengths must match.
        """
        if isinstance(direction, (list, tuple)) and objectives is not None:
            specs = getattr(objectives, "specs", None)
            if specs is not None and len(specs) != len(direction):
                raise ValueError(
                    f"SweepSpace: direction has {len(direction)} entries but the "
                    f"paired objective has {len(specs)} specs; they must match."
                )
        return make_config(
            hydra_defaults=["_self_", {"override /hydra/sweeper": sweeper}],
            hydra=dict(sweeper=dict(direction=direction, params=self.params())),
        )
