"""A *study* — a family of experiments you run and compare.

Where :func:`experiments._make.experiment` builds **one** experiment (one
point, sweepable over hparams), a :class:`Study` is the *family of points*:
the matrix a `+sweep` is run across and a report aggregates. It owns the
registered presets, exposes tag-filtered selection (used by the launcher),
and names its points.

This is deliberately kept in ``experiments/`` (not the library) and is
currently proven on a single family (init-centering); a generic base can be
extracted once a second study exists. The orchestration/scheduling layer (the
``plan-campaign`` skill) *consumes* a study; it does not define one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class StudyPoint:
    """One registered experiment in a study, plus the metadata to launch it.

    ``config`` bakes the *tiny* size; ``size_overrides`` yields the Hydra
    overrides that re-shape it for a named size (``"tiny"`` -> none,
    ``"paper"`` -> a doubled ``latent_dim``, etc.). ``tags`` carry the axis
    coordinates (baseline_form/mode/tracking, dataset, ...) for filtering.
    """

    name: str
    config: Any
    tags: Mapping[str, str]
    size_overrides: Callable[[str], list[str]] = field(
        default=lambda _size: []
    )


@dataclass(frozen=True)
class Study:
    """A named family of experiment points + the sweep they share."""

    name: str
    points: tuple[StudyPoint, ...]
    sweep: str

    def register(self, store: Callable[..., Any]) -> None:
        """Register every point's config into a ``conf.registry`` store."""
        for p in self.points:
            store(p.config, name=p.name)

    def names(self) -> list[str]:
        return [p.name for p in self.points]

    def select(self, **tag_filters: Any) -> list[StudyPoint]:
        """Return points whose ``tags`` match every filter.

        A filter value may be a scalar (exact match) or a collection
        (membership). ``select(baseline_form="mlp", dataset={"1d", "mv"})``.
        """

        def matches(p: StudyPoint) -> bool:
            for key, want in tag_filters.items():
                have = p.tags.get(key)
                if isinstance(want, (set, frozenset, list, tuple)):
                    if have not in want:
                        return False
                elif have != want:
                    return False
            return True

        return [p for p in self.points if matches(p)]

    def point(self, name: str) -> StudyPoint:
        for p in self.points:
            if p.name == name:
                return p
        raise KeyError(f"no point named {name!r} in study {self.name!r}")


__all__ = ["Study", "StudyPoint"]
