"""A *Study* — a parametrized family of experiments to run and compare.

`experiments._make.experiment()` builds one point (one runnable / sweepable
experiment); a `Study` is the *family* of points — the matrix a sweep is run
across and a report aggregates. It is **pure** (no I/O): it holds opaque
experiment configs + the per-point launch intent, and exposes registration +
tag-filtered selection. Running a study is :class:`ddssm.launch.StudyOrchestrator`.

Define a study from axes (the common case) or an explicit point list::

    Study.from_axes(
        "init_centering",
        axes=[Axis("cell", cells, key=lambda c: c.name),
              Axis("dataset", datasets, key=lambda d: d.label)],
        build=lambda p: p["cell"].build(p["dataset"]),
        launch=lambda pt: PointLaunch(strategy="optuna_single_node", ...),
    )

**Axes are the comparison dimensions** — each combination is a distinct
*registered* preset. The Optuna **sweep** (the hyperparameter search *within* a
point) is part of the per-point launch intent, not an axis. **Replication**
(seeds) is an orchestrator concern, not an axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from ddssm.launch import PointLaunch


@dataclass(frozen=True)
class Axis:
    """One comparison dimension: a named set of values + a string key per value.

    ``key(value)`` is the primary token (used in the point name and as the
    ``{axis.name: key}`` tag). ``tags(value)`` optionally contributes *extra*
    tags so a compound value (e.g. a ``Cell``) can expose its sub-fields for
    ``select(...)`` and reporting (e.g. ``baseline_form``, ``tracking_mode``).
    """

    name: str
    values: Sequence[Any]
    key: Callable[[Any], str]
    tags: Callable[[Any], Mapping[str, str]] | None = None


@dataclass(frozen=True)
class StudyPoint:
    """One registered experiment in a study + the metadata to launch it.

    ``tags`` are the per-axis string keys (for ``select`` + naming); ``coords``
    are the raw axis values (so a per-point ``launch`` fn can scale resources
    off an axis value, e.g. ``j=16 -> nodes=4``).
    """

    name: str
    config: Any
    tags: Mapping[str, str]
    coords: Mapping[str, Any]


def _default_namer(prefix: str) -> Callable[[Mapping[str, str]], str]:
    def name_point(tags: Mapping[str, str]) -> str:
        parts = ([prefix] if prefix else []) + list(tags.values())
        return "__".join(parts)

    return name_point


@dataclass(frozen=True)
class Study:
    """A named family of experiment points + their per-point launch intent."""

    name: str
    points: tuple[StudyPoint, ...]
    launch: Callable[[StudyPoint], "PointLaunch"]
    variants: Mapping[str, Callable[[StudyPoint], list[str]]] = field(default_factory=dict)

    @classmethod
    def from_axes(
        cls,
        name: str,
        *,
        axes: Sequence[Axis],
        build: Callable[[Mapping[str, Any]], Any],
        launch: Callable[[StudyPoint], "PointLaunch"],
        name_point: Callable[[Mapping[str, str]], str] | None = None,
        variants: Mapping[str, Callable[[StudyPoint], list[str]]] | None = None,
        filter: Callable[[Mapping[str, Any]], bool] | None = None,
        prefix: str = "",
    ) -> "Study":
        """Build a study as the (filtered) cross-product of ``axes``.

        ``build(coords)`` maps ``{axis_name: value}`` to an experiment config;
        ``name_point(tags)`` maps ``{axis_name: key}`` to the preset name
        (default: ``"__".join`` of the keys, with an optional ``prefix``).
        """
        namer = name_point or _default_namer(prefix)
        points: list[StudyPoint] = []
        for combo in product(*[axis.values for axis in axes]):
            coords = {axis.name: val for axis, val in zip(axes, combo)}
            if filter is not None and not filter(coords):
                continue
            tags: dict[str, str] = {}
            for axis, val in zip(axes, combo):
                tags[axis.name] = axis.key(val)
                if axis.tags is not None:
                    tags.update(axis.tags(val))
            points.append(
                StudyPoint(name=namer(tags), config=build(coords), tags=tags, coords=coords)
            )
        return cls(name=name, points=tuple(points), launch=launch, variants=dict(variants or {}))

    @classmethod
    def from_points(
        cls,
        name: str,
        points: Sequence[StudyPoint],
        *,
        launch: Callable[[StudyPoint], "PointLaunch"],
        variants: Mapping[str, Callable[[StudyPoint], list[str]]] | None = None,
    ) -> "Study":
        """Escape hatch for irregular studies that don't fit a clean cross-product."""
        return cls(name=name, points=tuple(points), launch=launch, variants=dict(variants or {}))

    def register(self, store: Callable[..., Any]) -> None:
        """Register every point's config into a ``ddssm.stores`` store."""
        for p in self.points:
            store(p.config, name=p.name)

    def names(self) -> list[str]:
        return [p.name for p in self.points]

    def select(self, **tag_filters: Any) -> list[StudyPoint]:
        """Points whose ``tags`` match every filter (scalar = equals; collection = membership)."""

        def matches(p: StudyPoint) -> bool:
            for k, want in tag_filters.items():
                have = p.tags.get(k)
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


__all__ = ["Axis", "Study", "StudyPoint"]
