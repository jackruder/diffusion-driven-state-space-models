# A library `Study` abstraction for parametrized experiment families

`experiments._make.experiment()` builds **one** point — one runnable, sweepable
experiment. Nothing modeled the *family* of points you actually run and compare
(an ablation grid). That family — its enumeration, naming, registration, and the
matrix a report aggregates — was hand-rolled per experiment family: a `cells.py`,
a registration loop in `experiments.py`, dataset/size tuples duplicated across
launchers, and bespoke iteration in `report.py`. An in-family `Study`
(commit `fb2815f`) proved the shape; this ADR promotes it to a **general,
library-level** abstraction so a new ablation is a declaration, not a re-scaffold.

A *campaign* (the `plan-campaign` skill) schedules a study across resources; it
*consumes* a study and is out of scope here. A study is the *definition*.

## Decision

1. **`src/ddssm/study.py` (pure, no I/O)** provides `Axis`, `StudyPoint`, and
   `Study`. A `Study` holds a tuple of `StudyPoint`s (each: a registered `name`,
   an opaque experiment `config`, `tags`, and the raw `coords`) plus a per-point
   `launch` callable and named `variants`.

2. **Axis-declarative construction.** `Study.from_axes(name, axes=[...], build,
   name_point, launch, variants, filter)` takes the (optionally `filter`-ed)
   cross-product of the axes. `build(coords)` maps `{axis: value}` to an
   experiment config; `name_point(tags)` names the point (default: `"__".join`
   of the per-axis keys). An `Axis(name, values, key, tags=None)` exposes a
   primary `key(value)` token plus optional extra `tags(value)` so a compound
   value (a `Cell`) can surface its sub-fields (`baseline_form`, …) for
   filtering/reporting. `Study.from_points([...])` is the escape hatch for
   irregular studies.

3. **Axes are comparison dimensions; the sweep is not an axis.** Each axis
   combination is a **distinct registered preset** (`Study.register(store)`
   registers every point). The Optuna sweep — the hyperparameter search *within*
   a point — is part of the per-point launch intent (ADR-0008), never an axis.

4. **Replication (seeds) is not an axis** — it is an orchestrator knob
   (ADR-0008). Axes model *what you compare*, not *how many times you repeat it*.

5. `Study` exposes `register(store)`, `names()`, `select(**tag_filters)`
   (scalar = equals, collection = membership), `points`, `point(name)`.

## Considered alternatives

- **Thin flat-points `Study` (the in-family version), just moved to the library.**
  Rejected: each family would re-hand-write its cross-product, naming, and
  filtering. The axis form makes a new ablation a declaration; `from_points`
  retains the flexibility for the rare irregular case.
- **A declarative YAML study schema.** Rejected: CLAUDE.md mandates Python
  presets, not YAML; the spec-schema YAML is a scaffolding *input*, not the
  runtime model.
- **`Study` base class with overridable `points()/build()`.** Rejected:
  inheritance over composition fits this codebase (dataclasses + builders) worse
  than a declared `Study` value.
- **Make seeds / the sweep into axes.** Rejected: seeds are replication
  (orchestrator) and the sweep is intra-point search; folding either into the
  comparison grid conflates distinct concepts and explodes the registry.

## Consequences

- A new ablation = declare axes + a `build` fn + a `launch` fn; registration,
  launching (ADR-0008), and report iteration are generic.
- Large comparison axes register many presets (e.g. 16 for a `j=1..16` study;
  ~384 for `j × cell × dataset`) — accepted: hydra-zen configs are cheap and
  `experiment=<name>` always runs the real thing (no launch-time override
  indirection, which we removed by baking datasets into presets).
- The init-centering family shrinks to one `Study.from_axes` declaration
  (`experiments/init_centering/study.py`); `cells.py`'s grid point is now a
  `Cell` value (a `NamedTuple` that still unpacks like the old triple).
