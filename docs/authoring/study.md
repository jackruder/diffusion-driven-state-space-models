# Step 7 — Defining a study

A single preset is one runnable experiment. A **study** is a *family* of presets
— the grid you compare across and launch together. It's the right tool once you
have more than a couple of cells (an ablation, a dataset sweep, a size axis).
Studies are defined with {py:class}`ddssm.cluster.study.Study` (ADR-0007/0008)
and launched with `python -m ddssm.launch`.

```{note}
A study replaces a hand-rolled registration loop. Earlier pages registered the
`synthval__*` presets one per dataset; the family actually does this through a
**Study** (`experiments/synthetic_validation/study.py`) — which both registers
the presets *and* makes the grid launchable. A plain loop is fine for a handful
of presets; reach for a Study when you want filtering, per-point launch intent,
size variants, and one-command launch.
```

## Anatomy

A `Study` is **pure** (no I/O): it holds the points and their launch intent.

- **`Axis`** — one comparison dimension: a `name`, a list of `values`, a
  `key(value)` → string token (used in the point name and as a `{name: key}`
  tag), and optional `tags(value)` for extra filter/report tags.
- **`StudyPoint`** — one registered experiment: `name`, `config` (the experiment
  built for this coordinate), `tags` (string keys, for `select` + naming), and
  `coords` (the raw axis values, so a launch fn can scale resources off them).
- **`Study.from_axes(name, axes=, build=, launch=, name_point=, variants=,
  filter=)`** — the common constructor: points are the (optionally `filter`-ed)
  cross-product of the axes. `build(coords)` maps `{axis: value}` →
  experiment config; `name_point(tags)` maps `{axis: key}` → preset name.
  `Study.from_points(...)` is the escape hatch for irregular grids.

Duplicate point names raise at construction (a name collision would silently
overwrite a preset).

## Per-point launch intent — `PointLaunch`

`Study.launch(point)` returns a {py:class}`ddssm.launch.PointLaunch` describing
*how* to run that point:

| Field | Meaning |
| ----- | ------- |
| `strategy` | a registered `LaunchStrategy` (the run shape, below) |
| `sweep` | `+sweep=` name for the Optuna strategies (the search *within* a point) |
| `n_trials` | total trial budget for the point (split across workers) |
| `n_workers` / `workers_per_gpu` | multi-worker / GPU-packing knobs |
| `resources` | a `ResourceSpec` (= `SBatch`) overriding the experiment's own sbatch; `None` = project default |
| `preemptive`, `preempt_grace_seconds` | requeue-safe preemptible runs |

Strategies (`ddssm.launch`): `single_job` (one run, no sweep), `optuna_single_node`
(one multirun on a node), `optuna_multi_node` (workers share an NFS DB),
`optuna_packed_node` (pack N workers per GPU), `local_parallel` (local
subprocesses sharing a SQLite DB).

```{important}
The **sweep is launch intent, not an axis** — the hyperparameter search *within*
a point (see {doc}`sweeps`). **Replication** (seeds) is an orchestrator concern
(`--seeds`), also not an axis. Axes are only the comparison dimensions.
```

## Registering

{py:func}`ddssm.launch.register_study` puts the study in the launcher registry
so `python -m ddssm.launch <name>` finds it. Pass `into=experiment_store` to
**also** publish every point's config as an `experiment=<point_name>` preset in
the same call (keeping the two registries in sync):

```python
SYNTHVAL_STUDY = register_study(study, into=experiment_store)
```

The study module must be imported for this side effect — the family's
`__init__.py` imports it, and `experiments/__init__.py` imports the family
(`register_experiments()` triggers the chain).

## Launching

```bash
python -m ddssm.launch synthval                 # dry-run: print sbatch per point
python -m ddssm.launch synthval --select dataset=harmonic   # filter by tag
python -m ddssm.launch synthval --size smoke    # apply a variant override hook
python -m ddssm.launch synthval --local --size smoke        # run each point locally
python -m ddssm.launch synthval --write-dir runs/sbatch     # write one .sbatch per job
python -m ddssm.launch synthval --write-dir runs/sbatch --submit   # ...and sbatch them
python -m ddssm.launch synthval --seeds 0 1 2   # replicate each point
```

- `--select K=V` filters points by tag (`Study.select`); repeatable.
- `--size` applies a named entry from the study's `variants` (e.g. `tiny` /
  `paper` / `smoke`) — a function returning extra Hydra overrides per point.
- `--dry-run` (default) prints sbatch; `--write-dir` writes scripts; `--submit`
  submits them; `--local` runs on this machine; `--seeds` replicates.
- `--storage-url` points all cells at one shared Optuna DB (else per-cell SQLite
  under `--storage-dir`) — see {doc}`sweeps` → "Storage backends" for the
  SQLite-vs-Postgres choice.

## The worked example

`experiments/synthetic_validation/study.py` is a one-axis study (`dataset`),
single training run per point, registered into the experiment store:

```python
SYNTHVAL_STUDY = register_study(
    Study.from_axes(
        "synthval",
        axes=[Axis("dataset", list(DATASETS), key=lambda tag: tag)],
        build=_build,                                   # → one dataset's experiment
        name_point=lambda tags: f"synthval__{tags['dataset']}",
        launch=lambda pt: PointLaunch(strategy="single_job", n_trials=1),
        variants={"tiny": lambda p: [], "smoke": _smoke_overrides},
    ),
    into=experiment_store,
)
```

Run the whole grid locally as a fast check:

```bash
python -m ddssm.launch synthval --local --size smoke
```

This trains each dataset cell once through both stages (it's a "does the model
recover known dynamics?" harness, not a hyperparameter search). To add an axis —
say a `latent_dim` size axis — add an `Axis` and let `_build` read it from
`coords`; the cross-product names (`synthval__<dataset>__<size>`) register
automatically.

## A richer reference

`experiments/init_centering/study.py` is the production example: two axes
(`cell` × `dataset`) → 24 points, the `cell` axis exposing `baseline_form` /
`tracking_mode` as filter tags, per-point `optuna_packed_node` launch intent with
real Tempest `ResourceSpec`s, and `paper` / `smoke` variants. It's the template
for a cluster-scale campaign.
||||||| f055350
=======
# Step 7 — Defining a study

A single preset is one runnable experiment. A **study** is a *family* of presets
— the grid you compare across and launch together. It's the right tool once you
have more than a couple of cells (an ablation, a dataset sweep, a size axis).
Studies are defined with {py:class}`ddssm.cluster.study.Study` (ADR-0007/0008)
and launched with `python -m ddssm.launch`.

```{note}
A study replaces a hand-rolled registration loop. Earlier pages registered the
`synthval__*` presets one per dataset; the family actually does this through a
**Study** (`experiments/synthetic_validation/study.py`) — which both registers
the presets *and* makes the grid launchable. A plain loop is fine for a handful
of presets; reach for a Study when you want filtering, per-point launch intent,
size variants, and one-command launch.
```

## Anatomy

A `Study` is **pure** (no I/O): it holds the points and their launch intent.

- **`Axis`** — one comparison dimension: a `name`, a list of `values`, a
  `key(value)` → string token (used in the point name and as a `{name: key}`
  tag), and optional `tags(value)` for extra filter/report tags.
- **`StudyPoint`** — one registered experiment: `name`, `config` (the experiment
  built for this coordinate), `tags` (string keys, for `select` + naming), and
  `coords` (the raw axis values, so a launch fn can scale resources off them).
- **`Study.from_axes(name, axes=, build=, launch=, name_point=, variants=,
  filter=)`** — the common constructor: points are the (optionally `filter`-ed)
  cross-product of the axes. `build(coords)` maps `{axis: value}` →
  experiment config; `name_point(tags)` maps `{axis: key}` → preset name.
  `Study.from_points(...)` is the escape hatch for irregular grids.

Duplicate point names raise at construction (a name collision would silently
overwrite a preset).

## Per-point launch intent — `PointLaunch`

`Study.launch(point)` returns a {py:class}`ddssm.launch.PointLaunch` describing
*how* to run that point:

| Field | Meaning |
| ----- | ------- |
| `strategy` | a registered `LaunchStrategy` (the run shape, below) |
| `sweep` | `+sweep=` name for the Optuna strategies (the search *within* a point) |
| `n_trials` | total trial budget for the point (split across workers) |
| `n_workers` / `workers_per_gpu` | multi-worker / GPU-packing knobs |
| `resources` | a `ResourceSpec` (= `SBatch`) overriding the experiment's own sbatch; `None` = project default |
| `preemptive`, `preempt_grace_seconds` | requeue-safe preemptible runs |

Strategies (`ddssm.launch`): `single_job` (one run, no sweep), `optuna_single_node`
(one multirun on a node), `optuna_multi_node` (workers share an NFS DB),
`optuna_packed_node` (pack N workers per GPU), `local_parallel` (local
subprocesses sharing a SQLite DB).

```{important}
The **sweep is launch intent, not an axis** — the hyperparameter search *within*
a point (see {doc}`sweeps`). **Replication** (seeds) is an orchestrator concern
(`--seeds`), also not an axis. Axes are only the comparison dimensions.
```

## Registering

{py:func}`ddssm.launch.register_study` puts the study in the launcher registry
so `python -m ddssm.launch <name>` finds it. Pass `into=experiment_store` to
**also** publish every point's config as an `experiment=<point_name>` preset in
the same call (keeping the two registries in sync):

```python
SYNTHVAL_STUDY = register_study(study, into=experiment_store)
```

The study module must be imported for this side effect — the family's
`__init__.py` imports it, and `experiments/__init__.py` imports the family
(`register_experiments()` triggers the chain).

## Launching

```bash
python -m ddssm.launch synthval                 # dry-run: print sbatch per point
python -m ddssm.launch synthval --select dataset=harmonic   # filter by tag
python -m ddssm.launch synthval --size smoke    # apply a variant override hook
python -m ddssm.launch synthval --local --size smoke        # run each point locally
python -m ddssm.launch synthval --write-dir runs/sbatch     # write one .sbatch per job
python -m ddssm.launch synthval --write-dir runs/sbatch --submit   # ...and sbatch them
python -m ddssm.launch synthval --seeds 0 1 2   # replicate each point
```

- `--select K=V` filters points by tag (`Study.select`); repeatable.
- `--size` applies a named entry from the study's `variants` (e.g. `tiny` /
  `paper` / `smoke`) — a function returning extra Hydra overrides per point.
- `--dry-run` (default) prints sbatch; `--write-dir` writes scripts; `--submit`
  submits them; `--local` runs on this machine; `--seeds` replicates.
- `--storage-url` points all cells at one shared Optuna DB (else per-cell SQLite
  under `--storage-dir`).

## The worked example

`experiments/synthetic_validation/study.py` is a one-axis study (`dataset`),
single training run per point, registered into the experiment store:

```python
SYNTHVAL_STUDY = register_study(
    Study.from_axes(
        "synthval",
        axes=[Axis("dataset", list(DATASETS), key=lambda tag: tag)],
        build=_build,                                   # → one dataset's experiment
        name_point=lambda tags: f"synthval__{tags['dataset']}",
        launch=lambda pt: PointLaunch(strategy="single_job", n_trials=1),
        variants={"tiny": lambda p: [], "smoke": _smoke_overrides},
    ),
    into=experiment_store,
)
```

Run the whole grid locally as a fast check:

```bash
python -m ddssm.launch synthval --local --size smoke
```

This trains each dataset cell once through both stages (it's a "does the model
recover known dynamics?" harness, not a hyperparameter search). To add an axis —
say a `latent_dim` size axis — add an `Axis` and let `_build` read it from
`coords`; the cross-product names (`synthval__<dataset>__<size>`) register
automatically.

## A richer reference

`experiments/init_centering/study.py` is the production example: two axes
(`cell` × `dataset`) → 24 points, the `cell` axis exposing `baseline_form` /
`tracking_mode` as filter tags, per-point `optuna_packed_node` launch intent with
real Tempest `ResourceSpec`s, and `paper` / `smoke` variants. It's the template
for a cluster-scale campaign.
