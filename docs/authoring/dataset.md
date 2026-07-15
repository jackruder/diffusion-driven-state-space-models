# Step 0 — Picking a dataset

A model trains against a {py:class}`~ddssm.data.datamodule.TimeSeriesDataModule`. You
either reuse a library preset, configure the synthetic generator, or implement a
custom data module.

## The data-module interface

`TimeSeriesDataModule` (`src/ddssm/data/datamodule.py`) exposes `train_loader()`,
`val_loader()`, `test_loader()`, a `batch_transform`, and a
{py:class}`~ddssm.data.datamodule.DataMetadata` (`data_dim`, `T`,
`covariate_dim`, `forecast_split`, normalization stats, …). Implementations:

- {py:class}`~ddssm.data.datamodule.SyntheticDataModule` — closed-form synthetic
  generators (below).
- {py:class}`~ddssm.data.datamodule.KDDDataModule` — real PM2.5 windowed series.
- {py:class}`~ddssm.data.datamodule.NullDataModule` — no data (skips
  `trainer.fit`); used for build-only smoke tests.
- GluonTS loaders (`src/ddssm/data/gluonts.py`).

Each batch is a dict the model consumes (all moved to device by
`parse_batch`):

| Key | Shape | Notes |
| --- | ----- | ----- |
| `observed_data` | `(B, D, T)` | the sequence |
| `observation_mask` | `(B, D, T)` | 1 = observed; all ones for synthetic |
| `timepoints` | `(B, T)` | integer time index |
| `gt_latent` | `(B, d, T)` | only when `expose_gt_latents=True` and the mode supports it |

## Synthetic modes

{py:class}`~ddssm.data.synthetic.SyntheticDataModule` (`mode=...`) provides
closed-form generators (`src/ddssm/data/synthetic.py`):

| `mode` | Process |
| ------ | ------- |
| `iid` | i.i.d. Gaussian (baseline) |
| `lgssm` | linear-Gaussian SSM: `z_t = 0.9 z_{t-1} + 0.1·N`; `x = z + 0.1·N` |
| `nonlinear` | `z_t = sin(3 z_{t-1}) + 0.1·N` |
| `nongaussian` | mixture-of-Gaussians (bimodal jump) transition |
| `harmonic` | sine waves `sin(ωt+φ)`, ω∈[0.3,0.6], low noise (σ≈0.05) |
| `harmonic-noisy` | as above, ω∈[0.1,0.4], moderate noise (σ≈0.2) |
| `bimodal` / `bimodal-noisy` / `bimodal-block` | bimodal random walks |
| `nonlinear-bimodal-lift` / `-mv` | 1-D / multivariate latent lifted to obs (exposes GT latents) |
| `robot-basis-pursuit` | 2-D navigation around an obstacle (needs `D≥2`) |
| `student_t` | heavy-tailed AR process |

Key constructor args: `mode`, `D` (channels), `T` (length), `N_per_split`,
`batch_size`, `dataset_seed`, `expose_gt_latents`, `use_observation_mask`.

## Library presets

`src/ddssm/data/presets.py` wraps `SyntheticDataModule` in `builds(...)` configs
(all `T=32`); `experiments/datasets.py` registers them into the `data` group:

`LGSSM` (`lgssm`), `Harmonic` (`harmonic`), `Bimodal` (`bimodal`),
`BimodalNoisy` (`bimodal_noisy`), `Robot2D` (`robot2d`),
`NonlinBimodalLift1D` (`nonlin_bimodal_lift_1d`), `NonlinBimodalLiftMV`
(`nonlin_bimodal_lift_mv`).

### `+data=NAME` vs `experiment.data.X=V` — two different operations

These look similar but do different things, and the `+data=` form trips people
up. The distinction:

- **`+data=bimodal` selects a whole config-group option.** `data` is a Hydra
  *config group* (a menu of dataset presets), and selecting one swaps the
  **entire** `experiment.data` subtree for that preset. You override a *field*
  with a dotted path; you select a *group option* by the group name.
- **`experiment.data.batch_size=64` overrides one field** of whatever dataset is
  currently in the tree — it doesn't change which dataset you're using.

```bash
python -m ddssm.app experiment=synthval__harmonic +data=bimodal            # swap the dataset
python -m ddssm.app experiment=synthval__harmonic experiment.data.batch_size=64   # tweak a field
```

Two things explain the exact syntax (both in `src/ddssm/experiment/stores.py`):

1. **Why `+` and not `data=`?** Group selections that are *already in the
   defaults list* are overridden bare (e.g. `wandb=enabled`). `data` is **not**
   in `config.yaml`'s defaults — each experiment bakes its own dataset — so you
   *append* the group with `+`. A bare `data=bimodal` errors: ``Could not
   override 'data' … use +data=``.
2. **Why does it land at `experiment.data` and not a top-level `data:`?** The
   store is registered with `data_store = store(group="data",
   package="experiment.data")`. The `package=` tells Hydra to merge the selected
   option **into `experiment.data`**, replacing the dataset the preset baked in —
   instead of writing an unread top-level `data:` key. So `+data=NAME` is exactly
   "replace `experiment.data` with preset `NAME`"; you just can't write it as
   `experiment.data=NAME` because that's a node path, not a group selector.

(See {doc}`../hydra` → "Add vs. override" for the general `+` / `++` rules.)

## A custom dataset

Subclass `TimeSeriesDataModule`, return loaders yielding the batch dict above, and
expose `DataMetadata`. Then either wrap it in `builds(MyDataModule, ...)` and
pass it as `experiment(data=...)`, or register it with
{py:obj}`ddssm.experiment.stores.data_store` (`package="experiment.data"`) so
`+data=my_dataset` works.

## In the worked example

`experiments/synthetic_validation/study.py` makes the **dataset** a comparison
axis: a `dict` of library presets, one experiment per dataset. The per-dataset
experiment is built by `_build(coords)`, which picks the preset for that
coordinate:

```python
from ddssm.data.presets import LGSSM, Bimodal, Harmonic

DATASETS = {"harmonic": Harmonic, "lgssm": LGSSM, "bimodal": Bimodal}

def _build(coords):
    data = DATASETS[coords["dataset"]]
    return experiment(data=data, model=SynthValModel(data_dim=1, latent_dim=1, j=1), ...)
```

The axis (`Axis("dataset", list(DATASETS), key=...)`) crosses into one preset per
dataset, `synthval__<tag>` — see {doc}`study` for how the axis becomes registered
presets. All three are `D=1, T=32`, so one model shape (`data_dim=1`) fits all.
To add more, drop another `D=1` preset into `DATASETS` (e.g. `harmonic-noisy`);
for a multivariate run you'd also bump the model's `data_dim`/`latent_dim`
({doc}`model`).
||||||| f055350
=======
# Step 0 — Picking a dataset

A model trains against a {py:class}`~ddssm.data.datamodule.TimeSeriesDataModule`. You
either reuse a library preset, configure the synthetic generator, or implement a
custom data module.

## The data-module interface

`TimeSeriesDataModule` (`src/ddssm/data/datamodule.py`) exposes `train_loader()`,
`val_loader()`, `test_loader()`, a `batch_transform`, and a
{py:class}`~ddssm.data.datamodule.DataMetadata` (`data_dim`, `T`,
`covariate_dim`, `forecast_split`, normalization stats, …). Implementations:

- {py:class}`~ddssm.data.datamodule.SyntheticDataModule` — closed-form synthetic
  generators (below).
- {py:class}`~ddssm.data.datamodule.KDDDataModule` — real PM2.5 windowed series.
- {py:class}`~ddssm.data.datamodule.NullDataModule` — no data (skips
  `trainer.fit`); used for build-only smoke tests.
- GluonTS loaders (`src/ddssm/data/gluonts.py`).

Each batch is a dict the model consumes (all moved to device by
`parse_batch`):

| Key | Shape | Notes |
| --- | ----- | ----- |
| `observed_data` | `(B, D, T)` | the sequence |
| `observation_mask` | `(B, D, T)` | 1 = observed; all ones for synthetic |
| `timepoints` | `(B, T)` | integer time index |
| `gt_latent` | `(B, d, T)` | only when `expose_gt_latents=True` and the mode supports it |

## Synthetic modes

{py:class}`~ddssm.data.synthetic.SyntheticDataModule` (`mode=...`) provides
closed-form generators (`src/ddssm/data/synthetic.py`):

| `mode` | Process |
| ------ | ------- |
| `iid` | i.i.d. Gaussian (baseline) |
| `lgssm` | linear-Gaussian SSM: `z_t = 0.9 z_{t-1} + 0.1·N`; `x = z + 0.1·N` |
| `nonlinear` | `z_t = sin(3 z_{t-1}) + 0.1·N` |
| `nongaussian` | mixture-of-Gaussians (bimodal jump) transition |
| `harmonic` | sine waves `sin(ωt+φ)`, ω∈[0.3,0.6], low noise (σ≈0.05) |
| `harmonic-noisy` | as above, ω∈[0.1,0.4], moderate noise (σ≈0.2) |
| `bimodal` / `bimodal-noisy` / `bimodal-block` | bimodal random walks |
| `nonlinear-bimodal-lift` / `-mv` | 1-D / multivariate latent lifted to obs (exposes GT latents) |
| `robot-basis-pursuit` | 2-D navigation around an obstacle (needs `D≥2`) |
| `student_t` | heavy-tailed AR process |

Key constructor args: `mode`, `D` (channels), `T` (length), `N_per_split`,
`batch_size`, `dataset_seed`, `expose_gt_latents`, `use_observation_mask`.

## Library presets

`src/ddssm/data/presets.py` wraps `SyntheticDataModule` in `builds(...)` configs
(all `T=32`); `experiments/datasets.py` registers them into the `data` group:

`LGSSM` (`lgssm`), `Harmonic` (`harmonic`), `Bimodal` (`bimodal`),
`BimodalNoisy` (`bimodal_noisy`), `Robot2D` (`robot2d`),
`NonlinBimodalLift1D` (`nonlin_bimodal_lift_1d`), `NonlinBimodalLiftMV`
(`nonlin_bimodal_lift_mv`).

Because they're registered in the `data` group, you can swap the dataset baked
into any preset from the CLI with `+data=NAME` (the `+` appends; see
{doc}`../hydra`):

```bash
python -m ddssm.app experiment=synthval__harmonic +data=bimodal
```

## A custom dataset

Subclass `TimeSeriesDataModule`, return loaders yielding the batch dict above, and
expose `DataMetadata`. Then either wrap it in `builds(MyDataModule, ...)` and
pass it as `experiment(data=...)`, or register it with
{py:obj}`ddssm.experiment.stores.data_store` (`package="experiment.data"`) so
`+data=my_dataset` works.

## In the worked example

`experiments/synthetic_validation/study.py` makes the **dataset** a comparison
axis: a `dict` of library presets, one experiment per dataset. The per-dataset
experiment is built by `_build(coords)`, which picks the preset for that
coordinate:

```python
from ddssm.data.presets import LGSSM, Bimodal, Harmonic

DATASETS = {"harmonic": Harmonic, "lgssm": LGSSM, "bimodal": Bimodal}

def _build(coords):
    data = DATASETS[coords["dataset"]]
    return experiment(data=data, model=SynthValModel(data_dim=1, latent_dim=1, j=1), ...)
```

The axis (`Axis("dataset", list(DATASETS), key=...)`) crosses into one preset per
dataset, `synthval__<tag>` — see {doc}`study` for how the axis becomes registered
presets. All three are `D=1, T=32`, so one model shape (`data_dim=1`) fits all.
To add more, drop another `D=1` preset into `DATASETS` (e.g. `harmonic-noisy`);
for a multivariate run you'd also bump the model's `data_dim`/`latent_dim`
({doc}`model`).
