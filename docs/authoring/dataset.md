# Step 0 — Picking a dataset

A model trains against a {py:class}`~ddssm.data.datamodule.DDSSMDataModule`. You
either reuse a library preset, configure the synthetic generator, or implement a
custom data module.

## The data-module interface

`DDSSMDataModule` (`src/ddssm/data/datamodule.py`) exposes `train_loader()`,
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

Subclass `DDSSMDataModule`, return loaders yielding the batch dict above, and
expose `DataMetadata`. Then either wrap it in `builds(MyDataModule, ...)` and
pass it as `experiment(data=...)`, or register it with
{py:obj}`ddssm.experiment.stores.data_store` (`package="experiment.data"`) so
`+data=my_dataset` works.

## In the worked example

`experiments/synthetic_validation/experiments.py` loops over a `dict` of
library presets and registers one experiment per dataset:

```python
from ddssm.data.presets import LGSSM, Bimodal, Harmonic

DATASETS = {"harmonic": Harmonic, "lgssm": LGSSM, "bimodal": Bimodal}
for tag, data in DATASETS.items():
    exp = experiment(data=data, model=SynthValModel(data_dim=1, latent_dim=1, j=1), ...)
    experiment_store(exp, name=f"synthval__{tag}")
```

All three are `D=1, T=32`, so one model shape (`data_dim=1`) fits all. To add
more, drop another `D=1` preset into `DATASETS` (e.g. `harmonic-noisy`); for a
multivariate run you'd also bump the model's `data_dim`/`latent_dim`
({doc}`model`).
