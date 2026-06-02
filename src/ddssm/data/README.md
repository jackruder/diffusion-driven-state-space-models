# `ddssm.data`

Dataset loaders for DDSSM experiments. Everything is exposed through a single
interface, `DDSSMDataModule`, which advertises `train_loader()`, `val_loader()`,
`test_loader()` (plus the shared `loader(split)` dispatch) and a
`batch_transform` that maps a raw batch + device to the canonical model-ready
dict. A `DataMetadata` block published alongside the loaders carries the shape /
normalization info (`data_dim`, `covariate_dim`, `T`, `use_observation_mask`,
optional `means`/`stds`, and the past/future `forecast_split`) the experiment
uses to wire the model. Each module advertises a `batch_format` of `"sequence"`
(full `(D, T)` sequences, synthetic) or `"windowed"` (`(D, L1+L2)` past/future
windows with real masks, KDD); both route through `parse_batch`.

## Files

- `datamodule.py` — the `DDSSMDataModule` ABC, the `DataMetadata` dataclass, and
  the concrete modules: `SyntheticDataModule` (sequence), `KDDDataModule`
  (windowed; loads a preprocessed `.pt` payload), and `NullDataModule`
  (`train_loader()` returns `None`, so the experiment skips `trainer.fit` — used
  for smoke tests / interactive use).
- `dataload.py` — GluonTS-based loading utilities: sliding-window
  Dataset/loaders, `parse_batch` batch parsing, z-score scaling, and
  `build_loaders_for_expt` (shared train/val/test builder). GluonTS imports are
  guarded so the module degrades when GluonTS is absent.
- `synthetic.py` — `SyntheticDataset` plus the closed-form synthetic generators
  (IID, LGSSM, harmonic, bimodal, nonlinear-bimodal-lift 1-D and multivariate);
  also defines the NLBL constants shared with `ddssm.eval.synthetic_kernels`.
- `kdd.py` — KDD Cup 2018 PM2.5 loading from Monash `.tsf` files
  (`parse_kdd_tsf`); builds loaders via `build_loaders_for_expt`. (Note: the
  `KDDDataModule` class itself lives in `datamodule.py`.)
- `gluonts.py` — GluonTS repository dataset loaders (solar, electricity,
  traffic, taxi, wiki) with per-dataset window/length defaults.
- `presets.py` — reusable `SyntheticDataModule` configs (LGSSM, Harmonic,
  Bimodal, NLBL 1-D/MV, …) built on the fly; registered into the Hydra
  `data_store` by `experiments.datasets`.
- `__init__.py` — package marker.
