# `ddssm.data`

Dataset loaders for DDSSM experiments. Everything is exposed through a single
interface, `TimeSeriesDataModule`, which advertises `train_loader()`, `val_loader()`,
`test_loader()` (plus the shared `loader(split)` dispatch) and a
`batch_transform` that maps a raw batch + device to the canonical model-ready
dict. A `DataMetadata` block published alongside the loaders carries the shape /
normalization info (`data_dim`, `covariate_dim`, `T`, `use_observation_mask`,
optional `means`/`stds`, and the past/future `forecast_split`) the experiment
uses to wire the model. Each module advertises a `batch_format` of `"sequence"`
(full `(D, T)` sequences, synthetic) or `"windowed"` (`(D, L1+L2)` past/future
windows with real masks, KDD); both route through `parse_batch`.

## Files

- `datamodule.py` — the `TimeSeriesDataModule` ABC, the `DataMetadata` dataclass, and
  the concrete modules: `SyntheticDataModule` (sequence) and `NullDataModule`
  (`train_loader()` returns `None`, so the experiment skips `trainer.fit` — used
  for smoke tests / interactive use), plus `WindowedSeriesDataModule` — the
  windowed base that turns a `series_list` (+ optional dynamic / static
  covariates) into windowed loaders via `build_loaders_for_expt` — and its
  subclasses `KDDDataModule` (KDD Cup 2018 PM2.5; loads a preprocessed `.pt`
  payload) and `GluonTSDataModule` (lazily fetches a named GluonTS repository
  dataset: solar / electricity / traffic / taxi / wiki).
- `dataload.py` — windowing utilities: sliding-window Dataset/loaders,
  `parse_batch` batch parsing, z-score scaling, and `build_loaders_for_expt`
  (shared train/val/test builder, `"torch"` or `"gluonts"` backend). GluonTS
  imports are guarded so the module degrades when GluonTS is absent.
- `synthetic.py` — `SyntheticDataset` plus the closed-form synthetic generators
  (IID, LGSSM, harmonic, bimodal, nonlinear-bimodal-lift 1-D and multivariate,
  henon-lift); also defines the NLBL constants shared with
  `ddssm.eval.synthetic_kernels`.
- `mocap.py` — `MocapDataModule` (sequence) for the CMU MoCap subject-35
  walking benchmark (Wang-2007 preprocessing; 16/3/4 sequences × 300 × 50), with
  on-demand download of `data/mocap35.mat` from the canonical Course & Nair
  Dropbox mirror. Emits the same canonical batch dict as `SyntheticDataModule`.
- `presets.py` — reusable DataModule configs registered into the Hydra
  `data_store` by `experiments.datasets`: synthetic (LGSSM, Harmonic, Bimodal,
  NLBL 1-D/MV, …), GluonTS (solar / electricity / …), and KDD payloads.
- `__init__.py` — package marker.
