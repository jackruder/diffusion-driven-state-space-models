# `ddssm.experiment` — the composition layer

This package is the single composition point of DDSSM. `Experiment` ties a data
module, a model adapter, training scalars, optional eval/viz/variance
specs, and an Optuna objective into one dataclass that Hydra instantiates and
runs. The surrounding modules supply the hydra-zen config surface (`builders`),
the store→ConfigStore bridge (`stores`, `registry`), and re-exports so callers
import `Experiment` from the package root. No construction logic lives here:
Hydra wires the config; this class orchestrates.

## Files

- **`experiment.py`** — the `Experiment` dataclass, the central composition point.
  It owns `data` (`TimeSeriesDataModule`), `model` (a `ModelAdapter` — the raw
  `nn.Module` lives at `model.module`; `DDSSMAdapter` owns the `DDSSMTrainer`
  internally), `training` (`TrainingScalars`), `objective` (`ObjectiveSpec` /
  `Objectives` / `None`), and the `eval` / `viz` / `variance` specs, plus
  `hparams` (a `ModelConfig`, the single source of truth forwarded into
  `model.fit(...)`), `seed`, `wandb_config`, `sbatch`, and `model_config_yaml`.
  `train` delegates fit + checkpointing to the adapter; `evaluate`, `visualize`,
  and `variance_probe` are independent entry methods that load a checkpoint and
  don't trigger training. Also defines `TrainingScalars`, `ObjectiveSpec`
  (CSV/JSON objective reader with `penalty` fallback), `Objectives`
  (multi-objective wrapper), and `SBatch` (Slurm resource metadata).
- **`builders.py`** — a convenience surface of hydra-zen `builds(...)` configs
  (encoders, decoders, transitions, U-Nets, baselines, heads, data modules,
  `Training`, `Objective`, `SBatch`, `DDSSM`, `TrainerPartial`, `DDSSMAdapterC`
  (the `builds(DDSSMAdapter)` used by `_make.experiment` to wrap a bare model
  conf), eval/viz/probe specs, etc.) for assembling an `Experiment` ad hoc in a
  notebook, org src block, or script. The shipped `init_centering` family does
  **not** route through this — it composes the model from runtime classes
  directly.
- **`stores.py`** — pre-grouped partials of the hydra-zen `store` singleton, one
  per populated axis: `model_store` (`model`), `data_store` (`data`,
  packaged at `experiment.data` so `+data=NAME` overrides the preset's baked
  dataset), `experiment_store` (`experiment`), and `sweep_store` (`sweep`, merged
  at `_global_` so a preset can set `hydra.sweeper.*`, activated with
  `+sweep=NAME`). Finer-grained encoder/decoder/transition stores were removed as
  dead scaffolding.
- **`registry.py`** — `register_experiments()`: puts the repo root on `sys.path`
  so `import experiments` works, imports the `experiments/` package (its
  submodules self-register every preset into the hydra-zen `store` at import
  time), then calls `store.add_to_hydra_store(overwrite_ok=True)` to publish the
  whole registry into Hydra's `ConfigStore`.
- **`__init__.py`** — re-exports `Experiment`, `ObjectiveSpec`, `TrainingScalars`,
  `Objectives`, `SBatch` (and `_as_objective_spec`) from `experiment.experiment`,
  so `from ddssm.experiment import Experiment` works despite the class living in
  the package's `experiment.py` submodule.

## How it fits

`ddssm.app` (the `@hydra.main` entry point) calls `register_experiments()` on
startup, then `instantiate(cfg.experiment)` builds an `Experiment`, then
`experiment.train(device=..., run_dir=...)` runs it. To add a new preset, write a
Python module under `experiments/<family>/` and register it through the stores —
not here. This package is the machinery; the presets live in `experiments/`.
