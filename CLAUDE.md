# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DDSSM = Diffusion-Driven State Space Models — a PyTorch framework for probabilistic time-series forecasting. It jointly trains a variational encoder/decoder over latent states and a transition prior (Gaussian or CSDI-style diffusion) under an ELBO objective. See `README.md` for the high-level overview and full preset table.

Python 3.13, PyTorch ≥ 2.9. Dependencies are managed with `uv` (see `uv.lock`).

## Commands

```bash
# Install (editable)
pip install -e .                 # or: pip install -e .[wandb]

# Run the unified entry point
python -m ddssm.app                                    # default: harmonic_gauss
python -m ddssm.app experiment=synthetic_diffusion     # any registered preset
python -m ddssm.app experiment=kdd_gauss experiment.training.steps=2000

# Optuna sweep (preset search space)
python -m ddssm.app --multirun experiment=synthetic_gauss +sweep=synthetic_lr \
    hydra.sweeper.n_trials=20 hydra.sweeper.study_name=foo \
    hydra.sweeper.storage=sqlite:///foo.db

# Variance probe (train, then probe from checkpoint)
python -m ddssm.app experiment=variance_probe_lgssm +sweep=variance_probe
python -m ddssm.variance experiment=variance_probe_lgssm \
    checkpoint='${experiment.checkpoint_dir}/ckpt_latest.pth' +sweep=variance_probe

# Tests
pytest tests/                          # full suite
pytest tests/test_model.py             # single file
pytest tests/test_model.py::test_foo   # single test
pytest -m "not slow"                   # skip slow-marked tests

# Lint / format (ruff via pre-commit)
pre-commit run --all-files
```

CI runs `pre-commit run --all-files` (`.github/workflows/`). Ruff config is in `ruff.toml` (line length 88, google-style docstrings, isort with `force-sort-within-sections` and `length-sort`).

## Architecture

The code is structured around a single composition point — `Experiment` — that's built via hydra-zen and dispatched by Hydra. Reading these is enough to navigate everything else.

### Entry point: `src/ddssm/app.py`
`@hydra.main` reads `src/ddssm/conf/config.yaml`. On startup it calls `register_experiments()` (`_experiment_registry.py`), which:

1. Adds the repo root to `sys.path` so `import experiments` works.
2. Imports the `experiments/` package — its submodules (`experiments/synthetic/`, `experiments/kdd/`, `experiments/variance_probe/`) register named presets into the hydra-zen `store` singleton at import time.
3. Calls `store.add_to_hydra_store(overwrite_ok=True)` to publish everything to Hydra's `ConfigStore`.

Then `instantiate(cfg.experiment)` builds an `Experiment` and `experiment.train(device, run_dir)` runs it. The Hydra run directory holds `metrics.csv`, `tb_logs/`, `checkpoints/`, and `resolved_config.yaml`.

### Two parallel config worlds — important

- `src/ddssm/conf/` is a **small library of reusable Hydra YAML defaults**: top-level `config.yaml`, `hydra/launcher/`, `hydra/sweeper/`, `wandb/`. It does NOT contain the experiment presets themselves.
- `experiments/` (at the repo root) is where named presets are defined **in Python** via hydra-zen `builds(...)`. Each family subpackage (`synthetic/`, `kdd/`, `variance_probe/`) has the same shape: `datasets.py`, `models.py`, `encoders.py`, `decoders.py`, `z_inits.py`, `transitions.py`, `training.py`, `hparams.py`, `evals.py`, `vizs.py`, `sweeps.py`, and `experiments.py` which ties them together via `experiments._make.experiment(...)` and registers them with `experiment_store(...)`.

If you need a new experiment, **add a Python file under `experiments/<family>/`** and register it — don't add YAML.

`src/ddssm/builders.py` is the central import point for all hydra-zen `builds(...)` configs (models, encoders, transitions, etc.). Use these when composing experiments in Python.

### Experiment object
`src/ddssm/experiment.py` defines `Experiment`, a dataclass that owns:
- `data: DDSSMDataModule` — train/val/test loaders + batch transform
- `model: DDSSM_base` — the variational SSM (`src/ddssm/dssd.py`)
- `build_trainer: Callable[..., DDSSMTrainer]` — partial trainer factory
- `training: TrainingScalars` — `steps`, `log_every`, `validate_every`, `trainable` (per-module `requires_grad` mask), etc.
- `eval`, `viz`, `variance` — `EvalSpec` / `VizSpec` / `ProbeSpec` for the corresponding standalone stages
- `objective: ObjectiveSpec` — reads `metrics.csv` and returns the mean tail loss; used as the Optuna objective. If `None`, `train()` returns the trainer instead.

`Experiment.train`, `evaluate`, `visualize`, `variance_probe` are independent entry methods — eval/viz/variance load a checkpoint and don't trigger training.

### Model: `DDSSM_base` (`src/ddssm/dssd.py`)
ELBO over latent state-space: encoder `q_ϕ(z|x)`, decoder `p_θ(x|z)`, init prior `p_η(z_{1:j})`, transition `p_ψ(z_t|z_{t-j:t-1})`. The transition is pluggable: `GaussianTransition` or `DiffusionTransition` (CSDI U-Net denoiser). `j` is the history window. `forecast(...)` rolls out autoregressively.

The model exposes a `ProbeBatch` payload (encoded latents + log-q paths) reused by the variance probe.

### Pluggable sub-architectures (config groups)
Selected via top-level Hydra groups: `transition`, `encoder`, `decoder`, `z_init`, `context`, `unet`, `time_mixer`, `feature_mixer`. The CSDI residual stack composes per-channel time and feature mixers (`conv`/`gru`/`identity` × `transformer`/`conv`/`identity`); MLP variants exist as ablations. See the table in `README.md`.

### Trainer: `DDSSMTrainer` (`src/ddssm/train.py`)
Owns the optimizer per submodule (separate LRs for encoder/decoder/z_init/transition), AMP, CSV/TensorBoard/W&B logging, checkpointing, and the λ-warmup schedule. `_set_trainable(...)` toggles `requires_grad` per submodule — that mask is the single mechanism for stage-aware gradient suppression. The forward pass always computes every ELBO term; frozen submodules just don't accumulate gradients.

### Multi-stage training: `src/ddssm/stages.py`
`StageOrchestrator` runs sequential phases (e.g. recon-only → trans-only → joint) with per-stage trainable masks, LRs, scheduler, and λ-ramp. Configured via `StagesConf`.

### Standalone stages (no training)
Each has its own runner + registry (metric/plot dict). All load a checkpoint and read `metrics.csv` from the run dir.
- `src/ddssm/eval/`     — `EvalSpec`, metric registry (MAE, CRPS-sum, recon divergence), writes `metrics.json`
- `src/ddssm/viz/`      — `VizSpec`, plot registry (`forecast_1d`, etc.), writes PNGs
- `src/ddssm/variance/` — `ProbeSpec`, variance-probe metrics + plots, writes `variance_raw.csv` + `variance_summary.json`. Invoked via `python -m ddssm.variance`.

### Data modules: `src/ddssm/data/`
`DDSSMDataModule` is the interface (`train_loader()`, `val_loader()`, `test_loader()`, `batch_transform`). Implementations: `SyntheticDataModule`, `KDDDataModule`, `NullDataModule` (skips `trainer.fit(...)` — used for smoke tests). GluonTS loaders live in `gluonts.py`.

### Sweeps
Sweeper preset: `src/ddssm/conf/hydra/sweeper/ddssm_optuna.yaml` (uses the `dahlem/hydra` fork branch pinned in `pyproject.toml` for Optuna 4.2 support). Search spaces are presets under `src/ddssm/conf/sweep/` (e.g. `synthetic_lr`, `kdd_phase1`) or arbitrary CLI overrides on `hydra.sweeper.params.*`. SLURM launcher: `src/ddssm/conf/hydra/launcher/submitit_slurm.yaml`.

## Notes / conventions

- **Hydra `chdir=False`** — `Experiment.train` anchors checkpoints inside `run_dir` so per-run outputs are self-contained.
- **W&B is opt-in** — install `.[wandb]`; the logger silently no-ops if the package is missing. Activate by setting `wandb_config` on the `Experiment`.
- **`hparams` is shared** — `Experiment.hparams` and `Experiment.model.config.hyperparams` are kept identical so `tweak(exp, hparams__lr=1e-3)` works without descending into `model.config`. `Experiment.train` re-syncs them defensively.
- **Mamba is optional** — `build_mamba.sh` documents the manual build (CUDA arch-specific). Not in `pyproject.toml`.
- **Org files** — `verifications.org`, `esm_vs_dsm.org`, `model-v2.org` are literate-programming sources; `verifications.org` in particular originally tangled out config-group stores. Treat them as documentation, not the source of truth for running code.
