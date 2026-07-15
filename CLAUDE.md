# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DDSSM = Diffusion-Driven State Space Models — a PyTorch framework for probabilistic time-series forecasting. It jointly trains a variational encoder/decoder over latent states and a transition prior (Gaussian or CSDI-style diffusion) under an ELBO objective. Head-to-head baselines (CSDI first) run through the *same* workflow via a `ModelAdapter` hierarchy — see `docs/adr/0011-model-adapters.md`. See `README.md` for the high-level overview and full preset table.

Python 3.13, PyTorch ≥ 2.12. Dependencies are managed with `uv` (see `uv.lock`).

## Commands

```bash
# Install (editable)
pip install -e .                 # or: pip install -e .[wandb]

# Run the unified entry point (default experiment: init_smoke_simple)
python -m ddssm.app                                       # default smoke
python -m ddssm.app experiment=init_smoke_high_surface    # any registered preset
python -m ddssm.app experiment=init_smoke_simple experiment.training.steps=400

# (The step budget is training.steps; adapter-owned knobs live under
#  experiment.hparams.* — the family's ModelConfig.)

# List / run / render-sbatch for any registered preset
python -m experiments list                                # enumerate preset names
python -m experiments run  init_smoke_simple training.steps=200
python -m experiments sbatch --out=job.sh init_smoke_high_surface

# Launch a whole registered study (renders/submits all its points)
python -m ddssm.launch init_centering --size smoke --local

# Optuna sweep (search spaces are Python presets, activated with +sweep=)
python -m ddssm.app --multirun experiment=init_smoke_high_surface +sweep=init_ablation \
    hydra.sweeper.n_trials=20 hydra.sweeper.study_name=foo \
    hydra.sweeper.storage=sqlite:///foo.db

# Standalone post-training stages (load a checkpoint; no training)
python -m ddssm.evaluate experiment=init_smoke_high_surface \
    +checkpoint='${experiment.checkpoint_dir}/ckpt_latest.pth'
python -m ddssm.visualize experiment=init_smoke_high_surface +checkpoint=...
python -m ddssm.variance  experiment=init_smoke_high_surface +checkpoint=...

# Tests (run inside the uv environment)
uv run pytest tests/                          # full suite
uv run pytest tests/test_model.py             # single file
uv run pytest tests/test_model.py::test_foo   # single test
uv run pytest -m "not slow"                   # skip slow-marked tests

# Lint / format (ruff via pre-commit)
pre-commit run --all-files
```

CI runs `pre-commit run --all-files` (`.github/workflows/`). Ruff config is in `ruff.toml` (line length 88, google-style docstrings, isort with `force-sort-within-sections` and `length-sort`).

## Architecture

The code is structured around a single composition point — `Experiment` — that's built via hydra-zen and dispatched by Hydra. Reading these is enough to navigate everything else.

`src/ddssm/` is organized into subpackages: `nn/` (reusable neural building blocks), `model/` (the SSM: `dssd`, encoder/decoder, losses, `config.py` (the `ModelConfig` base), and the `transitions/`/`centering/`/`likelihood/` subpackages), `adapters/` (the `ModelAdapter` hierarchy: `base`, `ddssm`, `csdi` + re-vendored `_csdi_vendor/`), `training/` (trainer, stages, checkpoint, loggers), `experiment/` (the `Experiment` composition root + `builders`/`stores`/`registry`), `cluster/` (sbatch/study/report orchestration), plus the standalone-stage packages `eval/`/`viz/`/`variance/`, `data/`, and `conf/`. The `python -m ddssm.<x>` entry points (`app`, `evaluate`, `visualize`, `launch`, `colocate`, `variance`, `launch_remaining`) stay at the top level. Intra-package imports are absolute (`from ddssm.<pkg>.<mod> import …`).

### Entry point: `src/ddssm/app.py`
`@hydra.main` reads `src/ddssm/conf/config.yaml`. On startup it calls `register_experiments()` (`experiment/registry.py`), which:

1. Adds the repo root to `sys.path` so `import experiments` works.
2. Imports the `experiments/` package — its submodules (families under `experiments/<family>/`, e.g. `init_centering/`, `csdi/`, `gluonts_forecast/`, `arflow_headtohead/`) register named presets into the hydra-zen `store` singleton at import time.
3. Calls `store.add_to_hydra_store(overwrite_ok=True)` to publish everything to Hydra's `ConfigStore`.

Run `python -m experiments list` to enumerate every registered preset.

Then `instantiate(cfg.experiment)` builds an `Experiment` and `experiment.train(device, run_dir)` runs it. The Hydra run directory holds `metrics.csv`, `tb_logs/`, `checkpoints/`, and `resolved_config.yaml`.

### Two parallel config worlds — important

- `src/ddssm/conf/` is a **small library of reusable Hydra YAML defaults**: top-level `config.yaml`, `hydra/sweeper/` (`ddssm_optuna.yaml`, `ddssm_optuna_moo.yaml`), and `wandb/`. It does NOT contain the experiment presets themselves.
- `experiments/` (at the repo root) is where named presets are defined **in Python** via hydra-zen `builds(...)`. Each family is a subpackage under `experiments/<family>/`; the primary DDSSM family is `experiments/init_centering/`, with baseline families alongside it (`experiments/csdi/`, `experiments/gluonts_forecast/`, `experiments/arflow_headtohead/`). A family's modules typically are: `datasets.py`/`data.py` (data axis), `model.py` (a factory composing the model — or, for a baseline, a `builds(...)` targeting the family's `ModelAdapter`), `hparams.py` (the family's `ModelConfig` build), `experiments.py` (registers presets via `experiments._make.experiment(...)` + `experiment_store(...)`), `evals.py`, and `sweeps.py` — `init_centering` additionally has `cells.py`/`study.py` (the ablation grid + study) and `report.py`. The shared `experiments/_make.py` is the `experiment(...)` factory (it wraps bare DDSSM model confs in a `DDSSMAdapter` and curries `hparams` onto adapter confs); `experiments/datasets.py` registers library dataset presets into `data_store`; `experiments/_cli.py` backs `python -m experiments`.

If you need a new experiment, **add a Python file under `experiments/<family>/`** and register it — don't add YAML. A new *baseline* family additionally needs a `ModelAdapter` subclass (see `src/ddssm/adapters/` and `docs/adr/0011-model-adapters.md`).

`src/ddssm/experiment/builders.py` is a convenience surface of hydra-zen `builds(...)` configs (models, encoders, transitions, etc.) for assembling an `Experiment` ad hoc — e.g. in a notebook or script. Note the live `init_centering` family does **not** route through it; `model.py` composes the model from runtime classes directly. Of its exports only a handful (`Synthetic`, `Eval`, `Objective`(s), `Hparams`, `Training`, `CenteringHandoff`, `ExperimentC`, `TrainerPartial`) are imported elsewhere in the repo.

### Experiment object
`src/ddssm/experiment/experiment.py` defines `Experiment`, a dataclass that owns
(re-exported as `ddssm.experiment.Experiment`):
- `data: TimeSeriesDataModule` — train/val/test loaders + batch transform
- `model: ModelAdapter` — a model-family adapter (`src/ddssm/adapters/`); the raw `nn.Module` lives at `model.module`. `DDSSMAdapter` wraps `DDSSM_base` and owns the `DDSSMTrainer`; `CSDIAdapter` wraps the re-vendored CSDI baseline.
- `hparams: ModelConfig | None` — the family's config (single source of truth per ADR-0004); forwarded into `model.fit(...)` where it wins over the adapter's constructor `config`.
- `training: TrainingScalars` — `steps`, `log_every`, `validate_every`, etc.
- `eval`, `viz`, `variance` — `EvalSpec` / `VizSpec` / `ProbeSpec` for the corresponding standalone stages
- `objective: ObjectiveSpec` — reads `metrics.csv`/`metrics.json` and returns the mean tail loss; used as the Optuna objective. If `None`, `train()` returns without computing it.

`Experiment.train`, `evaluate`, `visualize`, `variance_probe` are independent entry methods — eval/viz/variance load a checkpoint and don't trigger training. `train()` delegates fit + checkpointing to `self.model` (the adapter); it no longer constructs a trainer directly.

### Model: `DDSSM_base` (`src/ddssm/model/dssd.py`)
ELBO over latent state-space: encoder `q_ϕ(z|x)`, decoder `p_θ(x|z)`, init prior `p_η(z_{1:j})`, transition `p_ψ(z_t|z_{t-j:t-1})`. The transition is pluggable: a Gaussian transition (`GaussianTransition`, or the baseline-centering `BaselineGaussianTransition`) or a diffusion transition (`DiffusionTransition`, a CSDI U-Net denoiser). `j` is the history window. `forecast(...)` rolls out autoregressively.

The model exposes a `ProbeBatch` payload (encoded latents + log-q paths) reused by the variance probe.

### Pluggable sub-architectures
The CSDI residual stack composes per-channel time and feature mixers (`conv`/`gru`/`identity` × `transformer`/`conv`/`identity`); MLP variants exist as ablations. These are selected in Python when building the model (see `experiments/init_centering/model.py` and `src/ddssm/experiment/builders.py`), not via CLI config groups: `conf/registry.py` defines `encoder`/`decoder`/`transition`/`unet`/… stores, but only the `experiment`, `data`, and `sweep` groups are actually populated with named presets — so `experiment=…` and `+sweep=…` work on the CLI while e.g. `encoder=…` has nothing to select.

### Adapters: `src/ddssm/adapters/`
`ModelAdapter` (`base.py`) is the ABC each model family implements: `module` (raw checkpointable `nn.Module`), `fit`, `forecast`, `save_checkpoint`, `load_checkpoint`, plus an optional shared `log_prob`. `DDSSMAdapter` (`ddssm.py`) wraps `DDSSM_base` and owns the `DDSSMTrainer`; `CSDIAdapter` (`csdi.py`) wraps a re-vendored CSDI copy (`_csdi_vendor/`, kept byte-identical to upstream — deliberately separate from `model/transitions/_csdi_vendor/`). Every family carries a `ModelConfig` subclass (`src/ddssm/model/config.py`) holding all family knobs. Checkpoint payloads are tagged (`ddssm_ckpt_v3` / `csdi_ckpt_v1`) and round-trip across processes; cross-format loads raise `ValueError`. Metric gating uses `MetricNotSupported` (a narrow `NotImplementedError` subclass) — the eval/viz/variance runners catch it to skip family-unsupported metrics while letting real `NotImplementedError`s propagate. See `docs/adr/0011-model-adapters.md`.

### Trainer: `DDSSMTrainer` (`src/ddssm/training/train.py`)
Owns the optimizer per submodule (separate LRs for encoder/decoder/z_init/transition), AMP, CSV/TensorBoard/W&B logging, checkpointing, and the λ-warmup schedule. Used internally by `DDSSMAdapter`. The forward pass always computes every ELBO term. (`_set_trainable(...)`, the former per-submodule `requires_grad` mask, is now dead code — staged training has been removed.)

### Standalone stages (no training)
Each has its own runner + registry (metric/plot dict). All load a checkpoint and read `metrics.csv` from the run dir.
- `src/ddssm/eval/`     — `EvalSpec`, metric registry (MAE, CRPS-sum, recon divergence), writes `metrics.json`. CLI: `python -m ddssm.evaluate`. Wired into `init_smoke_high_surface` and the study cells.
- `src/ddssm/viz/`      — `VizSpec`, plot registry (`forecast_1d`, etc.), writes PNGs. CLI: `python -m ddssm.visualize`.
- `src/ddssm/variance/` — `ProbeSpec`, variance-probe metrics + plots, writes `variance_raw.csv` + `variance_summary.json`. CLI: `python -m ddssm.variance`.

Note: `Experiment.viz` and `Experiment.variance` are currently `None` for every registered preset (only `eval` is wired), so the viz/variance CLIs run only against a preset you've added a spec to.

### Data modules: `src/ddssm/data/`
`TimeSeriesDataModule` is the interface (`train_loader()`, `val_loader()`, `test_loader()`, `batch_transform`). Implementations: `SyntheticDataModule`, `KDDDataModule`, `NullDataModule` (skips `trainer.fit(...)` — used for smoke tests). GluonTS loaders live in `gluonts.py`.

### Sweeps
Sweeper presets: `src/ddssm/conf/hydra/sweeper/ddssm_optuna.yaml` (single-objective) and `ddssm_optuna_moo.yaml` (multi-objective NSGA-II), both using the `dahlem/hydra` fork branch pinned in `pyproject.toml` for Optuna 4.2 support. Search spaces are **Python presets** registered into the `sweep` group (e.g. `experiments/init_centering/sweeps.py`: `init_ablation`, `init_ablation_moo`, `init_ablation_moo_r2`, `init_pilot`; `experiments/csdi/sweeps.py`: `csdi_lean`), activated with `+sweep=<name>` — or arbitrary CLI overrides on `hydra.sweeper.params.*`. Sweeps that target adapter-owned knobs use the `experiment.hparams.*` prefix (all family knobs live on the `ModelConfig`). SLURM submission is rendered via `python -m experiments sbatch` / `src/ddssm/cluster/sbatch.py`, or orchestrated for a whole study by `python -m ddssm.launch`.

## Notes / conventions

- **Hydra `chdir=False`** — `Experiment.train` anchors checkpoints inside `run_dir` so per-run outputs are self-contained.
- **W&B is opt-in** — install `.[wandb]`; the logger silently no-ops if the package is missing. Off by default (`wandb=disabled`); turn on with `wandb=enabled` (the `conf/wandb/` group sets `experiment.wandb_config`). Note `disabled.yaml`/`enabled.yaml` carry `# @package experiment.wandb_config`, so the chosen group always populates that field — a preset that bakes its own `wandb_config` would be overridden by the group default.
- **`hparams` is the single source of truth** — `Experiment.hparams` (a `ModelConfig`) is forwarded into `model.fit(...)`/`load_checkpoint(...)`, where it wins over the adapter's constructor `config` (ADR-0004). So `tweak(exp, hparams__lr=1e-3)` takes effect without descending into the adapter. `Experiment.train` also syncs `hparams.batch_size` onto the data module before fit.
- **Mamba is optional** — `build_mamba.sh` documents the manual build (CUDA arch-specific). Not in `pyproject.toml`.
- **Org files** — `verifications.org`, `esm_vs_dsm.org`, `model-v2.org` are literate-programming sources; `verifications.org` in particular originally tangled out config-group stores. Treat them as documentation, not the source of truth for running code.
