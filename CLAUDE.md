# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DDSSM = Diffusion-Driven State Space Models — a PyTorch framework for probabilistic time-series forecasting. It jointly trains a variational encoder/decoder over latent states and a transition prior (Gaussian or CSDI-style diffusion) under an ELBO objective. See `README.md` for the high-level overview and full preset table.

Python 3.13, PyTorch ≥ 2.9. Dependencies are managed with `uv` (see `uv.lock`).

## Commands

```bash
# Install (editable)
pip install -e .                 # or: pip install -e .[wandb]

# Run the unified entry point (default experiment: init_smoke_simple)
python -m ddssm.app                                       # default smoke
python -m ddssm.app experiment=init_smoke_high_surface    # any registered preset
python -m ddssm.app experiment=init_mlp_pinned_per_t__1d experiment.training.stages.n_stage2=2000

# (Shipped presets are multi-stage: the step budget is training.stages.n_pretrain
#  / n_stage2, NOT training.steps — that's only read by the single-fit path.)

# List / run / render-sbatch for any registered preset
python -m experiments list                                # enumerate preset names
python -m experiments run  init_smoke_simple training.stages.n_pretrain=200
python -m experiments sbatch --out=job.sh init_mlp_pinned_per_t__mv

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

`src/ddssm/` is organized into subpackages: `nn/` (reusable neural building blocks), `model/` (the SSM: `dssd`, encoder/decoder, losses, and the `transitions/`/`centering/`/`likelihood/` subpackages), `training/` (trainer, stages, checkpoint, loggers), `experiment/` (the `Experiment` composition root + `builders`/`stores`/`registry`), `cluster/` (sbatch/study/report orchestration), plus the standalone-stage packages `eval/`/`viz/`/`variance/`, `data/`, and `conf/`. The `python -m ddssm.<x>` entry points (`app`, `evaluate`, `visualize`, `launch`, `colocate`, `variance`, `launch_remaining`) stay at the top level. Intra-package imports are absolute (`from ddssm.<pkg>.<mod> import …`).

### Entry point: `src/ddssm/app.py`
`@hydra.main` reads `src/ddssm/conf/config.yaml`. On startup it calls `register_experiments()` (`experiment/registry.py`), which:

1. Adds the repo root to `sys.path` so `import experiments` works.
2. Imports the `experiments/` package — its submodules (currently the single `experiments/init_centering/` family) register named presets into the hydra-zen `store` singleton at import time.
3. Calls `store.add_to_hydra_store(overwrite_ok=True)` to publish everything to Hydra's `ConfigStore`.

Run `python -m experiments list` to enumerate every registered preset.

Then `instantiate(cfg.experiment)` builds an `Experiment` and `experiment.train(device, run_dir)` runs it. The Hydra run directory holds `metrics.csv`, `tb_logs/`, `checkpoints/`, and `resolved_config.yaml`.

### Two parallel config worlds — important

- `src/ddssm/conf/` is a **small library of reusable Hydra YAML defaults**: top-level `config.yaml`, `hydra/sweeper/` (`ddssm_optuna.yaml`, `ddssm_optuna_moo.yaml`), and `wandb/`. It does NOT contain the experiment presets themselves.
- `experiments/` (at the repo root) is where named presets are defined **in Python** via hydra-zen `builds(...)`. The repo currently ships one family, `experiments/init_centering/`. Its modules: `datasets.py`/`data.py` (data axis), `model.py` (the `SmokeModel` factory that composes encoder/decoder/baseline/transitions from runtime classes), `hparams.py` (multi-stage config), `cells.py` + `study.py` (the ablation grid + study), `experiments.py` (registers the two smoke presets and the study-generated cells via `experiments._make.experiment(...)` + `experiment_store(...)`), `evals.py`, `sweeps.py`, and `report.py`. The shared `experiments/_make.py` is the `experiment(...)` factory; `experiments/datasets.py` registers library dataset presets into `data_store`; `experiments/_cli.py` backs `python -m experiments`.

If you need a new experiment, **add a Python file under `experiments/<family>/`** and register it — don't add YAML.

`src/ddssm/experiment/builders.py` is a convenience surface of hydra-zen `builds(...)` configs (models, encoders, transitions, etc.) for assembling an `Experiment` ad hoc — e.g. in a notebook or script. Note the live `init_centering` family does **not** route through it; `model.py` composes the model from runtime classes directly. Of its exports only a handful (`Synthetic`, `Eval`, `Objective`(s), `Hparams`, `Training`, `CenteringHandoff`, `ExperimentC`, `TrainerPartial`) are imported elsewhere in the repo.

### Experiment object
`src/ddssm/experiment/experiment.py` defines `Experiment`, a dataclass that owns
(re-exported as `ddssm.experiment.Experiment`):
- `data: DDSSMDataModule` — train/val/test loaders + batch transform
- `model: DDSSM_base` — the variational SSM (`src/ddssm/model/dssd.py`)
- `build_trainer: Callable[..., DDSSMTrainer]` — partial trainer factory
- `training: TrainingScalars` — `steps`, `log_every`, `validate_every`, `trainable` (per-module `requires_grad` mask), etc.
- `eval`, `viz`, `variance` — `EvalSpec` / `VizSpec` / `ProbeSpec` for the corresponding standalone stages
- `objective: ObjectiveSpec` — reads `metrics.csv` and returns the mean tail loss; used as the Optuna objective. If `None`, `train()` returns the trainer instead.

`Experiment.train`, `evaluate`, `visualize`, `variance_probe` are independent entry methods — eval/viz/variance load a checkpoint and don't trigger training.

### Model: `DDSSM_base` (`src/ddssm/model/dssd.py`)
ELBO over latent state-space: encoder `q_ϕ(z|x)`, decoder `p_θ(x|z)`, init prior `p_η(z_{1:j})`, transition `p_ψ(z_t|z_{t-j:t-1})`. The transition is pluggable: a Gaussian transition (`GaussianTransition`, or the baseline-centering `BaselineGaussianTransition`) or a diffusion transition (`DiffusionTransition`, a CSDI U-Net denoiser). `j` is the history window. `forecast(...)` rolls out autoregressively.

The model exposes a `ProbeBatch` payload (encoded latents + log-q paths) reused by the variance probe.

### Pluggable sub-architectures
The CSDI residual stack composes per-channel time and feature mixers (`conv`/`gru`/`identity` × `transformer`/`conv`/`identity`); MLP variants exist as ablations. These are selected in Python when building the model (see `experiments/init_centering/model.py` and `src/ddssm/experiment/builders.py`), not via CLI config groups: `conf/registry.py` defines `encoder`/`decoder`/`transition`/`unet`/… stores, but only the `experiment`, `data`, and `sweep` groups are actually populated with named presets — so `experiment=…` and `+sweep=…` work on the CLI while e.g. `encoder=…` has nothing to select.

### Trainer: `DDSSMTrainer` (`src/ddssm/training/train.py`)
Owns the optimizer per submodule (separate LRs for encoder/decoder/z_init/transition), AMP, CSV/TensorBoard/W&B logging, checkpointing, and the λ-warmup schedule. `_set_trainable(...)` toggles `requires_grad` per submodule — that mask is the single mechanism for stage-aware gradient suppression. The forward pass always computes every ELBO term; frozen submodules just don't accumulate gradients.

### Multi-stage training: `src/ddssm/training/stages.py`
`StageOrchestrator` runs sequential phases (e.g. recon-only → trans-only → joint) with per-stage trainable masks, LRs, scheduler, and λ-ramp. Configured via `StagesConf`.

### Standalone stages (no training)
Each has its own runner + registry (metric/plot dict). All load a checkpoint and read `metrics.csv` from the run dir.
- `src/ddssm/eval/`     — `EvalSpec`, metric registry (MAE, CRPS-sum, recon divergence), writes `metrics.json`. CLI: `python -m ddssm.evaluate`. Wired into `init_smoke_high_surface` and the study cells.
- `src/ddssm/viz/`      — `VizSpec`, plot registry (`forecast_1d`, etc.), writes PNGs. CLI: `python -m ddssm.visualize`.
- `src/ddssm/variance/` — `ProbeSpec`, variance-probe metrics + plots, writes `variance_raw.csv` + `variance_summary.json`. CLI: `python -m ddssm.variance`.

Note: `Experiment.viz` and `Experiment.variance` are currently `None` for every registered preset (only `eval` is wired), so the viz/variance CLIs run only against a preset you've added a spec to.

### Data modules: `src/ddssm/data/`
`DDSSMDataModule` is the interface (`train_loader()`, `val_loader()`, `test_loader()`, `batch_transform`). Implementations: `SyntheticDataModule`, `KDDDataModule`, `NullDataModule` (skips `trainer.fit(...)` — used for smoke tests). GluonTS loaders live in `gluonts.py`.

### Sweeps
Sweeper presets: `src/ddssm/conf/hydra/sweeper/ddssm_optuna.yaml` (single-objective) and `ddssm_optuna_moo.yaml` (multi-objective NSGA-II), both using the `dahlem/hydra` fork branch pinned in `pyproject.toml` for Optuna 4.2 support. Search spaces are **Python presets** registered into the `sweep` group (`experiments/init_centering/sweeps.py`: `init_ablation`, `init_ablation_moo`, `init_ablation_moo_r2`, plus the `init_pilot` alias), activated with `+sweep=<name>` — or arbitrary CLI overrides on `hydra.sweeper.params.*`. SLURM submission is rendered via `python -m experiments sbatch` / `src/ddssm/cluster/sbatch.py`, or orchestrated for a whole study by `python -m ddssm.launch`.

## Notes / conventions

- **Hydra `chdir=False`** — `Experiment.train` anchors checkpoints inside `run_dir` so per-run outputs are self-contained.
- **W&B is opt-in** — install `.[wandb]`; the logger silently no-ops if the package is missing. Off by default (`wandb=disabled`); turn on with `wandb=enabled` (the `conf/wandb/` group sets `experiment.wandb_config`). Note `disabled.yaml`/`enabled.yaml` carry `# @package experiment.wandb_config`, so the chosen group always populates that field — a preset that bakes its own `wandb_config` would be overridden by the group default.
- **`hparams` is shared** — `Experiment.hparams` and `Experiment.model.config.hyperparams` are kept identical so `tweak(exp, hparams__lr=1e-3)` works without descending into `model.config`. `Experiment.train` re-syncs them defensively.
- **Mamba is optional** — `build_mamba.sh` documents the manual build (CUDA arch-specific). Not in `pyproject.toml`.
- **Org files** — `verifications.org`, `esm_vs_dsm.org`, `model-v2.org` are literate-programming sources; `verifications.org` in particular originally tangled out config-group stores. Treat them as documentation, not the source of truth for running code.
