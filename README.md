# Diffusion-Driven State Space Models (DDSSM)

A PyTorch framework for probabilistic time-series forecasting that combines
**variational state space models** with **diffusion-based transition priors**.

## Overview

DDSSM learns a latent state-space representation of multivariate time series and
uses a diffusion model (CSDI-style U-Net) as the transition prior between latent
states. An ELBO objective jointly trains:

- **Encoder** — maps observed windows to Gaussian latent distributions
- **Decoder** — reconstructs observations from latent states
- **Transition model** — predicts the next latent state, either Gaussian or
  diffusion-based

## Package structure

```
src/ddssm/
  config.py          # Pydantic config models (DDSSMConfig, …)
  dssd.py            # Core model: DDSSM_base (ELBO forward pass)
  experiment.py      # Experiment composition root (data + model + trainer)
  train.py           # DDSSMTrainer (fit / checkpoint helpers)
  stages.py          # Multi-stage training orchestration
  encoder.py         # Variational encoder networks
  decoder.py         # Decoder networks
  transitions/       # Transition models (Gaussian + diffusion)
  diffnets.py        # CSDIUnet and related networks
  net_utils.py       # Shared utilities (time embeddings, side info)
  loggers.py         # CSV + TensorBoard + W&B logging helpers
  eval_utils.py      # Visualisation utilities
  eval_metrics.py    # MAE / CRPS-sum metrics + recon-divergence detection
  eval/              # Hydra evaluation stage (runner + metric registry)
  viz/               # Hydra visualisation stage (runner + plot registry)
  variance/          # Hydra variance probe stage (runner + metric/plot registries)
  data/              # Dataset loaders (GluonTS, PM2.5, KDD, synthetic)
```

## Installation

```bash
pip install -e .
```

Requires Python 3.13 and PyTorch ≥ 2.9.

## Running experiments

There are two complementary entry points:

1. **`python -m ddssm.app`** — the Hydra-native, composable entry point (use
   for new experiments and Optuna sweeps; described below).
2. **`scripts/experiments/...`** — legacy YAML-driven scripts (still used for
   workflows that need the multi-stage Pydantic ``stages`` orchestrator).

Legacy YAML invocation example:

```bash
python scripts/experiments/kdd/kdd_train.py \
    --data_path data/kdd.pt \
    --config configs/base.yaml \
    --set hyperparams.batch_size=32
```

### Hydra experiment presets

Reusable experiment presets are registered in `src/ddssm/conf/experiments/`.
Each preset selects a transition (`gaussian`/`diffusion`), a dataset,
root-level dimensions, hyperparameters, and training scalars. Activate one
with `experiment=NAME`:

| Preset                          | Dataset    | Transition | Notes                                              |
| ------------------------------- | ---------- | ---------- | -------------------------------------------------- |
| `synthetic_gauss`               | synthetic  | gaussian   | LGSSM, runs end-to-end via `ddssm.app`             |
| `synthetic_diffusion`           | synthetic  | diffusion  | LGSSM, runs end-to-end via `ddssm.app`             |
| `kdd_gauss`                     | kdd        | gaussian   | Model recipe; pair with `kdd_train.py` for data    |
| `kdd_diffusion`                 | kdd        | diffusion  | Model recipe; pair with `kdd_train.py` for data    |

```bash
# Single end-to-end run on synthetic data
python -m ddssm.app experiment=synthetic_gauss

# Override anything the experiment sets
python -m ddssm.app experiment=synthetic_diffusion \
    experiment.training.steps=2000 experiment.hyperparams.batch_size=64
```

#### Variance probe workflow (train, then probe)

Variance probe presets are optimized for quick diagnostics and write checkpoints
to stable per-preset directories under `runs/variance_probe/...`.

| Preset                                 | Purpose |
| -------------------------------------- | ------- |
| `variance_probe_lgssm`                 | Baseline linear-Gaussian case for sanity-checking variance trends. |
| `variance_probe_bimodal_clean`         | Multimodal target without observation noise; tests mode handling only. |
| `variance_probe_bimodal_noisy`         | Same multimodal structure with added noise; tests robustness. |
| `variance_probe_nonlinear_bimodal_lift`| Higher-dimensional nonlinear stress case for the probe metrics. |

```bash
# 1) Train one preset and produce a stable checkpoint
python -m ddssm.app experiment=variance_probe_lgssm +sweep=variance_probe

# 2) Run offline variance analysis from the trained checkpoint
python -m ddssm.variance \
    experiment=variance_probe_lgssm \
    checkpoint='${experiment.checkpoint_dir}/ckpt_latest.pth' \
    +sweep=variance_probe
```

The variance stage writes:

- `variance_raw.csv`: per-replica/per-seed probe rows
- `variance_summary.json`: aggregate metrics and metadata
- plot files (defaults): `var_grad_vs_tau.png`, `var_loss_vs_tau.png`,
  `ratio_vs_tau.png`, `summary_table.png`

When `cfg.experiment.data` is a `NullDataModule`, `ddssm.app` builds the model and
trainer but skips `trainer.fit(...)`. Use this for smoke tests.

### Hydra + Optuna sweeps

Hydra-based sweeps use Optuna through the `hydra-optuna-sweeper` plugin pinned
in `pyproject.toml`. This intentionally tracks the requested `dahlem/hydra`
fork branch until an equivalent tagged or official release is available.
The repo provides a reusable sweeper preset at
`src/ddssm/conf/hydra/sweeper/ddssm_optuna.yaml` plus pre-defined
search-space presets in `src/ddssm/conf/sweep/`:

| Sweep preset      | Pairs with               | Search space                                        |
| ----------------- | ------------------------ | --------------------------------------------------- |
| `synthetic_lr`    | `synthetic_*` experiments | enc/dec/zinit/trans LR, λ-warmup, λ-end, batch size |
| `kdd_phase1`      | `kdd_*` experiments       | LRs (capped), λ schedule, weight decay, batch size  |

Each sweep preset re-activates the Optuna sweeper, so a multirun is just:

```bash
python -m ddssm.app --multirun \
    experiment=synthetic_gauss \
    +sweep=synthetic_lr \
    hydra.sweeper.n_trials=20 \
    hydra.sweeper.study_name=ddssm_synth_lr \
    hydra.sweeper.storage=sqlite:///ddssm_synth_lr.db
```

`ddssm.app` returns the mean tail of `loss/total` from the run's `metrics.csv`
as the Optuna objective. Override `experiment.training.return_objective=false` if you
want the trainer object back instead. Failed trials surface as `+inf`, which
Optuna's `minimize` direction handles cleanly.

Ad-hoc search spaces can still be defined directly on the CLI without a
preset:

```bash
python -m ddssm.app --multirun \
    hydra/sweeper=ddssm_optuna \
    experiment=synthetic_gauss \
    hydra.sweeper.n_trials=50 \
    hydra.sweeper.study_name=ddssm_example \
    hydra.sweeper.storage=sqlite:///ddssm_example.db \
    'hydra.sweeper.params.experiment.hyperparams.enc_lr=interval(1e-5,1e-3)' \
    'hydra.sweeper.params.experiment.hyperparams.batch_size=choice(32,64,128)'
```

Relative SQLite storage URLs are resolved from Hydra's runtime working
directory. Use an absolute `sqlite:///...` path for shared studies or CI.

The checked-in `src/ddssm/conf/` tree is intentionally a small library of
reusable Hydra defaults; large or experiment-specific search spaces should
live either as additional `src/ddssm/conf/sweep/*` presets or as external
assets / CLI overrides.

## Logging

Metrics are written to TensorBoard and CSV by default; **W&B is opt-in**.

```bash
pip install -e .[wandb]
```

Pass a ``wandb_config`` dict to ``DDSSMTrainer`` to activate it, or use the
``--wandb_project`` flag on the argparse-based ``verifications.py`` script:

```bash
# Cloud W&B
python scripts/experiments/verifications.py \
    --config configs/synthetic_gauss.yaml \
    --mode lgssm \
    --wandb_project ddssm

# Self-hosted W&B server
python scripts/experiments/verifications.py \
    --config configs/synthetic_gauss.yaml \
    --mode bimodal \
    --wandb_project ddssm \
    --wandb_base_url https://wandb.example.com
```

W&B is a *soft dependency*: if the ``wandb`` package isn't installed the
logger silently no-ops and training continues with TensorBoard + CSV.

## SLURM

A ready-to-use submitit launcher config lives at
``src/ddssm/conf/hydra/launcher/submitit_slurm.yaml``. Combine it with any experiment
preset and (optionally) a sweep preset to launch a multirun on a Slurm
cluster:

```bash
python -m ddssm.app --multirun \
    hydra/launcher=submitit_slurm \
    experiment=synthetic_diffusion \
    +sweep=synthetic_lr \
    hydra.sweeper.n_trials=64 \
    hydra.sweeper.study_name=ddssm_synth_diff \
    hydra.sweeper.storage=sqlite:///$PWD/ddssm_synth_diff.db
```

Resource overrides (``partition``, ``gpus_per_node``, ``timeout_min``, …) are
plain CLI flags, e.g. ``hydra.launcher.timeout_min=240``.

## Development

Run tests:

```bash
pytest tests/
```

Format and lint (requires [pre-commit](https://pre-commit.com/)):

```bash
pre-commit run --all-files
```
