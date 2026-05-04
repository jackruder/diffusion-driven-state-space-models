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
  ddssm.py           # Core model: DDSSM_base (ELBO forward pass)
  train.py           # DDSSMTrainer (fit / checkpoint helpers)
  stages.py          # Multi-stage training orchestration
  encoder.py         # Variational encoder networks
  decoder.py         # Decoder networks
  transitions/       # Transition models (Gaussian + diffusion)
  diffnets.py        # CSDIUnet and related networks
  net_utils.py       # Shared utilities (time embeddings, side info)
  logging.py         # CSV + TensorBoard logging helpers
  eval_utils.py      # Evaluation and visualisation utilities
  data/              # Dataset loaders (GluonTS, PM2.5, KDD, synthetic)
```

## Installation

```bash
pip install -e .
```

Requires Python 3.13 and PyTorch ≥ 2.9.

## Running experiments

### Quick start – single local run

The root `train.py` is a [Hydra](https://hydra.cc) entry point.
All configuration lives under `conf/`.

```bash
# KDD experiment with defaults from conf/config.yaml + conf/dataset/kdd.yaml
python train.py dataset=kdd

# Override training steps and a model hyperparameter
python train.py dataset=kdd steps=5000 hp.vae_lr=3e-4

# Use the large Beijing config instead of the single-station default
python train.py dataset=kdd \
    "model_configs=[configs/kdd_gauss_beijing.yaml]" \
    steps=10000
```

Outputs (checkpoints, TensorBoard logs, model config snapshot) are written to
`runs/<job-name>/<timestamp>/` by default.

### Hydra config structure

```
conf/
  config.yaml            ← top-level defaults; all keys are overrideable
  dataset/
    kdd.yaml             ← KDD data path and sensible seq_len/split defaults
  hydra/launcher/
    submitit_slurm.yaml  ← SLURM resource requests (partition, GPUs, memory …)
  sweep/
    kdd_p1.yaml          ← Optuna phase-1 sweep search space
```

Key config keys:

| Key | Default | Description |
|-----|---------|-------------|
| `model_configs` | `[configs/kdd_gauss_single.yaml]` | Model YAML files merged left-to-right |
| `steps` | 1 000 (5 000 for `dataset=kdd`) | Optimizer steps |
| `seq_len` / `split` | 32 / 16 (120 / 72) | Total window length and encoder horizon |
| `hp.vae_lr` | *(from model config)* | Shorthand LR for encoder, decoder, z-init |
| `hp.trans_lr` | *(from model config)* | Transition model learning rate |
| `hp.batch_size` | *(from model config)* | Batch size override |
| `do_eval` | `false` | Run evaluation after training and return CRPS-sum |

Any key in `hp.*` is applied on top of the loaded model config YAMLs.
Individual keys (`hp.enc_lr`, `hp.dec_lr`, `hp.zinit_lr`) take precedence
over the `hp.vae_lr` shorthand when both are provided.

### Hyperparameter sweep on SLURM

The `conf/sweep/kdd_p1.yaml` config defines a phase-1 Optuna sweep.
Each trial trains for 1 000 steps, evaluates on the validation set, and
reports the CRPS-sum back to the Optuna study.

**Prerequisites**

```bash
pip install -e .   # installs hydra-submitit-launcher and hydra-optuna-sweeper
```

The Optuna study is stored in a SQLite file so that all workers share state.
Place it on a filesystem accessible from all compute nodes (e.g. a shared
network filesystem).

**Launch**

```bash
python train.py --multirun \
    +sweep=kdd_p1 \
    hydra/launcher=submitit_slurm \
    hydra/sweeper=optuna \
    "++hydra.sweeper.storage=sqlite:////shared/fs/runs/optuna/kdd_p1.db" \
    "++hydra.launcher.partition=gpu" \
    "++hydra.launcher.timeout_min=720"
```

Each trial is submitted as an independent SLURM job.  Optuna uses TPE to
suggest the next set of hyperparameters after each completed trial.

**Override SLURM resources at the CLI**

```bash
python train.py --multirun +sweep=kdd_p1 \
    hydra/launcher=submitit_slurm hydra/sweeper=optuna \
    "++hydra.sweeper.storage=sqlite:////shared/fs/runs/optuna/kdd_p1.db" \
    "++hydra.launcher.partition=gpu_long" \
    "++hydra.launcher.mem_gb=64" \
    "++hydra.launcher.timeout_min=1440"
```

**Monitor sweep progress**

```bash
# Live Optuna dashboard
optuna-dashboard sqlite:////shared/fs/runs/optuna/kdd_p1.db

# Or use the CLI
optuna best-trial \
    --storage "sqlite:////shared/fs/runs/optuna/kdd_p1.db" \
    --study-name ddssm_kdd_p1
```

### Legacy scripts

The original argparse-based entry points remain under
`scripts/experiments/kdd/` and accept `--config` / `--set` flags for direct
use without Hydra.  These are suitable for one-off runs or debugging.

```bash
python scripts/experiments/kdd/kdd_train.py \
    --config configs/kdd_gauss_single.yaml \
    --data_path data/kdd_processed.pt \
    --steps 5000
```

## Development

Run tests:

```bash
pytest tests/
```

Format and lint (requires [pre-commit](https://pre-commit.com/)):

```bash
pre-commit run --all-files
```
