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
    synthetic.yaml       ← in-memory synthetic data; no data file needed
  hydra/launcher/
    submitit_slurm.yaml  ← SLURM resource requests (partition, GPUs, memory …)
  sweep/
    kdd_p1.yaml          ← Optuna phase-1 sweep for KDD
    synthetic_p1.yaml    ← Optuna phase-1 sweep for synthetic data (fast, CPU-ok)
```

Key config keys:

| Key | Default | Description |
|-----|---------|-------------|
| `model_configs` | `[configs/kdd_gauss_single.yaml]` | Model YAML files merged left-to-right |
| `steps` | 1,000 (5,000 for `dataset=kdd`) | Optimizer steps |
| `seq_len` / `split` | 32 / 16 (120 / 72) | Total window length and encoder horizon |
| `hp.vae_lr` | *(from model config)* | Shorthand LR for encoder, decoder, z-init |
| `hp.trans_lr` | *(from model config)* | Transition model learning rate |
| `hp.batch_size` | *(from model config)* | Batch size override |
| `do_eval` | `false` | Run evaluation after training and return CRPS-sum |

Any key in `hp.*` is applied on top of the loaded model config YAMLs.
Individual keys (`hp.enc_lr`, `hp.dec_lr`, `hp.zinit_lr`) take precedence
over the `hp.vae_lr` shorthand when both are provided (a warning is emitted
when both are set simultaneously).

**Adding a new hyperparameter** only requires adding it to the `hp:` block in
`conf/config.yaml` (and any sweep YAMLs).  No change to `train.py` is needed
— all `hp.*` keys are automatically forwarded to `DDSSMConfig.hyperparams.*`
unless they have a special mapping (currently `S_k` and `k_chunk`).

#### Architecture overrides

Use `arch.*` to override individual DDSSMConfig fields directly from the CLI or
sweep configs, without creating a separate model YAML file:

```bash
# Change encoder hidden dim and latent dim in a one-off run
python train.py dataset=kdd arch.encoder.hidden_dim=128 arch.latent_dim=4

# Sweep over architecture choices with Optuna
python train.py --multirun dataset=kdd \
    +sweep=kdd_p1 hydra/sweeper=optuna \
    "++hydra.sweeper.params.arch.encoder.hidden_dim=choice(64,128)" \
    "++hydra.sweeper.params.arch.latent_dim=choice(2,4,8)"
```

Keys must use the full DDSSMConfig dot-path (e.g. `encoder.hidden_dim`,
`transition.schedule.num_steps`).

### Synthetic data — getting started immediately

No data download needed.  Sequences are generated in-memory by
`SyntheticDataset` covering eight modes (`iid`, `lgssm`, `nonlinear`,
`harmonic`, `bimodal`, `bimodal-block`, …).

**Option A — `verifications.py` (standalone argparse, quickest)**

Trains on synthetic data and produces a multi-panel forecast plot.

```bash
# 500-step joint run on LGSSM data, Gaussian transition
python scripts/experiments/verifications.py \
    --config configs/synthetic_gauss.yaml \
    --mode lgssm --training_mode joint --steps 500

# Harder bimodal task
python scripts/experiments/verifications.py \
    --config configs/synthetic_gauss.yaml \
    --mode bimodal --training_mode joint --steps 1000

# Recon-only stage to validate encoder/decoder before training transition
python scripts/experiments/verifications.py \
    --config configs/synthetic_gauss.yaml \
    --mode harmonic --training_mode recon_only --steps 300
```

Output goes to `runs/verify/<mode>/<timestamp>/` and includes a checkpoint
and a `verify_<mode>_<stage>.png` forecast plot.

**Option B — Hydra `train.py` (reproducible, sweepable)**

```bash
# Quick smoke-test (500 steps, D=1 LGSSM, CPU-friendly)
python train.py dataset=synthetic

# Switch to bimodal mode and run longer
python train.py dataset=synthetic dataset.mode=bimodal steps=1000

# Evaluate after training (returns CRPS-sum)
python train.py dataset=synthetic dataset.mode=harmonic steps=500 do_eval=true

# Local Optuna sweep over 20 trials (~10 min on CPU)
python train.py --multirun \
    dataset=synthetic dataset.mode=lgssm \
    +sweep=synthetic_p1 \
    hydra/sweeper=optuna \
    "++hydra.sweeper.storage=sqlite:///runs/optuna/synthetic_p1.db"
```

TensorBoard logs land at `runs/<job-name>/<timestamp>/tb_logs/`; view with:

```bash
tensorboard --logdir runs/
```

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

### Logging and monitoring

Metrics are logged to **three** sinks (TensorBoard and CSV are always on;
W&B is opt-in):

| Sink | Path / target | Enable with |
|------|---------------|-------------|
| **TensorBoard** | `<run_dir>/tb_logs/` | always on — `tensorboard --logdir runs/` |
| **CSV** | `<run_dir>/csv_logs/train_metrics.csv` | always on |
| **W&B** | your W&B project | see below |

#### Enabling W&B logging

**`train.py` (Hydra)**

```bash
# Cloud W&B
python train.py dataset=synthetic wandb.enabled=true wandb.project=ddssm

# Self-hosted server
python train.py dataset=synthetic \
    wandb.enabled=true \
    wandb.project=ddssm \
    wandb.base_url=https://wandb.example.com

# With entity and explicit run name
python train.py dataset=kdd \
    wandb.enabled=true wandb.project=ddssm \
    wandb.entity=my-team wandb.name=kdd-baseline-run1
```

**Run name auto-linking:** when `wandb.name` is not set, the run name is
automatically derived from the Hydra override dirname (e.g.
`hp.vae_lr=3e-4,hp.trans_lr=1e-4`), which makes it trivial to find the
matching Hydra output directory from a W&B run.

**Sweep grouping:** when running an Optuna sweep, all trial runs are
automatically placed in a W&B group named after the Optuna study (e.g.
`ddssm_kdd_p1`).  You can also set `wandb.group` explicitly:

```bash
python train.py --multirun +sweep=kdd_p1 wandb.enabled=true wandb.group=my-sweep
```

**Architecture in run config:** the `model_configs` list is stored in every
W&B run's config so runs with different architectures are distinguishable in
the UI without needing manual tags.

**`verifications.py` (argparse)**

```bash
# Minimal: just supply --wandb_project to activate W&B
python scripts/experiments/verifications.py \
    --config configs/synthetic_gauss.yaml \
    --mode lgssm \
    --wandb_project ddssm

# Self-hosted server
python scripts/experiments/verifications.py \
    --config configs/synthetic_gauss.yaml \
    --mode bimodal \
    --wandb_project ddssm \
    --wandb_base_url https://wandb.example.com
```

#### Self-hosted W&B server

Set **one** of these — they are equivalent:

```bash
# Option A: environment variable (applies process-wide)
export WANDB_BASE_URL=https://wandb.example.com

# Option B: CLI flag (Hydra)
python train.py wandb.enabled=true wandb.base_url=https://wandb.example.com

# Option C: CLI flag (verifications.py)
python scripts/experiments/verifications.py --wandb_base_url https://wandb.example.com
```

W&B is a soft dependency — if `wandb` is not installed the logger silently
becomes a no-op and training continues with TensorBoard + CSV only.

Run tests:

```bash
pytest tests/
```

Format and lint (requires [pre-commit](https://pre-commit.com/)):

```bash
pre-commit run --all-files
```
