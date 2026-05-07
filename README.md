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
  logging.py         # CSV + TensorBoard + W&B logging helpers
  eval_utils.py      # Visualisation utilities
  eval_metrics.py    # MAE / CRPS-sum metrics + recon-divergence detection
  data/              # Dataset loaders (GluonTS, PM2.5, KDD, synthetic)
```

## Installation

```bash
pip install -e .
```

Requires Python 3.13 and PyTorch ≥ 2.9.

## Running experiments

Experiment scripts live under `scripts/experiments/`.  Training entry points
pass YAML configs and optional dot-notation overrides:

```bash
python scripts/experiments/kdd/kdd_train.py \
    --config configs/base.yaml \
    --override hyperparams.batch_size=32
```

### Hydra + Optuna sweeps

Hydra-based sweeps use Optuna through the pinned plugin dependency
`hydra-optuna-sweeper @ git+https://github.com/dahlem/hydra.git@feature/upgrade-optuna-4.2.1#subdirectory=plugins/hydra_optuna_sweeper`.
The repo provides one reusable sweeper preset at
`conf/hydra/sweeper/ddssm_optuna.yaml`:

```bash
python -m ddssm.app --multirun \
    hydra/sweeper=ddssm_optuna \
    hydra.sweeper.study_name=ddssm_example \
    hydra.sweeper.storage=sqlite:///ddssm_example.db \
    hydra.sweeper.n_trials=50 \
    'hydra.sweeper.params.hyperparams.enc_lr=interval(1e-5,1e-3)' \
    'hydra.sweeper.params.hyperparams.batch_size=choice(32,64,128)'
```

Keep concrete study definitions and large search-space lists outside `conf/`
(for example under `experiments/` or in command files). The checked-in
`conf/` tree should remain a small library of reusable Hydra defaults, while
Optuna study parameters can be supplied from external experiment assets and CLI
overrides.

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

## SLURM (preview)

A ready-to-use submitit launcher config lives at
``conf/hydra/launcher/submitit_slurm.yaml``. It will be wired up to a real
entry point in the upcoming `hydra-zen` migration; the YAML is checked in
now so resource-request defaults can be reviewed alongside this PR.

## Development

Run tests:

```bash
pytest tests/
```

Format and lint (requires [pre-commit](https://pre-commit.com/)):

```bash
pre-commit run --all-files
```
