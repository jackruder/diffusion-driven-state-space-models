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
  app.py             # main training entry point;     CLI: python -m ddssm.app
  evaluate.py        # eval stage entry;              CLI: python -m ddssm.evaluate
  visualize.py       # viz stage entry;               CLI: python -m ddssm.visualize
  launch.py          # study orchestrator;            CLI: python -m ddssm.launch <study>
  colocate.py        # multi-cell GPU packing;        CLI: python -m ddssm.colocate
  launch_remaining.py# remaining-trial-budget helper (used by generated sbatch)

  nn/                # reusable neural building blocks
    net_utils.py     #   shared utilities (time embeddings, side info)
    diffnets.py      #   CSDIUnet and related networks
    gaussians.py     #   Gaussian distribution math
    # aggregators / combiners / fusions / futsum / dist_heads / aux_posterior / torch_compile
  model/             # the variational state-space model
    dssd.py          #   DDSSM_base (ELBO forward pass)
    encoder.py       #   variational encoder networks
    decoder.py       #   decoder networks
    losses.py        #   ELBO loss terms
    transitions/     #   transition models (Gaussian, baseline-Gaussian, diffusion)
    centering/       #   baseline (μ_p) heads + stage-1 → stage-2 centering handoff
    likelihood/      #   IWAE / prob-flow / VHP likelihood estimators
  training/          # training stack
    train.py         #   DDSSMTrainer (fit / checkpoint helpers)
    stages.py        #   multi-stage training orchestration (StageOrchestrator)
    # train_utils / checkpoint / loggers
  experiment/        # composition layer
    experiment.py    #   Experiment composition root (data + model + trainer)
    builders.py      #   convenience hydra-zen builds() for ad-hoc assembly
    stores.py        #   hydra-zen store helpers
    registry.py      #   register_experiments() (imports the experiments/ package)
  cluster/           # SLURM/study orchestration internals (sbatch.py, study.py, report.py)
  eval/              # evaluation stage (runner + metric registry, incl. eval_metrics.py)
  viz/               # visualisation stage (runner + plot registry)
  variance/          # variance-probe stage;          CLI: python -m ddssm.variance
  data/              # dataset loaders (GluonTS, KDD, synthetic)
  conf/              # reusable Hydra defaults (config.yaml, hydra/sweeper/, wandb/)

experiments/         # named presets defined in Python (registered at import time)
  _make.py           # experiment(...) factory + run()/override() helpers
  _cli.py            # backs `python -m experiments list|run|sbatch`
  datasets.py        # library dataset presets registered into data_store
  init_centering/    # the shipped experiment family (baseline-centering ablation)
```

## Installation

```bash
pip install -e .
```

Requires Python 3.13 and PyTorch ≥ 2.9.

## Running experiments

Two equivalent entry points:

- `python -m ddssm.app experiment=<name>` — the Hydra-native entry; composes
  the named experiment from the hydra-zen `ConfigStore` and runs it.
- `python -m experiments <subcommand>` — a thin wrapper around the same
  registry. Useful for listing, running ad-hoc, or emitting Slurm scripts:

  ```bash
  python -m experiments list                 # all registered experiments
  python -m experiments run    <name> [hydra overrides...]
  python -m experiments sbatch <name> [resource flags...] [hydra overrides...]
  ```

  `sbatch` writes a one-job Slurm submit script (`#SBATCH --partition`,
  `--time`, `--gres=gpu:N`, …) that launches `python -m ddssm.app
  experiment=<name>`. Resources come from the experiment's `SBatch` field if
  set, falling back to `src/ddssm/cluster/sbatch.py:DEFAULT_SBATCH`. CLI flags
  (`--partition=...`, `--time=...`, `--mem=...`, `--gpus=...`, etc.) take the
  final say; they must come **before** `<name>`. Hydra-style overrides after
  `<name>` are baked into the generated script.

### Per-family layout

Presets are defined in Python under `experiments/<family>/` and registered
with the hydra-zen `ConfigStore` at import time. The shipped family is
`experiments/init_centering/`:

| File | Contents |
|------|----------|
| `model.py` | `SmokeModel(...)` factory — composes encoder/decoder/baseline (μ_p)/transitions from runtime classes, parametric over the three ablation axes. |
| `datasets.py` / `data.py` | The dataset axis (`1d`, `mv`) and library data-module re-exports. |
| `hparams.py` | Hyperparameters + the multi-stage (`StagesConf`) builder. |
| `cells.py` / `study.py` | The ablation-grid cell enumeration and the registered study. |
| `evals.py` | `Eval`, `Objective`, and multi-objective specs. |
| `experiments.py` | The two smoke presets + `study.register(...)` for the grid cells. |
| `sweeps.py` | Optuna search-space presets registered into the `sweep` group. |

To add a new experiment, add a Python file under `experiments/<family>/` and
register it — don't add YAML.

### Experiment presets

List everything registered:

```bash
python -m experiments list
```

The shipped presets are the `init_centering` family: two smoke configs
(`init_smoke_simple`, `init_smoke_high_surface`) plus an ablation grid named
`init_<form>_<mode>_<tracking>__<dataset>`, where

- **form** ∈ `zero`, `identity`, `linear`, `mlp` (the baseline μ_p head)
- **mode** ∈ `pinned`, `learnable` (auto-pinned for the param-free `zero`/`identity` forms)
- **tracking** ∈ `fixed`, `per_t`
- **dataset** ∈ `1d` (nonlinear-bimodal-lift, D=1), `mv` (D=8)

e.g. `init_mlp_pinned_per_t__1d`, `init_zero_pinned_fixed__mv`.

```bash
# Default smoke run
python -m ddssm.app

# Pick a preset and override anything it sets. The shipped presets are
# multi-stage, so the step budget is training.stages.n_pretrain / n_stage2
# (training.steps is only read by the single-fit path); batch size is on hparams.
python -m ddssm.app experiment=init_mlp_pinned_per_t__1d \
    experiment.training.stages.n_stage2=2000 experiment.hparams.batch_size=64
```

### Architecture selection

DDSSM has pluggable sub-architectures — the CSDI residual stack composes
per-channel **time mixers** (`conv`/`gru`/`identity`) and **feature mixers**
(`transformer`/`conv`/`identity`), with MLP context/U-Net variants as
ablations. These are chosen **in Python** when the model is built (see
`experiments/init_centering/model.py` and the `builds(...)` configs in
`src/ddssm/experiment/builders.py`) rather than via CLI config groups: `conf/registry.py`
declares `encoder`/`decoder`/`transition`/`unet`/… stores, but only the
`experiment`, `data`, and `sweep` groups ship populated, so `experiment=…` and
`+sweep=…` are the live CLI selectors.

#### Standalone post-training stages

The eval / viz / variance stages each load a checkpoint and read `metrics.csv`
from the run dir; they don't train:

```bash
python -m ddssm.evaluate  experiment=<name> checkpoint='${experiment.checkpoint_dir}/ckpt_latest.pth'
python -m ddssm.visualize experiment=<name> checkpoint=...
python -m ddssm.variance  experiment=<name> checkpoint=...
```

The variance stage writes `variance_raw.csv` (per-replica/per-seed probe rows),
`variance_summary.json` (aggregate metrics), and plot files
(`var_grad_vs_tau.png`, `var_loss_vs_tau.png`, `ratio_vs_tau.png`,
`summary_table.png`). Note that `Experiment.viz` and `Experiment.variance` are
`None` for every shipped preset (only `eval` is wired), so the viz/variance
CLIs are only useful against a preset you've added a spec to.

When `cfg.experiment.data` is a `NullDataModule`, `ddssm.app` builds the model and
trainer but skips `trainer.fit(...)`. Use this for smoke tests.

### Hydra + Optuna sweeps

Hydra-based sweeps use Optuna through the `hydra-optuna-sweeper` plugin pinned
in `pyproject.toml`. This intentionally tracks the requested `dahlem/hydra`
fork branch until an equivalent tagged or official release is available.
The repo provides two reusable sweeper presets under
`src/ddssm/conf/hydra/sweeper/` — `ddssm_optuna.yaml` (single-objective) and
`ddssm_optuna_moo.yaml` (multi-objective NSGA-II). Search-space presets are
defined **in Python** (`experiments/init_centering/sweeps.py`) and registered
into the `sweep` group:

| Sweep preset           | Sweeper          | Search space                                              |
| ---------------------- | ---------------- | -------------------------------------------------------- |
| `init_ablation`        | single-objective | baseline LRs, λ schedule, anchor strengths, batch size   |
| `init_ablation_moo`    | NSGA-II MOO      | same space, multi-objective                               |
| `init_ablation_moo_r2` | NSGA-II MOO      | narrowed round-2 space                                    |
| `init_pilot`           | single-objective | back-compat alias for `init_ablation`                    |

Each sweep preset re-activates the Optuna sweeper, so a multirun is just:

```bash
python -m ddssm.app --multirun \
    experiment=init_smoke_high_surface \
    +sweep=init_ablation \
    hydra.sweeper.n_trials=20 \
    hydra.sweeper.study_name=init_ablation \
    hydra.sweeper.storage=sqlite:///init_ablation.db
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
    experiment=init_smoke_high_surface \
    hydra.sweeper.n_trials=50 \
    hydra.sweeper.study_name=ddssm_example \
    hydra.sweeper.storage=sqlite:///ddssm_example.db \
    'hydra.sweeper.params.experiment.hyperparams.enc_lr=interval(1e-5,1e-3)' \
    'hydra.sweeper.params.experiment.hyperparams.batch_size=choice(32,64,128)'
```

Relative SQLite storage URLs are resolved from Hydra's runtime working
directory. Use an absolute `sqlite:///...` path for shared studies or CI.

The checked-in `src/ddssm/conf/` tree is intentionally a small library of
reusable Hydra defaults; experiment-specific search spaces live in Python under
`experiments/<family>/sweeps.py` (registered into the `sweep` group) or as
CLI overrides.

## Logging

Metrics are written to TensorBoard and CSV by default; **W&B is opt-in**.

```bash
pip install -e .[wandb]
```

W&B is off by default (`wandb=disabled`). Turn it on with the `wandb=enabled`
config group and override fields on the CLI:

```bash
# Cloud W&B
python -m ddssm.app experiment=init_smoke_high_surface wandb=enabled \
    experiment.wandb_config.project=ddssm

# Self-hosted W&B server
python -m ddssm.app experiment=init_smoke_high_surface wandb=enabled \
    experiment.wandb_config.project=ddssm \
    experiment.wandb_config.base_url=https://wandb.example.com
```

The `conf/wandb/{disabled,enabled}.yaml` files carry
`# @package experiment.wandb_config`, so the chosen group always populates
`experiment.wandb_config`. W&B is a *soft dependency*: if the ``wandb`` package
isn't installed the logger silently no-ops and training continues with
TensorBoard + CSV.

## SLURM

Two paths, depending on what you need:

**Single job — emit an `sbatch` script and submit it yourself.** Recommended
for one-off training runs. Resources can live alongside the experiment
config (set `sbatch=SBatch(...)` on `experiment(...)`) and are overridable
on the CLI:

```bash
# emit the script (writes to stdout if --out is omitted)
python -m experiments sbatch --partition=gpu --time=12:00:00 --mem=64G \
    --out=runs/init_mlp.sbatch init_mlp_pinned_per_t__mv

# submit it
sbatch runs/init_mlp.sbatch
# or pass extra Hydra overrides at submit time:
sbatch runs/init_mlp.sbatch experiment.training.stages.n_stage2=12000
```

**Study / sweep — orchestrate a whole registered study.** Recommended for
multi-cell ablations and Optuna search; `ddssm.launch` renders (and optionally
submits) one job per study point:

```bash
# print the sbatch scripts for every point in the study (dry-run is the default)
python -m ddssm.launch init_centering

# write one .sbatch per job and submit them
python -m ddssm.launch init_centering --write-dir runs/jobs --submit

# select a subset / variant and run locally for a smoke check
python -m ddssm.launch init_centering --select dataset=1d --size smoke --local
```

## Development

Run tests (inside the uv environment):

```bash
uv run pytest tests/
```

Format and lint (requires [pre-commit](https://pre-commit.com/)):

```bash
pre-commit run --all-files
```
