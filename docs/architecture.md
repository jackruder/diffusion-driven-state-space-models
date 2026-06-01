# Architecture & implementation guide

This guide describes how the code is organized and how a run is assembled — the
machinery *around* the model, not the modeling math (for that, see the
top-level `README.md`). The whole system hangs off one composition point,
{py:class}`~ddssm.experiment.experiment.Experiment`; once you understand how it
is built and run, the rest of the tree is navigable.

## Package layout

```
src/ddssm/
  app.py  evaluate.py  visualize.py        # CLI entry points (python -m ddssm.<x>)
  launch.py  colocate.py  launch_remaining.py
  nn/          # reusable neural building blocks (layers, embeddings, mixers)
  model/       # the variational SSM: dssd, encoder, decoder, losses
    transitions/   centering/   likelihood/
  training/    # trainer, multi-stage orchestration, checkpoint, loggers
  experiment/  # Experiment composition root + builders / stores / registry
  cluster/     # sbatch rendering, study points, reporting
  eval/  viz/  variance/   # standalone post-training stages
  data/        # data modules (synthetic, KDD, GluonTS, null)
  conf/        # reusable Hydra YAML defaults (NOT the presets themselves)
```

Each subpackage has a `README.md` describing its interface and files — start
there when working inside one.

**Import convention.** Intra-package imports are absolute
(`from ddssm.<pkg>.<mod> import …`). The `python -m ddssm.<x>` entry-point
module paths are a stable public contract (they are baked into generated sbatch
scripts and the docs) and live at the top level.

## How a run is assembled

The entry point is {py:mod}`ddssm.app`:

1. `@hydra.main` reads `src/ddssm/conf/config.yaml`.
2. {py:func}`ddssm.experiment.registry.register_experiments` adds the repo root
   to `sys.path`, imports the `experiments/` package (whose submodules register
   named presets into the hydra-zen `store` at import time), and publishes them
   to Hydra's `ConfigStore`.
3. `instantiate(cfg.experiment)` builds an
   {py:class}`~ddssm.experiment.experiment.Experiment`.
4. `experiment.train(device, run_dir)` runs it. The Hydra run directory holds
   `metrics.csv`, `tb_logs/`, `checkpoints/`, and `resolved_config.yaml`.

```{note}
`python -m experiments list` enumerates every registered preset, and
`python -m ddssm.app experiment=<name> --cfg job` resolves a config without
instantiating runtime classes — the fastest way to surface a broken preset.
```

### Two config worlds

These are deliberately separate:

- **`src/ddssm/conf/`** is a small library of reusable Hydra YAML defaults
  (`config.yaml`, `hydra/sweeper/`, `wandb/`). It does **not** contain
  experiment presets.
- **`experiments/`** (at the repo root, outside `src/`) is where named presets
  are defined **in Python** via hydra-zen `builds(...)`. To add an experiment,
  add a Python file under `experiments/<family>/` and register it — do not add
  YAML.

## The `Experiment` object

{py:class}`~ddssm.experiment.experiment.Experiment` is a dataclass that owns the
full run definition:

- `data` — a {py:class}`~ddssm.data.datamodule.DDSSMDataModule`
  (train/val/test loaders + a `batch_transform`).
- `model` — the variational SSM ({py:class}`ddssm.model.dssd.DDSSM_base`).
- `build_trainer` — a partial {py:class}`~ddssm.training.train.DDSSMTrainer`
  factory.
- `training` — `TrainingScalars`: step budget, logging cadence, and the
  per-module `trainable` (`requires_grad`) mask.
- `eval` / `viz` / `variance` — specs for the standalone stages.
- `objective` — reads `metrics.csv` and returns the mean tail loss, used as the
  Optuna objective; if `None`, `train()` returns the trainer instead.

`Experiment.train`, `evaluate`, `visualize`, and `variance_probe` are
independent entry methods — the latter three load a checkpoint and do not train.

The public objects are re-exported from the package root, so
`from ddssm.experiment import Experiment` works despite `Experiment` living in
`ddssm/experiment/experiment.py`.

{py:mod}`ddssm.experiment.builders` is a convenience surface of hydra-zen
`builds(...)` configs for assembling an `Experiment` ad hoc (e.g. in a
notebook). The shipped `init_centering` family does **not** route through it —
its `model.py` composes the model from runtime classes directly.

## The model (`ddssm.model`)

{py:class}`ddssm.model.dssd.DDSSM_base` is the ELBO model: an encoder
`q_ϕ(z|x)`, decoder `p_θ(x|z)`, and a **pluggable** transition prior. The
transition is either Gaussian
({py:class}`~ddssm.model.transitions.transitions.GaussianTransition`, or the
baseline-centering
{py:class}`~ddssm.model.transitions.baseline_gaussian.BaselineGaussianTransition`)
or a CSDI-style diffusion denoiser
({py:class}`~ddssm.model.transitions.diffusion.DiffusionTransition`).

The encoder/decoder/transition networks are themselves composed from reusable
mixers in {py:mod}`ddssm.nn` (per-channel time and feature mixers:
`conv`/`gru`/`identity` × `transformer`/`conv`/`identity`). These are selected
in Python when building the model, not via CLI config groups.

The model exposes a `ProbeBatch` payload (encoded latents + log-q paths) reused
by the variance probe.

## The training stack (`ddssm.training`)

{py:class}`~ddssm.training.train.DDSSMTrainer` owns a per-submodule optimizer
(separate LRs for encoder/decoder/z_init/transition), AMP, CSV/TensorBoard/W&B
logging, checkpointing, and the λ-warmup schedule.

```{important}
`DDSSMTrainer._set_trainable(...)` toggles `requires_grad` per submodule — that
mask is the **single** mechanism for stage-aware gradient suppression. The
forward pass always computes every ELBO term; frozen submodules simply do not
accumulate gradients.
```

{py:class}`~ddssm.training.stages.StageOrchestrator` runs sequential phases
(e.g. recon-only → trans-only → joint) with per-stage trainable masks, LRs,
scheduler, and λ-ramp. The shipped presets are multi-stage, so the step budget
is `training.stages.n_pretrain` / `n_stage2` (not `training.steps`, which only
the single-fit path reads).

## Standalone stages (`eval` / `viz` / `variance`)

Each has its own runner plus a registry (a metric or plot dict). All three load
a checkpoint and read `metrics.csv` from the run dir — they do **not** train.

- {py:mod}`ddssm.eval` — `EvalSpec`, metric registry (MAE, CRPS-sum, recon
  divergence), writes `metrics.json`. CLI: `python -m ddssm.evaluate`.
- {py:mod}`ddssm.viz` — `VizSpec`, plot registry, writes PNGs. CLI:
  `python -m ddssm.visualize`.
- {py:mod}`ddssm.variance` — `ProbeSpec`, variance-probe metrics/plots, writes
  `variance_raw.csv` + `variance_summary.json`. CLI: `python -m ddssm.variance`.

## Cluster orchestration (`ddssm.cluster`)

{py:mod}`ddssm.cluster.sbatch` renders SLURM submit scripts (the generated
scripts invoke `python -m ddssm.app` and `python -m ddssm.launch_remaining`).
{py:mod}`ddssm.cluster.study` defines study points, and {py:mod}`ddssm.launch`
orchestrates rendering/submitting a whole study; {py:mod}`ddssm.colocate` packs
multiple cells onto one GPU.

## Conventions

- **Hydra `chdir=False`** — `Experiment.train` anchors checkpoints inside
  `run_dir`, so per-run outputs are self-contained.
- **W&B is opt-in** — install `.[wandb]`; the logger silently no-ops if the
  package is missing. Off by default (`wandb=disabled`).
- **`hparams` is shared** — `Experiment.hparams` and
  `Experiment.model.config.hyperparams` are kept identical; `Experiment.train`
  re-syncs them defensively.
- **Mamba is optional** — `build_mamba.sh` documents the manual build; the
  `MambaTimeLayer` wiring is present but commented out.
