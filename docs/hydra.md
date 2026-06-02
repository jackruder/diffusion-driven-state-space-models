# Hydra: a working tutorial

[Hydra](https://hydra.cc) is how every DDSSM run is configured from the command
line. This page teaches the Hydra concepts you need *through the way this
project actually uses them* — so the examples are all runnable. For the broader
picture of how a run is assembled, see {doc}`architecture`.

## The one-sentence model

Hydra **composes** a single config tree from a base file plus *config groups*,
lets you **override** any value (or swap a whole group) from the command line,
and can **multirun** the same command across a sweep of values. That's it —
everything below is a variation on compose / override / multirun.

## The base config

Every invocation of `python -m ddssm.app` starts from
`src/ddssm/conf/config.yaml`:

```yaml
defaults:
  - experiment: init_smoke_simple   # the `experiment` config group
  - wandb: disabled                 # the `wandb` config group
  - _self_

# (no other top-level keys — the content lives under `experiment.*`)
```

The `defaults:` list is Hydra's **composition recipe**. Each `group: option`
line pulls in one option from a config group and merges it into the tree.
`_self_` says "apply this file's own keys last" (here there are none beyond the
defaults). The result is a nested config rooted almost entirely under
`experiment.*` (see [The config namespace](#the-config-namespace)).

```{note}
Run `python -m ddssm.app --cfg job` to print the fully composed config without
running anything. This is the single most useful Hydra command while developing
— it shows exactly what your overrides produced.
```

## Two config worlds (important for this project)

Most Hydra projects put their config *options* in YAML files inside the group
directories (`conf/experiment/foo.yaml`). **DDSSM does not.** It has two
deliberately separate worlds:

- **`src/ddssm/conf/`** — a small library of reusable **YAML** defaults only:
  `config.yaml`, the `wandb/` group, and the `hydra/sweeper/` presets. No
  experiment presets live here.
- **`experiments/`** (repo root, outside `src/`) — the experiment presets,
  defined **in Python** with [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/)
  `builds(...)` and registered into Hydra's config store at import time.

So when you write `experiment=init_smoke_high_surface`, Hydra is selecting an
option that was registered from Python, not from a `conf/experiment/*.yaml`
file. From the command line this is invisible — you still select and override it
exactly like a YAML-defined group. (The bridge: on startup
{py:func}`ddssm.experiment.registry.register_experiments` imports the
`experiments/` package, whose modules register presets into the hydra-zen
`store`, which is then published to Hydra's `ConfigStore`.)

To **list** every registered option:

```bash
python -m experiments list          # all experiment presets
```

## Selecting a config-group option

Swap the whole `experiment` group by name (no `+`, because `experiment` is
already in the defaults list):

```bash
python -m ddssm.app                                      # default: init_smoke_simple
python -m ddssm.app experiment=init_smoke_high_surface   # pick another preset
```

Same for `wandb` (also in the defaults list):

```bash
python -m ddssm.app wandb=enabled
```

## Overriding individual values

Use **dotted paths** into the config tree, `key.subkey=value`:

```bash
# single value
python -m ddssm.app experiment=init_smoke_high_surface \
    experiment.training.stages.n_stage2=2000

# several at once
python -m ddssm.app \
    experiment.data.batch_size=64 \
    experiment.model.latent_dim=8 \
    experiment.hparams.enc_lr=3e-4
```

### The config namespace

Almost everything lives under `experiment.*`. The branches you'll override most:

| Path prefix                      | What it controls                                  |
| -------------------------------- | ------------------------------------------------- |
| `experiment.data.*`              | data module (`mode`, `T`, `D`, `batch_size`, …)   |
| `experiment.model.*`             | model dims/architecture (`latent_dim`, `j`, …)    |
| `experiment.hparams.*`           | optimizer hparams (`enc_lr`, `dec_lr`, `S`, …)    |
| `experiment.training.*`          | run scalars (`log_every`, `validate_every`, …)    |
| `experiment.training.stages.*`   | the multi-stage step budget (see gotcha below)    |
| `experiment.wandb_config.*`      | W&B fields (when `wandb=enabled`)                 |

```{important}
**Step budget gotcha.** The shipped presets are *multi-stage*, so the budget is
`experiment.training.stages.n_pretrain` / `n_stage2`. The flat
`experiment.training.steps` is read **only** by the single-fit (non-staged)
path, so overriding it on a staged preset does nothing. When in doubt,
`--cfg job` and look for the `stages:` block.
```

### Add vs. override: the `+` and `++` prefixes

- `key=value` — override a key that **already exists**. Errors if it doesn't.
- `+key=value` — **append** a new key (or select a group not in the defaults).
- `++key=value` — append-or-override (force), whether or not it exists.

Two config groups are populated but **not** in the defaults list, so you *add*
them with `+`:

```bash
python -m ddssm.app experiment=init_smoke_simple +data=harmonic   # add the `data` group
python -m ddssm.app experiment=... +sweep=init_ablation           # add a sweep search space
```

The standalone stages take their checkpoint the same way (the key isn't in their
base config, so it's appended):

```bash
python -m ddssm.evaluate experiment=init_smoke_high_surface \
    +checkpoint='${experiment.checkpoint_dir}/ckpt_latest.pth'
python -m ddssm.evaluate experiment=... +checkpoint=path/to/ckpt.pth +csv_path=out/metrics.csv
```

If you forget the `+`, Hydra raises `Could not override 'data'. ... To append to
your config use +data=...` — that error message is telling you exactly what to
do.

### Quoting

Your shell, not Hydra, splits arguments — quote anything with characters the
shell would eat. Lists and interpolations are the usual culprits:

```bash
python -m ddssm.app wandb=enabled \
    'experiment.wandb_config.tags=[recon,synth]' \
    +checkpoint='${experiment.checkpoint_dir}/ckpt_latest.pth'
```

## Inspecting before you run

```bash
python -m ddssm.app experiment=foo --cfg job     # composed config (resolved)
python -m ddssm.app --help                        # base config + how to override
python -m ddssm.app experiment=foo --info config  # config sources / search path
```

## The run directory

Hydra creates a timestamped run directory (under `outputs/…` for single runs,
`multirun/…` for sweeps) and writes a full record there: `resolved_config.yaml`
(the exact composed config), plus this project's `metrics.csv`, `tb_logs/`, and
`checkpoints/`.

```{note}
This project sets **`hydra.job.chdir=False`** — the process does *not* `cd` into
the run dir. Instead, `Experiment.train` anchors `checkpoints/`, logs, etc.
inside the passed `run_dir`, so per-run outputs are self-contained without
relying on a working-directory switch.
```

## Multirun & Optuna sweeps

`--multirun` (`-m`) runs the command once per value in a comma-separated list, or
— with the Optuna sweeper — searches a space:

```bash
# grid: three runs, one per latent_dim
python -m ddssm.app --multirun experiment.model.latent_dim=4,8,16

# Optuna search over a registered space (`+sweep=`), 20 trials, named study
python -m ddssm.app --multirun experiment=init_smoke_high_surface +sweep=init_ablation \
    hydra.sweeper.n_trials=20 \
    hydra.sweeper.study_name=my_study \
    hydra.sweeper.storage=sqlite:///my_study.db
```

The sweeper itself is configured by the `hydra/sweeper` group
(`src/ddssm/conf/hydra/sweeper/ddssm_optuna.yaml` and the MOO variant). Its keys
live under the special `hydra.*` namespace — override them with
`hydra.sweeper.<key>=...`.

Search **spaces** are Python presets too (registered into the `sweep` group in
`experiments/init_centering/sweeps.py`: `init_ablation`, `init_ablation_moo`,
`init_ablation_moo_r2`, plus the `init_pilot` alias). You can also define a space
inline with `hydra.sweeper.params.*` overrides instead of `+sweep=`.

```{important}
**Two sweep footguns** (both documented in the sweeper YAML):

- **`storage`** defaults to an absolute path under `$PWD`. A *relative*
  `sqlite:///…` URL resolves against each trial's runtime working directory and
  silently fragments the study (one DB per trial). Use an absolute path or a
  network URL for shared/CI studies.
- **`hydra.sweeper.sampler.seed`** defaults to `42`. A distributed launch must
  override it *per worker*, or every worker draws the identical trial sequence
  and the cell collapses to duplicate configs. `ddssm.launch` injects a
  per-worker seed automatically; if you roll your own parallelism, set
  `hydra.sweeper.sampler.seed=<N>` yourself.
```

For rendering/submitting whole studies to SLURM, see `python -m ddssm.launch`
and {py:mod}`ddssm.cluster.sbatch` (covered in {doc}`architecture`).

## Interpolation & resolvers

Hydra configs (via OmegaConf) support `${…}` interpolation. The `wandb=enabled`
config is a live example of the resolvers this project relies on:

```yaml
name:  ${hydra:job.override_dirname}                       # Hydra runtime info
group: ${oc.select:hydra.sweeper.study_name,${hydra:job.name}}  # value-or-fallback
```

- `${hydra:…}` — read Hydra's own runtime metadata (job name, override dirname).
- `${oc.select:a,b}` — use `a` if it resolves, else fall back to `b`.
- `${oc.env:VAR,default}` — read an environment variable (used by the sweeper's
  `storage` default, `sqlite:///${oc.env:PWD,.}/runs/optuna/…`).

These resolve **lazily**, when the value is read — which is why `--cfg job`
prints the literal `${…}` while `--cfg job --resolve` shows the resolved value.

## Why hydra-zen, and what it changes for you

Nothing, at the command line. hydra-zen's `builds(SomeClass, ...)` produces a
structured config that targets a real Python class (`_target_`), and
`instantiate(cfg)` constructs it. The benefit is that presets are typed Python
(refactor-safe, importable, testable) rather than stringly-typed YAML. You still
select presets with `experiment=NAME` and override with the same dotted paths —
the `_target_` keys you see in `--cfg job` output are just how hydra-zen records
which class to build.

To **add** an experiment, add a Python file under `experiments/<family>/` and
register it (see the family's `README.md` and {doc}`architecture`) — do **not**
add a YAML file under `conf/`.

## Cheat sheet

```bash
python -m experiments list                              # list presets
python -m ddssm.app experiment=NAME --cfg job           # inspect composed config
python -m ddssm.app experiment=NAME                     # run it
python -m ddssm.app experiment=NAME k.path=value        # override a value
python -m ddssm.app experiment=NAME +data=harmonic      # add a group not in defaults
python -m ddssm.app wandb=enabled                       # swap a default group
python -m ddssm.app --multirun k=1,2,3                  # grid multirun
python -m ddssm.app -m experiment=NAME +sweep=SPACE \   # Optuna sweep
    hydra.sweeper.n_trials=N hydra.sweeper.study_name=S
```

Further reading: the [Hydra docs](https://hydra.cc/docs/intro/) and the
[hydra-zen docs](https://mit-ll-responsible-ai.github.io/hydra-zen/).
