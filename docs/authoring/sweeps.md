# Step 5 — Configuring sweeps

A sweep is an Optuna search over config overrides. The CLI mechanics
(`--multirun`, `+sweep=`, `hydra.sweeper.*`, the storage/seed footguns) are in
{doc}`../hydra`; this page is about **authoring a search space in Python**.

## `SweepSpace`

`experiments/_sweep.py` provides a small builder. You point it at a target
config (whose fields it validates against) and a `prefix` (the Hydra override
path), then declare axes:

```python
from experiments._sweep import SweepSpace
from experiments.init_centering.hparams import StagesB

space = SweepSpace(target=StagesB, prefix="experiment.training.stages")
space.log("base_lr", 1e-5, 1e-3)      # log-uniform float
space.log_int("n_pretrain", 5, 500)   # log-uniform int
space.uniform("dropout", 0.0, 0.5)    # uniform float
space.raw("foo", "choice(a, b, c)")   # raw Optuna distribution string
```

Every field name is checked against `target` **at construction time**, so a typo
fails fast on `python -m experiments list` rather than mid-trial. `.params()`
returns the accumulated `{override_path: distribution}` dict.

## Building & registering

`.build(...)` produces a preset that selects the sweeper group and sets
`hydra.sweeper.direction` + `params`. Register it into the `sweep` group with
{py:obj}`ddssm.experiment.stores.sweep_store`:

```python
from ddssm.experiment.stores import sweep_store

# single-objective (TPE)
MySweep = space.build(sweeper="ddssm_optuna", direction="minimize")
sweep_store(MySweep, name="my_sweep")

# multi-objective (NSGA-II) — direction length must match objectives.specs
MyMOO = space.build(sweeper="ddssm_optuna_moo",
                    direction=["minimize", "minimize"],
                    objectives=MyMOObjective)   # an Objectives instance
sweep_store(MyMOO, name="my_sweep_moo")
```

`sweeper="ddssm_optuna"` / `"ddssm_optuna_moo"` select the sweeper presets in
`src/ddssm/conf/hydra/sweeper/`. The single vs multi choice must agree with the
experiment's `objective` (a single `Objective` vs an `Objectives`; see
{doc}`metrics`).

## Running it

```bash
python -m ddssm.app --multirun experiment=synthval__harmonic +sweep=my_sweep \
    hydra.sweeper.n_trials=20 \
    hydra.sweeper.study_name=synthval_lr \
    hydra.sweeper.storage=sqlite:///$PWD/runs/optuna/synthval_lr.db
```

```{important}
Heed the two footguns from {doc}`../hydra`: give `storage` an **absolute** path
(a relative `sqlite://` fragments the study), and on distributed launches
override `hydra.sweeper.sampler.seed` **per worker** (else every worker draws
identical trials).
```

## Reference example

`experiments/init_centering/sweeps.py` builds three real spaces
(`init_ablation`, `init_ablation_moo`, `init_ablation_moo_r2`) over `StagesB`
fields — the clearest template for authoring your own. For
`synthetic_validation`, a natural first sweep targets the LRs/budget on
`StagesB` (e.g. `base_lr`, `n_pretrain`, `n_stage2`) with
`objective=Objective(metric="loss/total")`.
