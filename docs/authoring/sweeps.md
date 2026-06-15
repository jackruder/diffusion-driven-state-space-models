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

## Storage backends — SQLite vs Postgres

Optuna persists a study in an **RDB** so trials can be added incrementally and
many workers can collaborate. The choice of backend is purely a storage URL.

**SQLite (default).** The sweeper preset
(`src/ddssm/conf/hydra/sweeper/ddssm_optuna.yaml`) defaults to a per-study
SQLite file under `runs/optuna/<study_name>.db`. SQLite is fine for a single
node and a handful of workers. Over NFS it tolerates only ~8 concurrent workers
per DB (lock contention) — the `plan-campaign` skill caps it there. The
`optuna-dashboard sqlite:///path.db` UI reads it directly.

**Postgres (many workers / shared study).** When you need more concurrency than
NFS-SQLite allows, or you want **every cell of a study in one database**, point
the storage at a Postgres server (`psycopg2-binary` is already a dependency).
One server, reachable from every compute node; each cell is a distinct Optuna
`study_name` within the same DB. Three entry points use it:

```bash
# 1. A single sweep — override the sweeper storage with a Postgres URL:
python -m ddssm.app --multirun experiment=synthval__harmonic +sweep=my_sweep \
    hydra.sweeper.n_trials=40 hydra.sweeper.study_name=synthval_lr \
    hydra.sweeper.storage=postgresql://ddssm@dbhost:5432/ddssm

# 2. A whole study — --storage-url puts all cells in one DB as distinct studies
#    (the orchestrator initialises the schema once via Optuna's RDBStorage):
python -m ddssm.launch <study> --storage-url postgresql://ddssm@dbhost:5432/ddssm \
    --study-prefix round1 --write-dir runs/sbatch --submit

# 3. Add trials to an existing shared-DB study (low per-cell concurrency for
#    NSGA-II to evolve) — point ddssm.colocate at the SAME url + --study-prefix:
python -m ddssm.colocate <study> --select dataset=mv \
    --n-gpus 3 --workers-per-cell 2 --target 96 --sweep init_ablation_moo_r2 \
    --storage-url postgresql://ddssm@dbhost:5432/ddssm --study-prefix round1
```

Setup is just "have a reachable Postgres DB and pass its URL" — create a
database (e.g. `createdb ddssm`) on a host the nodes can reach, then use the
`postgresql://user@host:port/db` URL everywhere. Inspect any backend with
`optuna-dashboard <url>`.

```{note}
The `study_name` is the join key. To **resume or extend** a run, reuse the same
`--storage-url` + `--study-prefix` (study launches) or `hydra.sweeper.study_name`
(single sweeps); a new name starts a fresh study in the same DB. Preemptible
launches compute the remaining budget against the existing COMPLETE trials in
that study (`ddssm.launch_remaining`), so requeues converge to the target rather
than over-running.
```

## Reference example

`experiments/init_centering/sweeps.py` builds three real spaces
(`init_ablation`, `init_ablation_moo`, `init_ablation_moo_r2`) over `StagesB`
fields — the clearest template for authoring your own. For
`synthetic_validation`, a natural first sweep targets the LRs/budget on
`StagesB` (e.g. `base_lr`, `n_pretrain`, `n_stage2`) with
`objective=Objective(metric="loss/total")`.
