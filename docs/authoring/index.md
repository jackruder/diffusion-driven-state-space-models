# Authoring an experiment

This guide walks through creating a new DDSSM experiment end-to-end: choosing
data, building a model, configuring training, logging, metrics, sweeps, and
visualization. For the runtime architecture see {doc}`../architecture`; for the
Hydra mechanics (overrides, `--cfg job`, multirun) see {doc}`../hydra`.

## The mental model

An experiment is a Python-defined preset, not a YAML file:

- Presets are built with **hydra-zen** `builds(...)` configs and registered into
  a store **at import time** (see {doc}`../architecture` → "Two config worlds").
- The {py:func}`experiments._make.experiment` factory ties a dataset + model +
  training spec into one config and curries `hparams` onto the trainer.
- A **family** is a package under `experiments/<family>/`. Its `__init__.py`
  imports the submodule that registers presets, and the family is added to the
  import list in `experiments/__init__.py` so `register_experiments()` loads it.

To add an experiment you write Python under `experiments/<family>/` and register
it with {py:obj}`ddssm.experiment.stores.experiment_store` — you never add a
YAML file under `conf/`.

## The worked example: `synthetic_validation`

Every page below is anchored by a real, runnable family,
`experiments/synthetic_validation/`. It trains **one simple, hand-built model**
across **several 1-D synthetic datasets** (sine/`harmonic`, `lgssm`, `bimodal`)
— a simple validation that the model recovers known dynamics. It registers one preset
per dataset:

```bash
python -m experiments list | grep synthval
# synthval__bimodal
# synthval__harmonic
# synthval__lgssm

python -m ddssm.app experiment=synthval__harmonic --cfg job   # inspect
python -m ddssm.app experiment=synthval__harmonic              # train
```

The family is three small files — read them alongside this guide:

| File | Role |
| ---- | ---- |
| `experiments/synthetic_validation/model.py` | the hand-written model factory ({doc}`model`) |
| `experiments/synthetic_validation/experiments.py` | the dataset loop + registration ({doc}`dataset`, {doc}`training`, {doc}`metrics`) |
| `experiments/synthetic_validation/__init__.py` | import-time registration trigger |

## The seven steps

```{toctree}
:maxdepth: 1

dataset
model
training
logging
metrics
sweeps
visualization
```

0. {doc}`dataset` — pick (or define) the data.
1. {doc}`model` — decide the architecture and write the model factory.
2. {doc}`training` — set the training scalars, hyperparameters, and stages.
3. {doc}`logging` — understand what is logged and where; enable W&B.
4. {doc}`metrics` — choose eval metrics and the Optuna objective.
5. {doc}`sweeps` — define an Optuna search space.
6. {doc}`visualization` — wire forecast/diagnostic plots.
