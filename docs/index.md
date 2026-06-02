# DDSSM documentation

**Diffusion-Driven State Space Models** — a PyTorch framework for probabilistic
time-series forecasting. It jointly trains a variational encoder/decoder over
latent states and a transition prior (Gaussian or CSDI-style diffusion) under an
ELBO objective.

This documentation focuses on the **implementation**: how the code is laid out,
how the pieces compose, and the API reference generated from the source
docstrings. For the modeling background and the preset table, see the top-level
`README.md`.

```{toctree}
:maxdepth: 2
:caption: Contents

architecture
hydra
api
```

## Quick orientation

| You want to…                        | Start at                                   |
| ----------------------------------- | ------------------------------------------ |
| Understand how a run is assembled   | {doc}`architecture` → "Composition"        |
| Add a new experiment preset         | `experiments/<family>/` (see `README.md`)  |
| Find a class/function's signature   | {doc}`api`                                 |
| Understand one subsystem            | the `README.md` inside that package dir    |

## Entry points

All CLIs are `python -m ddssm.<name>`:

- `ddssm.app` — train a registered experiment (the main entry point)
- `ddssm.evaluate` / `ddssm.visualize` / `ddssm.variance` — standalone
  post-training stages (load a checkpoint; no training)
- `ddssm.launch` — render/submit a whole study
- `ddssm.colocate` — pack multiple cells onto a GPU
