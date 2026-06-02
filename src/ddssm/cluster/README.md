# `ddssm.cluster`

SLURM / study orchestration internals. This package holds the pure, reusable
building blocks used to define a *study* (a parametrized family of experiment
points), render the SLURM submit scripts that launch them, and summarize the
health of the runs they produce. It contains no top-level entry point of its
own — the drivers (`python -m ddssm.launch`, `python -m ddssm.colocate`) live
at the package root and import from here.

## Files

- **`sbatch.py`** — renders SLURM submit scripts (and submits them).
  `render_sbatch(name, ...)` emits a single-job `.sbatch` that runs
  `python -m ddssm.app experiment=<name> "$@"`; `render_packed_sbatch(...)`
  packs K workers of one cell onto a GPU, and `render_multicell_packed_sbatch(...)`
  packs workers from several cells onto one GPU. `submit_sbatch(path)` shells
  out to `sbatch`. Resources resolve as CLI overrides → the experiment's
  `SBatch` field → `DEFAULT_SBATCH` (the project default). Under preempt mode a
  `PreemptSpec` injects `--requeue` / `--signal=B:USR1@<grace>` /
  `--open-mode=append` plus a bash preamble that calls
  `python -m ddssm.launch_remaining` to size the per-worker trial budget,
  exports the trainer's preempt env vars, and traps `SIGUSR1`/`SIGTERM` to the
  child. `CellWorker` describes one packed worker for the multi-cell renderer.
- **`study.py`** — the pure (no-I/O) `Study` abstraction. A `Study` is a named
  family of `StudyPoint`s plus a per-point `launch` callable (returning a
  `ddssm.launch.PointLaunch`). Build one from `Study.from_axes(...)` (the
  cross-product of `Axis` comparison dimensions, with collision-checked point
  names) or `Study.from_points(...)`. Methods: `register(store)`, `names()`,
  tag-filtered `select(**filters)`, and `point(name)`.
- **`report.py`** — self-describing run health. `summarize_run(run_dir)` reduces
  a run's `metrics.csv` to a compact dict (final/head/tail loss, λ-warmup state,
  σ_data² drift, val loss, non-finite count, elapsed, stages); `write_run_summary`
  dumps it to `<run_dir>/run_summary.json` (called by `Experiment.train` at
  exit). The `python -m ddssm.cluster.report <path>` CLI prints one run's
  summary, or a one-row-per-run table over a parent directory of run dirs.

## How it fits

These modules are pure mechanism. The orchestration drivers stay at the
top level as the actual entry points and consume them:

- `python -m ddssm.launch <study>` (`ddssm/launch.py`, **not** in this package)
  drives a whole `Study` — its `StudyOrchestrator` resolves each point's
  `PointLaunch` and renders/submits via `sbatch.py`.
- `python -m ddssm.colocate` packs every cell onto each GPU using the
  multi-cell packed renderer to add trials to an existing study at low
  concurrency.

All imports here are absolute (`from ddssm.experiment import SBatch`,
`ddssm.launch.PointLaunch`); `launch.py` imports from `sbatch.py`, never the
reverse.
