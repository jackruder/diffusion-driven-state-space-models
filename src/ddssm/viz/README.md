# `ddssm.viz`

Independent visualization stage for trained DDSSM models. Same three-layer
shape as `ddssm.eval`: it loads a checkpoint into the experiment's built model,
selects a data-module loader (`test` by default), walks the plots named in a
`VizSpec`, and saves PNGs to the run dir. Each saved PNG is also best-effort
pushed to a resumed W&B run under the `viz/` namespace. Layers: stateless plot
functions registered in `PLOT_REGISTRY`, the `runner` glue, and the CLI.

**Standalone stage:** loads a checkpoint and does NOT train. Drive it with
`python -m ddssm.visualize experiment=<preset> checkpoint=<path>`.

## Files

- `runner.py` — the `PlotSpec` (registry `name`, `save_filename`, `kwargs`) and
  `VizSpec` (list of plots, `split`, `num_samples`, `T_split`) dataclasses, plus
  `visualize(...)`, which loads the checkpoint, builds the `PlotContext`, runs
  each plot, and returns the saved PNG paths.
- `plots.py` — stateless plot functions and the `PLOT_REGISTRY` /
  `register_plot` decorator. Registered plots: `forecast_1d`,
  `forecast_2d_spatial`, `forecast_distribution`, and `metrics_csv`. Drawing is
  split into composable pieces so a single panel can be produced in isolation.
- `__init__.py` — re-exports `VizSpec`, `PlotSpec`, `visualize`, `PlotContext`,
  `PLOT_REGISTRY`, `register_plot`.
