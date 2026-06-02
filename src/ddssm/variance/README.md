# `ddssm.variance`

Independent variance-probe stage for trained DDSSM models. It loads a
checkpoint, runs a probe loop that measures the variance of the diffusion-loss
and its gradients across replicas and τ-bins (over a grid of `ProbeCell`s —
objective `esm`/`dsm` × k-sampling `uniform`/`lsgm_is`), computes the requested
probe metrics, renders the requested probe plots, and writes a per-sample
`variance_raw.csv` plus a `variance_summary.json` to the run dir. Same
registry-driven shape as `ddssm.eval`/`ddssm.viz`: metric and plot registries,
a `ProbeSpec`-driven `runner`, and the CLI.

**Standalone stage:** loads a checkpoint and does NOT train (it reads the train
loader). Drive it with `python -m ddssm.variance experiment=<preset>` (this
package ships a `__main__.py`). The CLI also supports `+checkpoint=<path>` and a
`+per_step=true` mode that probes every step checkpoint and stitches GIFs.

## Files

- `runner.py` — the config dataclasses (`ProbeCell`, `ProbeMetricSpec`,
  `ProbePlotSpec`, and the top-level `ProbeSpec` carrying cells, replica counts
  `R`/`B_var`/`n_batches`, `K_bins`, `seeds`, frozen submodules, metrics, plots,
  and output filenames) plus `variance(...)`, which runs the probe and writes
  the CSV/JSON/PNGs.
- `probe.py` — the core probe loop (`run_probe`) that drives the forward/backward
  passes producing the raw per-sample variance samples.
- `metrics.py` — `ProbeContext`, the `PROBE_METRIC_REGISTRY` /
  `register_probe_metric` decorator, and metrics: `loss_var`, `grad_var`,
  `ratio_esm_dsm`, and their per-τ variants (`loss_var_per_tau`,
  `grad_var_per_tau`, `ratio_per_tau`, `var_per_tau`).
- `plots.py` — `ProbePlotContext`, the `PROBE_PLOT_REGISTRY` /
  `register_probe_plot` decorator, and plots: `var_grad_vs_tau`,
  `var_loss_vs_tau`, `ratio_vs_tau`, `summary_table`, `var_grad_vs_step`.
- `aggregate.py` — `aggregate_summaries(...)`: collects many
  `variance_summary.json` files into one markdown report.
- `cli.py` — the Hydra entry point (`main`) with the single / `+checkpoint` /
  `+per_step` modes.
- `__main__.py` — wires `python -m ddssm.variance` to `cli.main`.
- `__init__.py` — re-exports the specs, contexts, registries, and `variance`.
