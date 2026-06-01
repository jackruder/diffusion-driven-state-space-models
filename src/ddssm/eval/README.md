# `ddssm.eval`

Independent evaluation stage for trained DDSSM models. It loads a checkpoint
into the experiment's already-built model, selects a data-module loader
(`test` by default), walks the metrics named in an `EvalSpec`, merges their
results, and writes a single `metrics.json` to the run dir. Some metrics
(e.g. `loss_tail`) are derived from a training `metrics.csv` rather than from
forward passes. Eval scalars are also best-effort logged to a resumed W&B run
under the `eval/` namespace. Layers (each replaceable): stateless metric
functions registered in `METRIC_REGISTRY`, the `runner` glue, and the CLI.

**Standalone stage:** loads a checkpoint and does NOT train. Drive it with
`python -m ddssm.evaluate experiment=<preset> checkpoint=<path>`.

## Files

- `runner.py` — the `EvalSpec` dataclass (metrics, split, `num_samples`,
  `T_split`, `output_filename`, per-metric `kwargs`) and `evaluate(...)`, which
  loads the checkpoint, builds the `EvalContext`, runs each metric, and writes
  `metrics.json`.
- `metrics.py` — stateless metric functions and the `METRIC_REGISTRY` /
  `register_metric` decorator. Registered metrics include `mae`, `crps_sum`,
  `crps_sum_latent`, `energy_score`, `recon_mse`, `nll`, `bimodal_jsd`,
  `gt_latent_jsd`, `loss_tail`, `wallclock_to_target`,
  `wallclock_to_relative_target`, `stage2_elbo_surrogate`, `sigma_data_drift`,
  `q_aux_kl_trajectory`, and `log_sigma_p2_collapse`.
- `eval_metrics.py` — lower-level building blocks (`mae_metrics`,
  `crps_sum_metrics`, CSV-column readers, divergence detection) folded in here
  and reused by `metrics.py` unchanged.
- `synthetic_kernels.py` — closed-form ground-truth transition kernels
  (`lgssm`, `nonlinear-bimodal-lift` 1-D/MV) used by the `gt_latent_jsd` metric
  to compare the learned `p_ψ(z_t | z_{t-1})` against the true generator.
- `__init__.py` — re-exports `EvalSpec`, `evaluate`, `EvalContext`,
  `METRIC_REGISTRY`, `register_metric`.
