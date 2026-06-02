# Step 3 — Logging

Training metrics are aggregated by a {py:class}`~ddssm.training.loggers.MetricStore`
and fanned out to several backends. You rarely configure this directly — the
trainer wires it from your `Training`/`wandb` settings — but you should know what
lands where and what the metric keys mean.

## Backends

`src/ddssm/training/loggers.py`:

- **CSVLogger** → `metrics.csv` in the run dir. Tolerates schema drift (new
  keys at stage/λ boundaries trigger a header rewrite). One row per flush, with
  `split` + `step` columns plus every metric key seen.
- **TensorBoardLogger** → `tb_logs/`. Scalars namespaced `<split>/<key>`.
- **WandbLogger** → Weights & Biases (opt-in; no-op if W&B absent or disabled).
  Persists the run id to `<run_dir>/.wandb_run_id` so post-training stages reuse
  the run.
- **ConsoleLogger** → stdout per-step summaries.

`MetricStore` accumulates into per-split meters chosen by glob (`MetricSpec`):
`mean` (validation, batch-weighted), `last` (per-step training values), `sum`,
`ema`.

## What gets logged each step

The model returns unweighted `LossComponents`; the loss object weights them and
the trainer logs both. Keys you'll see as `metrics.csv` columns / TB scalars:

| Key | Meaning |
| --- | ------- |
| `loss/total` | weighted ELBO: `recon + λ(init_kl+trans_kl) + λ_σp·r_sigma_p + λ_μp·r_mu_p` |
| `loss/total_unweighted` | same terms, unweighted |
| `loss/distortion/rec` | reconstruction NLL |
| `loss/rate/init/tot` (+ `vhp`, `kl_aux`, `entropy`, `loss_init`) | initial-state term and sub-components |
| `loss/rate/trans/kl` | transition KL summed over time |
| `loss/rate/trans/r_sigma_p`, `r_mu_p` | centering regularizers (unweighted) |
| `optim/lambda` | rate-λ at this step (from the loss's schedule) |
| `calib/ratio_res2_to_sigma2` | decoder variance calibration |
| `diag/sigma_data2/t=<k>` | per-timestep σ_data² buffer values |
| `time/elapsed_s` | wall-clock since start (used by `wallclock_*` metrics) |
| `stage/idx`, `stage/step_within` | which stage, and step within it |
| `nonfinite/total` | cumulative NaN/Inf counter |

These are exactly the columns the worked example writes — inspect a run's
`metrics.csv`, or open `tb_logs/` with TensorBoard.

## Enabling Weights & Biases

W&B is a Hydra config group (`src/ddssm/conf/wandb/`), off by default. Turn it on
per run (see {doc}`../hydra` for the `group=` override mechanics):

```bash
python -m ddssm.app experiment=synthval__harmonic wandb=enabled \
    experiment.wandb_config.project=ddssm-synthval \
    'experiment.wandb_config.tags=[synthval,harmonic]'
```

`enabled.yaml` fields: `project`, `entity`, `name`
(`${hydra:job.override_dirname}`), `group`
(`${oc.select:hydra.sweeper.study_name,${hydra:job.name}}` — clusters sweep
trials), `tags`, `base_url` (self-hosted), and `watch_log` /`watch_log_freq`
(opt-in gradient/parameter histograms). When `enabled: false`, the trainer skips
building a `WandbLogger` entirely.

To make a run W&B-on by default for a preset, set
`experiment(..., wandb_config=...)` — but note the `wandb=` group default
overrides that field, so the CLI group is the reliable switch.
