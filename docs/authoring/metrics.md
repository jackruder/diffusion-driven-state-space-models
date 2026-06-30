# Step 4 — Metrics & the objective

Two related things live here: the **eval stage** (post-training metrics written
to `metrics.json`) and the **objective** (the scalar an Optuna sweep optimizes).

## The eval stage

{py:class}`~ddssm.eval.runner.EvalSpec` (`src/ddssm/eval/runner.py`) names a list
of registered metrics, the `split`, `num_samples` (for sample-based metrics),
`T_split` (forecast split), and per-metric `kwargs`. Attach it via the `Eval`
builder:

```python
from ddssm.experiment.builders import Eval
exp = experiment(..., eval=Eval(metrics=["mae", "crps_sum", "stage2_elbo_surrogate"],
                                split="val", num_samples=16))
```

It runs standalone (loads a checkpoint; no training) via the CLI, writing
`metrics.json`:

```bash
python -m ddssm.evaluate experiment=synthval__harmonic \
    +checkpoint='${experiment.checkpoint_dir}/ckpt_latest.pth'
```

## The metric registry

`src/ddssm/eval/metrics.py` holds `METRIC_REGISTRY`. Available metrics:

| Name | What it measures |
| ---- | ---------------- |
| `mae` | mean abs error of the forecast mean (+ per-t) |
| `crps_sum` | channel-summed CRPS over forecast samples (+ per-t) |
| `energy_score` | multivariate energy score over samples |
| `recon_mse` | MSE of decoded posterior mean vs observed (noisy) sequence |
| `denoise_mse` | MSE of decoded posterior mean vs the clean (noise-free) sequence; needs `expose_clean_data=True`; soft-skips when absent |
| `regime` | metastable regime-switching quality: residence accuracy, first-switch timing, and residence-time climatology JSD — parameterized by `channel`, `threshold`, `deadband` so it works for any system (Lorenz, double-well, etc.) |
| `stage2_elbo_surrogate` | held-out ELBO surrogate (rate + distortion, unweighted) |
| `nll` | marginal `−log p(x)` via prob-flow ODE + IWAE |
| `crps_sum_latent` / `gt_latent_jsd` | latent-space CRPS / JSD vs ground-truth (needs `expose_gt_latents`) |
| `loss_tail` | tail-mean of a training CSV column (no model needed) |
| `wallclock_to_target` / `wallclock_to_relative_target` | step/seconds to hit a (relative) threshold |
| `sigma_data_drift`, `q_aux_kl_trajectory`, `log_sigma_p2_collapse` | centering diagnostics |
| `bimodal_jsd` | one-step JSD vs analytic bimodal kernel |

## Adding a custom metric

Register a function that takes the `EvalContext` (model, loader, device,
`csv_path`, `T_split`, `num_samples`, `run_dir`) and returns a JSON-serializable
dict:

```python
from ddssm.eval.metrics import register_metric

@register_metric("my_metric")
def eval_my_metric(ctx, *, threshold=0.0):
    if ctx.model is None or ctx.loader is None:
        return {"my_metric_available": False}
    ...
    return {"my_metric": value}
```

Then reference it in `EvalSpec.metrics=["my_metric"]` and pass options via
`EvalSpec.kwargs={"my_metric": {"threshold": 1.0}}`.

## The objective

{py:class}`~ddssm.experiment.experiment.ObjectiveSpec` defines the Optuna
objective read after training:

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `metric` | `loss/total` | column/key to optimize |
| `split` | `train` | CSV split filter |
| `tail_frac` | 0.1 | average the final fraction of values |
| `source` | `csv` | read `metrics.csv` (`csv`) or `metrics.json` (`json`, runs eval first) |
| `penalty` | `inf` | fallback when the metric is missing (`inf` / `csv_tail_time` / `csv_tail_step`) |

Wire one (or several for multi-objective) via the `Objective` / `Objectives`
builders:

```python
from ddssm.experiment.builders import Objective, Objectives

# single-objective
exp = experiment(..., objective=Objective(metric="loss/total", split="train", source="csv"))

# multi-objective (paired with hydra.sweeper.direction=[...]); see init_centering/evals.py
exp = experiment(..., objective=Objectives(specs=[
    Objective(metric="wallclock_to_target_step", source="json", penalty="csv_tail_step"),
    Objective(metric="stage2_elbo_surrogate", source="json"),
]))
```

`Experiment.objective_value()` returns the scalar (or list); if any spec reads
`json` it runs `evaluate()` first. With no objective set, `train()` just returns
the trainer.

## In the worked example

Each `synthval__*` preset (built in `study.py`'s `_build`) wires
`eval=Eval(metrics=["mae", "crps_sum", "stage2_elbo_surrogate"], split="val")`
and a simple `objective=Objective(metric="loss/total", split="train",
source="csv")` — enough to rank a dataset cell by its tail training ELBO and to
emit forecast-accuracy metrics on demand.
