# Step 6 — Configuring visualization

The viz stage renders forecast and diagnostic plots from a checkpoint. Like
eval, it loads a checkpoint and does **not** train.

## `VizSpec` and the `Viz` builder

{py:class}`~ddssm.viz.runner.VizSpec` (`src/ddssm/viz/runner.py`) holds a list of
`PlotSpec` (each: a registry `name`, `save_filename`, and `kwargs`), plus
`split`, `num_samples`, `T_split`. Attach it with the `Viz` / `Plot` builders:

```python
from ddssm.experiment.builders import Viz, Plot

exp = experiment(..., viz=Viz(
    plots=[
        Plot(name="forecast_1d", save_filename="forecast.png", kwargs={"n_show": 4}),
        Plot(name="metrics_csv", save_filename="loss.png",
             kwargs={"keys": ["loss/total", "loss/distortion/rec"]}),
    ],
    split="test", num_samples=10, T_split=16,
))
```

Run it standalone (writes PNGs to the run dir):

```bash
python -m ddssm.visualize experiment=synthval__harmonic \
    +checkpoint='${experiment.checkpoint_dir}/ckpt_latest.pth'
```

## The plot registry

`src/ddssm/viz/plots.py` holds `PLOT_REGISTRY`:

| Name | Plot |
| ---- | ---- |
| `forecast_1d` | per-sample observed / reconstruction / forecast samples + mean |
| `forecast_2d_spatial` | X-vs-Y trajectory (for `D≥2`), optional obstacle box |
| `forecast_distribution` | histogram of forecast samples at one `(series, dim, t)` |
| `metrics_csv` | training-curve line plots from `metrics.csv` (needs `csv_path`) |

## Adding a custom plot

Register a function taking the `PlotContext` (model, loader, device, `csv_path`,
`T_split`, `num_samples`) and a `save_path`:

```python
from ddssm.viz.plots import register_plot

@register_plot("my_plot")
def plot_my_plot(ctx, save_path, *, n_show=4):
    if ctx.model is None or ctx.loader is None:
        raise ValueError("my_plot needs a model and loader.")
    # helpers: _gather_batch(ctx, ...) and _run_recon_and_forecast(ctx, ...)
    ...
    plt.savefig(save_path); plt.close()
```

Then reference `Plot(name="my_plot", kwargs={"n_show": 8})`.

## Wiring status

Every shipped preset (and `synthetic_validation`) currently ships `viz=None` —
the viz CLI only does something against a preset you've given a `VizSpec`. To add
forecast plots to the worked example, set `viz=Viz(plots=[Plot(name="forecast_1d")])`
on the `experiment(...)` call in `experiments/synthetic_validation/experiments.py`.

```{note}
A related standalone stage is the **variance probe**
({py:class}`~ddssm.variance.runner.ProbeSpec`, `python -m ddssm.variance`), which
measures gradient/loss variance across diffusion-step sampling modes. It has its
own metric/plot registries and is wired via `experiment(..., variance=...)`.
```
