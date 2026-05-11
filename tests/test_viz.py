"""Tests for the viz stage: registry, CSV plot, runner."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch
import pytest

from ddssm.viz import PLOT_REGISTRY, VizSpec, PlotSpec, PlotContext, visualize
from ddssm.viz.plots import plot_metrics_csv


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_registry_has_core_plots():
    for name in (
        "forecast_1d",
        "forecast_2d_spatial",
        "metrics_csv",
        "forecast_distribution",
    ):
        assert name in PLOT_REGISTRY


def test_metrics_csv_writes_png(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    rows = [
        {
            "step": str(i),
            "loss/total": str(1.0 / (i + 1)),
            "loss/recon": str(0.5 / (i + 1)),
        }
        for i in range(20)
    ]
    _write_csv(csv_path, rows)

    out = tmp_path / "out.png"
    ctx = PlotContext(
        model=None, loader=None, device=torch.device("cpu"), csv_path=str(csv_path)
    )
    plot_metrics_csv(ctx, str(out), keys=["loss/total", "loss/recon"])
    assert out.is_file() and out.stat().st_size > 0


def test_metrics_csv_raises_when_no_csv():
    ctx = PlotContext(
        model=None, loader=None, device=torch.device("cpu"), csv_path=None
    )
    with pytest.raises(ValueError):
        plot_metrics_csv(ctx, "/tmp/x.png")


def test_visualize_runner_smoke(tmp_path):
    """A VizSpec containing only the CSV plot needs no model and no loader."""
    csv_path = tmp_path / "m.csv"
    _write_csv(
        csv_path,
        [{"step": "0", "loss/total": "1.0"}, {"step": "1", "loss/total": "0.5"}],
    )

    class _StubData:
        batch_transform = staticmethod(lambda b, d: b)
        metadata = type("_M", (), {"forecast_split": None})()

        def train_loader(self):
            return None

        def val_loader(self):
            return None

        def test_loader(self):
            return None

    class _StubExpt:
        data = _StubData()
        model = torch.nn.Linear(1, 1)

    spec = VizSpec(
        plots=[
            PlotSpec(
                name="metrics_csv",
                save_filename="curves.png",
                kwargs={"keys": ["loss/total"]},
            )
        ],
        split="train",
    )
    saved = visualize(
        _StubExpt(),
        spec,
        device=torch.device("cpu"),
        run_dir=str(tmp_path),
        checkpoint_path=None,
        csv_path=str(csv_path),
    )
    assert len(saved) == 1
    assert Path(saved[0]).is_file()


def test_visualize_runner_unknown_plot_raises(tmp_path):
    class _StubData:
        batch_transform = staticmethod(lambda b, d: b)
        metadata = type("_M", (), {"forecast_split": None})()

        def train_loader(self):
            return None

        def val_loader(self):
            return None

        def test_loader(self):
            return None

    class _StubExpt:
        data = _StubData()
        model = torch.nn.Linear(1, 1)

    spec = VizSpec(plots=[PlotSpec(name="nope")], split="train")
    with pytest.raises(KeyError):
        visualize(_StubExpt(), spec, device=torch.device("cpu"), run_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# forecast_distribution plot: stub model + minimal loader → produces a PNG.
# ---------------------------------------------------------------------------


class _StubReconForecastModel(torch.nn.Module):
    """Stub providing both ``__call__`` (recon path) and ``.forecast()``."""

    emb_time_dim = 4
    j = 1
    data_dim = 1

    class _Decoder:
        def __call__(self, z_hist, time_embed, t_idx):
            B = z_hist.shape[0]
            return torch.zeros(B, 1), torch.zeros(B, 1)

    decoder = _Decoder()

    def __init__(self):
        super().__init__()

    def __call__(self, observed, mask, timepoints, train: bool = False):
        B, D, T = observed.shape
        zs = torch.zeros(B, 1, 2, T)  # (B, S=1, d=2, T)
        return None, None, None, None, {"zs": zs}

    def forecast(self, *, x_hist, x_mask, past_time, future_time, num_samples, **_):
        B = x_hist.shape[0]
        L2 = future_time.shape[1]
        # Bimodal: half samples at +1, half at -1
        s = (torch.randint(0, 2, (B, num_samples, 1, L2)).float() * 2.0) - 1.0
        return {"pred_samples": s, "pred_mean": torch.zeros(B, 1, L2)}


def test_forecast_distribution_writes_png(tmp_path):
    from ddssm.viz.plots import plot_forecast_distribution

    B, T = 2, 8
    obs = torch.zeros(B, 1, T)
    obs[:, 0, T - 1] = 1.5  # last past value
    obs[:, 0, T // 2 :] = 0.5  # synthetic future
    mask = torch.ones_like(obs)
    timepoints = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, -1)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return B

        def __getitem__(self, i):
            return {
                "observed_data": obs[i],
                "observation_mask": mask[i],
                "timepoints": timepoints[i],
            }

    loader = torch.utils.data.DataLoader(_DS(), batch_size=B)
    ctx = PlotContext(
        model=_StubReconForecastModel(),
        loader=loader,
        device=torch.device("cpu"),
        T_split=T // 2,
        num_samples=64,
    )

    out = tmp_path / "dist.png"
    plot_forecast_distribution(ctx, str(out), series_idx=0, dim_idx=0, t_future_idx=0)
    assert out.is_file() and out.stat().st_size > 0


def test_forecast_distribution_raises_without_t_split():
    from ddssm.viz.plots import plot_forecast_distribution

    ctx = PlotContext(model=None, loader=None, device=torch.device("cpu"), T_split=None)
    with pytest.raises(ValueError):
        plot_forecast_distribution(ctx, "/tmp/x.png")
