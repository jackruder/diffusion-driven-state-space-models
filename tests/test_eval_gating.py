"""Eval/viz metric gating behind ``MetricNotSupported`` (module #9).

A forecast-only :class:`ModelAdapter` (no ``log_prob`` / recon / ELBO) must
make the eval and viz runners *gracefully skip* DDSSM-only metrics/plots
rather than crash with ``AttributeError``. Forecast-based metrics/plots
(``mae`` / ``rmse`` / ``forecast_1d`` …) still run on any adapter. A deep
bare ``NotImplementedError`` (a load-bearing signal inside DDSSM internals)
must still PROPAGATE — only the narrow ``MetricNotSupported`` subtype is
swallowed.
"""

from __future__ import annotations

import json
import logging

import torch
import pytest

from ddssm.viz import VizSpec, PlotSpec, visualize
from ddssm.eval import EvalSpec, evaluate
from ddssm.adapters import ModelAdapter
from ddssm.variance import ProbeSpec, variance
from ddssm.model.config import ModelConfig
from ddssm.eval.metrics import METRIC_REGISTRY
from ddssm.adapters.base import MetricNotSupported
from ddssm.data.datamodule import DataMetadata, TimeSeriesDataModule


class _ForecastOnlyModule(torch.nn.Module):
    """A plain ``nn.Module`` that is NOT a ``DDSSM_base`` but can forecast."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def to(self, *a, **k):  # keep the runner's ``.to(device)`` a no-op
        return self


class _ForecastOnlyAdapter(ModelAdapter):
    """Forecast-only adapter: real ``forecast``; inherits the base ``log_prob`` raise.

    Its ``.module`` is a plain ``nn.Module`` (NOT a ``DDSSM_base``), so every
    ``require_module(DDSSM_base)`` gate must raise ``MetricNotSupported``.
    """

    def __init__(self) -> None:
        super().__init__(ModelConfig())
        self._module = _ForecastOnlyModule()

    @property
    def module(self) -> torch.nn.Module:
        return self._module

    def fit(self, *a, **k) -> None:  # pragma: no cover - unused here
        return None

    def forecast(
        self,
        x_hist,
        x_mask,
        past_time,
        future_time,
        past_covariates=None,
        future_covariates=None,
        static_covariates=None,
        *,
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        B, D, _ = x_hist.shape
        L2 = int(future_time.shape[-1])
        pred_mean = x_hist.new_zeros(B, D, L2)
        pred_samples = x_hist.new_zeros(B, num_samples, D, L2)
        return {"pred_mean": pred_mean, "pred_samples": pred_samples}

    def save_checkpoint(self, path: str) -> None:  # pragma: no cover - unused
        return None

    def load_checkpoint(self, path: str, **k) -> None:  # pragma: no cover - unused
        return None


class _ForecastData(TimeSeriesDataModule):
    """One-batch windowed loader so forecast metrics have real tensors."""

    batch_transform = staticmethod(lambda b, d: b)
    metadata = DataMetadata(data_dim=1, forecast_split=4)

    def __init__(self) -> None:
        B, D, T = 2, 1, 8
        obs = torch.randn(B, D, T)
        mask = torch.ones(B, D, T)
        tp = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, -1)

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return B

            def __getitem__(self, i):
                return {
                    "observed_data": obs[i],
                    "observation_mask": mask[i],
                    "timepoints": tp[i],
                }

        self._loader = torch.utils.data.DataLoader(_DS(), batch_size=B)

    def train_loader(self):
        return self._loader

    def val_loader(self):
        return self._loader

    def test_loader(self):
        return self._loader


class _ForecastExpt:
    def __init__(self) -> None:
        self.data = _ForecastData()
        self.model = _ForecastOnlyAdapter()


@pytest.fixture(autouse=True)
def _bypass_prepare_model(monkeypatch):
    """``prepare_model`` normally loads a checkpoint; return the adapter as-is."""

    def _fake(experiment, *, checkpoint_path=None, device=None, **k):
        return experiment.model

    monkeypatch.setattr("ddssm.training.checkpoint.prepare_model", _fake)


# ---------------------------------------------------------------------------
# Eval gating.
# ---------------------------------------------------------------------------


def test_forecast_metric_runs_ddssm_metric_skipped(tmp_path, caplog):
    """``rmse`` survives; the DDSSM-only ``recon_mse`` is omitted with a WARNING."""
    spec = EvalSpec(
        metrics=["rmse", "recon_mse"],
        split="test",
        num_samples=3,
        output_filename="m.json",
    )
    with caplog.at_level(logging.WARNING):
        out = evaluate(
            _ForecastExpt(),
            spec,
            device=torch.device("cpu"),
            run_dir=str(tmp_path),
            checkpoint_path=None,
        )

    assert any(k.startswith("rmse") for k in out), out
    assert "recon_mse" not in out
    assert not any("recon" in k for k in out)

    written = json.loads((tmp_path / "m.json").read_text())
    assert written == out

    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("recon_mse" in r.getMessage() for r in warned), (
        "expected a WARNING naming the skipped metric"
    )


def test_stage2_elbo_surrogate_skipped_on_forecast_only(tmp_path):
    """``stage2_elbo_surrogate`` (calls ``model(...)``) is gated + omitted."""
    spec = EvalSpec(
        metrics=["mae", "stage2_elbo_surrogate"],
        split="test",
        output_filename="m.json",
    )
    out = evaluate(
        _ForecastExpt(),
        spec,
        device=torch.device("cpu"),
        run_dir=str(tmp_path),
        checkpoint_path=None,
    )
    assert any(k.startswith("mae") for k in out)
    assert "stage2_elbo_surrogate" not in out


def test_deep_not_implemented_still_propagates(tmp_path, monkeypatch):
    """A bare ``NotImplementedError`` (deep DDSSM signal) is NOT swallowed."""

    def _boom(ctx, **k):
        raise NotImplementedError("deep load-bearing signal")

    monkeypatch.setitem(METRIC_REGISTRY, "rmse", _boom)

    spec = EvalSpec(metrics=["rmse"], split="test", output_filename="m.json")
    with pytest.raises(NotImplementedError):
        evaluate(
            _ForecastExpt(),
            spec,
            device=torch.device("cpu"),
            run_dir=str(tmp_path),
            checkpoint_path=None,
        )


def test_metric_not_supported_is_caught_but_not_bare(tmp_path, monkeypatch):
    """The runner catches ``MetricNotSupported`` (subclass) but not the base."""

    def _skip(ctx, **k):
        raise MetricNotSupported("gated")

    monkeypatch.setitem(METRIC_REGISTRY, "mae", _skip)

    spec = EvalSpec(metrics=["mae"], split="test", output_filename="m.json")
    out = evaluate(
        _ForecastExpt(),
        spec,
        device=torch.device("cpu"),
        run_dir=str(tmp_path),
        checkpoint_path=None,
    )
    assert out == {}


# ---------------------------------------------------------------------------
# Viz gating.
# ---------------------------------------------------------------------------


def test_ddssm_plot_skipped_forecast_plot_stays(tmp_path):
    """``forecast_1d`` (recon path) is gated; ``metrics_csv`` still renders."""
    import csv as _csv

    csv_path = tmp_path / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["step", "loss/total"])
        w.writeheader()
        for i in range(5):
            w.writerow({"step": i, "loss/total": 1.0 - 0.1 * i})

    spec = VizSpec(
        plots=[
            PlotSpec(name="forecast_1d", save_filename="fc.png"),
            PlotSpec(
                name="metrics_csv",
                save_filename="curve.png",
                kwargs={"keys": ["loss/total"]},
            ),
        ],
        split="test",
    )
    saved = visualize(
        _ForecastExpt(),
        spec,
        device=torch.device("cpu"),
        run_dir=str(tmp_path),
        checkpoint_path=None,
        csv_path=str(csv_path),
    )
    # The DDSSM-only forecast_1d is skipped (not saved); metrics_csv is saved.
    assert not (tmp_path / "fc.png").exists()
    assert (tmp_path / "curve.png").exists()
    assert any(p.endswith("curve.png") for p in saved)
    assert not any(p.endswith("fc.png") for p in saved)


# ---------------------------------------------------------------------------
# Variance gating: DDSSM-only stage → a loud early error, not a silent skip.
# ---------------------------------------------------------------------------


def test_variance_raises_metric_not_supported_on_forecast_only(tmp_path):
    """The DDSSM-only variance runner rejects a non-DDSSM adapter clearly."""
    spec = ProbeSpec(cells=[], metrics=[], plots=[])
    with pytest.raises(MetricNotSupported):
        variance(
            _ForecastExpt(),
            spec,
            device=torch.device("cpu"),
            run_dir=str(tmp_path),
            checkpoint_path=None,
        )
