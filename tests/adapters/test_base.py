"""Unit tests for the ``ModelAdapter`` ABC + ``MetricNotSupported``.

These are the only adapter tests that must run and pass at module 4 —
no concrete adapter exists yet, so they exercise the abstract surface
directly via a trivial in-test subclass.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
import pytest

import ddssm.adapters as adapters_pkg
from ddssm.adapters import ModelAdapter, MetricNotSupported
from ddssm.model.config import ModelConfig

if TYPE_CHECKING:
    from ddssm.data.datamodule import TimeSeriesDataModule
    from ddssm.experiment.experiment import TrainingScalars


class _StubAdapter(ModelAdapter):
    """Minimal concrete adapter: implements every abstractmethod as a stub.

    Enough to prove the ABC is instantiable once its contract is filled,
    and to exercise the inherited ``log_prob`` gating behaviour. Signatures
    mirror the ABC (see ``base.py``); bodies are inert.
    """

    @property
    def module(self) -> torch.nn.Module | None:
        return None

    def fit(
        self,
        *,
        data: TimeSeriesDataModule,
        training: TrainingScalars,
        device: torch.device,
        csv_log_path: str,
        tensorboard_dir: str,
        checkpoint_dir: str,
        hparams: ModelConfig | None = None,
        wandb_config: dict | None = None,
        model_config_yaml: str | None = None,
    ) -> None:
        pass

    def forecast(
        self,
        x_hist: torch.Tensor,
        x_mask: torch.Tensor,
        past_time: torch.Tensor,
        future_time: torch.Tensor,
        past_covariates: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
        static_covariates: torch.Tensor | None,
        *,
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        return {}

    def save_checkpoint(self, path: str) -> None:
        pass

    def load_checkpoint(
        self,
        path: str,
        *,
        device: torch.device,
        hparams: ModelConfig | None = None,
        load_ema: bool = True,
        expected_model_config_yaml: str | None = None,
        strict: bool = False,
    ) -> None:
        pass


def test_abc_cannot_be_instantiated_directly() -> None:
    """``ModelAdapter`` is abstract — direct construction must fail."""
    with pytest.raises(TypeError):
        ModelAdapter(ModelConfig())  # type: ignore[abstract]


def test_is_abstract_base_class() -> None:
    """``ModelAdapter`` derives from ``abc.ABC`` and has abstractmethods."""
    assert issubclass(ModelAdapter, abc.ABC)
    assert getattr(ModelAdapter, "__abstractmethods__", None)


def test_concrete_subclass_instantiates_and_stores_config() -> None:
    """A fully-implemented subclass builds and keeps ``config`` on ``.config``."""
    cfg = ModelConfig(batch_size=7)
    adapter = _StubAdapter(cfg)
    assert adapter.config is cfg
    assert adapter.config.batch_size == 7


def test_base_log_prob_raises_metric_not_supported_naming_subclass() -> None:
    """The inherited ``log_prob`` raises ``MetricNotSupported`` naming the class."""
    adapter = _StubAdapter(ModelConfig())
    with pytest.raises(MetricNotSupported) as excinfo:
        adapter.log_prob()
    # Message must name the concrete class, not the ABC.
    assert _StubAdapter.__name__ in str(excinfo.value)


def test_metric_not_supported_subclasses_not_implemented_error() -> None:
    """``MetricNotSupported`` is a narrow subtype of ``NotImplementedError``."""
    assert issubclass(MetricNotSupported, NotImplementedError)


def test_metric_not_supported_importable_from_package() -> None:
    """``MetricNotSupported`` is reachable straight off ``ddssm.adapters``."""
    # Redundant with the module-level import, but pins the public surface.
    assert adapters_pkg.MetricNotSupported is MetricNotSupported
    assert "MetricNotSupported" in adapters_pkg.__all__
