"""Contract + unit coverage for :class:`ddssm.adapters.ddssm.DDSSMAdapter`.

Lights up the five family-agnostic checks in
:class:`tests.adapters.contract.ModelAdapterContract` against a real (tiny)
``DDSSM_base`` module, plus a handful of DDSSM-specific unit assertions
(``log_prob`` delegation, ``module`` identity, the "fit hasn't run" guard on
``save_checkpoint``, and the explicit cross-format ``ValueError`` guard).

The small ``DDSSM_base`` fixture is cribbed verbatim from
``tests/test_trainer.py::make_small_model`` (j=1, DATA_DIM=3, LATENT_DIM=2);
``make_data`` uses ``SyntheticDataModule(mode="lgssm", T=16, D=2, ...)`` shape
from ``tests/test_trainer.py`` but with ``D=3`` so it matches the model's
``DATA_DIM``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import pytest

from ddssm.model.ddssm_config import (
    DDSSMModelConfig,
    DDSSMModelKnobs,
    DDSSMShape,
    DDSSMTrainingHparams,
)
from tests.test_trainer import DATA_DIM, EMB_TIME, J, LATENT_DIM, make_small_model
from ddssm.adapters.ddssm import DDSSMAdapter
from ddssm.data.datamodule import SyntheticDataModule
from tests.adapters.contract import ModelAdapterContract


def _small_ddssm_config() -> DDSSMModelConfig:
    """Decompose ``make_small_model()`` into a :class:`DDSSMModelConfig`.

    Under the config-path adapter, tests must hand the adapter a config that
    can build a module — not a pre-built module. We keep reusing the tiny
    ``make_small_model`` sub-modules by lifting them out and re-wrapping in
    the config's slots (the config accepts runtime ``nn.Module`` instances
    just as well as ``builds()`` confs — family factories rely on this).
    """
    m = make_small_model()
    return DDSSMModelConfig(
        shape=DDSSMShape(
            j=J,
            data_dim=DATA_DIM,
            latent_dim=LATENT_DIM,
            emb_time_dim=EMB_TIME,
            T_max=16,
        ),
        encoder=m.encoder,
        decoder=m.decoder,
        transition=m.transition,
        aux_posterior=m.aux_posterior,
        baseline=getattr(m, "baseline", None),
        sigma_data=getattr(m, "sigma_data", None),
        model_knobs=DDSSMModelKnobs(),
        training=DDSSMTrainingHparams(batch_size=4),
    )

if TYPE_CHECKING:
    from pathlib import Path

    from ddssm.data.datamodule import TimeSeriesDataModule

# History length carved out of each T=16 sequence for the forecast checks.
_HIST_LEN = 12

# Deterministic seed for the stochastic-forecast round-trip (check b): the
# reloaded module has bit-identical weights, but forecast SAMPLES, so equality
# of pred_mean only holds when the RNG stream is pinned before each rollout.
_FORECAST_SEED = 20260715


class TestDDSSMAdapterContract(ModelAdapterContract):
    """Run the shared ABC contract against a real tiny ``DDSSMAdapter``."""

    def make_adapter(self) -> DDSSMAdapter:
        """Fresh, unfitted adapter driven by a :class:`DDSSMModelConfig`."""
        return DDSSMAdapter(config=_small_ddssm_config())

    def make_data(self) -> SyntheticDataModule:
        """Small LGSSM data module with real train/val splits (D matches model)."""
        return SyntheticDataModule(
            mode="lgssm",
            T=16,
            D=DATA_DIM,
            N_per_split=8,
            batch_size=4,
        )

    def _forecast_inputs(self, data: TimeSeriesDataModule) -> dict:
        """Split one test batch into a history window + future horizon."""
        batch = next(iter(data.test_loader()))
        batch = data.batch_transform(batch, torch.device("cpu"))
        obs = batch["observed_data"]
        mask = batch["observation_mask"]
        tp = batch["timepoints"]
        return {
            "x_hist": obs[..., :_HIST_LEN],
            "x_mask": mask[..., :_HIST_LEN],
            "past_time": tp[:, :_HIST_LEN],
            "future_time": tp[:, _HIST_LEN:],
            "past_covariates": None,
            "future_covariates": None,
            "static_covariates": None,
        }

    def test_checkpoint_roundtrip_preserves_forecast(self, tmp_path: Path) -> None:
        """(b) Override: seed the RNG so the stochastic rollout is comparable.

        DDSSM ``forecast`` samples; identical weights only yield an identical
        ``pred_mean`` when the noise stream is pinned before each rollout, so we
        ``torch.manual_seed`` immediately before each call. ``load_ema=False``
        keeps the reloaded module on the checkpoint's live weights — the same
        weights the original adapter forecasts under — rather than swapping to
        the EMA shadows (which, after a 2-step fit, still differ from live).
        Both modules are put in ``eval()`` first: ``fit`` leaves the trained
        module in eval mode (post-validation) while a freshly-built one defaults
        to train mode, and that mode difference alone changes the rollout.
        """
        adapter = self.make_adapter()
        data = self.make_data()
        self._fit(adapter, data, tmp_path)
        batch = self._forecast_inputs(data)

        adapter.module.eval()
        torch.manual_seed(_FORECAST_SEED)
        before = adapter.forecast(**batch, num_samples=4)

        ckpt = str(tmp_path / "adapter.pth")
        adapter.save_checkpoint(ckpt)
        reloaded = self.make_adapter()
        reloaded.load_checkpoint(ckpt, device=torch.device("cpu"), load_ema=False)

        reloaded.module.eval()
        torch.manual_seed(_FORECAST_SEED)
        after = reloaded.forecast(**batch, num_samples=4)

        torch.testing.assert_close(before["pred_mean"], after["pred_mean"])


# --------------------------------------------------------------------------
# DDSSM-specific unit checks (beyond the shared contract).
# --------------------------------------------------------------------------


def _make_adapter() -> DDSSMAdapter:
    return DDSSMAdapter(config=_small_ddssm_config())


def test_module_property_returns_ddssm_base() -> None:
    """``module`` builds a ``DDSSM_base`` lazily from the config."""
    from ddssm.model.dssd import DDSSM_base

    adapter = _make_adapter()
    m = adapter.module
    assert isinstance(m, DDSSM_base)
    # Cached: second access returns the same instance.
    assert adapter.module is m


def test_log_prob_delegates_to_module() -> None:
    """``log_prob`` forwards to the module (overriding the base ABC raise)."""
    adapter = _make_adapter()
    module = adapter.module

    sentinel = torch.tensor([1.23, 4.56])
    calls: dict[str, object] = {}

    def _fake_log_prob(*args: object, **kwargs: object) -> torch.Tensor:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return sentinel

    module.log_prob = _fake_log_prob  # type: ignore[assignment]
    out = adapter.log_prob(1, foo="bar")
    assert out is sentinel
    assert calls["args"] == (1,)
    assert calls["kwargs"] == {"foo": "bar"}


def test_save_checkpoint_before_fit_raises_runtime_error(tmp_path: Path) -> None:
    """``save_checkpoint`` before ``fit`` raises ``RuntimeError`` (no trainer)."""
    adapter = _make_adapter()
    with pytest.raises(RuntimeError):
        adapter.save_checkpoint(str(tmp_path / "never.pth"))


def test_forecast_forwards_extra_sampling_kwargs() -> None:
    """``forecast`` forwards DDSSM-only kwonly sampling knobs to the module."""
    adapter = _make_adapter()
    module = adapter.module

    seen: dict[str, object] = {}

    def _fake_forecast(**kwargs: object) -> dict[str, torch.Tensor]:
        seen.update(kwargs)
        return {"pred_mean": torch.zeros(1), "pred_samples": torch.zeros(1)}

    module.forecast = _fake_forecast  # type: ignore[assignment]
    adapter.forecast(
        x_hist=torch.zeros(1),
        x_mask=torch.zeros(1),
        past_time=torch.zeros(1),
        future_time=torch.zeros(1),
        past_covariates=None,
        future_covariates=None,
        static_covariates=None,
        num_samples=4,
        s_churn=0.5,
    )
    assert seen["num_samples"] == 4
    assert seen["s_churn"] == pytest.approx(0.5)


def test_cross_format_checkpoint_raises_value_error(tmp_path: Path) -> None:
    """A non-DDSSM payload (unknown ``_format``) must raise ``ValueError``.

    Distinct from the contract's ``{"__foreign_format__": True}`` (a dict
    with no ``model_state`` — treated by ``Checkpoint.load`` as a legacy raw
    state_dict): here we also pin the explicit ``_format`` guard for a payload
    that *looks* like a checkpoint but carries a foreign format tag.
    """
    bogus = tmp_path / "foreign.pth"
    torch.save({"_format": "not_ddssm_v9", "model_state": {}}, bogus)
    adapter = _make_adapter()
    with pytest.raises(ValueError):
        adapter.load_checkpoint(str(bogus), device=torch.device("cpu"))
