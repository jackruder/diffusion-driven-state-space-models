"""Module 8a — ``_make.experiment`` wraps DDSSM presets in a ``ModelAdapter``.

These are config-level (no training): they assert the *shape* of the composed
``ExperimentC`` conf, not any runtime behaviour. Coverage:

* a registered DDSSM preset composes to an ``Experiment`` whose ``model`` conf
  targets :class:`~ddssm.adapters.ddssm.DDSSMAdapter` (the wrapped DDSSM factory
  now lives under ``model.module``);
* wrapping curries ``module`` / ``config`` / ``build_trainer`` onto the adapter;
* ``wrap=False`` leaves the model conf unwrapped;
* a *function*-target model conf wraps WITHOUT ``TypeError`` (the
  ``isinstance(t, type)`` guard before ``issubclass``);
* an *adapter*-target model conf gets ``config`` curried rather than
  re-wrapped (the future baseline-adapter path — exercised via a stub
  ``ModelAdapter`` subclass so this test doesn't depend on any live
  baseline family).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import hydra_zen
from hydra_zen import store, get_target

from ddssm.adapters import DDSSMAdapter, ModelAdapter
from ddssm.model.config import ModelConfig
from experiments._make import experiment
from ddssm.data.presets import NonlinBimodalLift1D
from ddssm.experiment.builders import Hparams, Training
from ddssm.experiment.registry import register_experiments
from experiments.init_centering.model import SmokeModel


@dataclass
class _StubConfig(ModelConfig):
    """Minimal ``ModelConfig`` subclass — just enough for wrap-detection tests."""


class _StubAdapter(ModelAdapter):
    """Minimal ``ModelAdapter`` — stubs every abstract method; never runs.

    Only its class identity matters here (the wrap path checks
    ``issubclass(target, ModelAdapter)`` to decide curry-vs-wrap).
    """

    def __init__(self, config: _StubConfig) -> None:
        super().__init__(config)

    @property
    def module(self):  # pragma: no cover - not exercised by these tests
        raise NotImplementedError

    def fit(self, **kw):  # pragma: no cover
        raise NotImplementedError

    def forecast(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError

    def save_checkpoint(self, path):  # pragma: no cover
        raise NotImplementedError

    def load_checkpoint(self, path, **kw):  # pragma: no cover
        raise NotImplementedError


def _smoke_model():
    return SmokeModel(baseline_form="zero", latent_dim=1, data_dim=1)


def _hparams():
    return Hparams(batch_size=16, enc_lr=5e-4, dec_lr=5e-4, trans_lr=5e-4)


def _exp_conf(model=None, *, wrap: bool = True):
    return experiment(
        data=NonlinBimodalLift1D,
        model=_smoke_model() if model is None else model,
        hparams=_hparams(),
        training=Training(steps=5, log_every=1),
        wrap=wrap,
    )


def test_registered_ddssm_preset_wraps_in_adapter() -> None:
    """A registered preset's model conf targets ``DDSSMAdapter`` post-wrap."""
    register_experiments()
    node = store["experiment"]["experiment", "init_smoke_simple"]
    assert get_target(node.model) is DDSSMAdapter
    # The DDSSM factory now lives one hop down, under ``model.config``
    # (post-refactor: adapter takes a DDSSMModelConfig; the factory returns
    # one and is curried into that slot by ``_make.experiment``).
    assert get_target(node.model.config).__name__.endswith(
        "_build_init_centering_model"
    )


def test_wrap_curries_config_and_build_trainer() -> None:
    """The wrapper carries ``config`` / ``build_trainer`` slots."""
    hp = _hparams()
    exp = experiment(
        data=NonlinBimodalLift1D,
        model=_smoke_model(),
        hparams=hp,
        training=Training(steps=5, log_every=1),
    )
    assert get_target(exp.model) is DDSSMAdapter
    # config now holds the wrapped DDSSM factory conf (which resolves to a
    # DDSSMModelConfig at instantiate time); training hparams reach the
    # trainer via ``build_trainer=TrainerPartial(hparams=hp)`` below.
    assert get_target(exp.model.config).__name__.endswith("_build_init_centering_model")
    # build_trainer is a curried TrainerPartial (a DDSSMTrainer partial conf).
    assert exp.model.build_trainer is not None
    assert get_target(exp.model.build_trainer).__name__.endswith("DDSSMTrainer")


def test_wrap_false_leaves_model_conf_unwrapped() -> None:
    """``wrap=False`` skips wrapping — the model conf is the bare factory conf."""
    exp = _exp_conf(wrap=False)
    t = get_target(exp.model)
    assert not (isinstance(t, type) and issubclass(t, ModelAdapter))
    assert t.__name__.endswith("_build_init_centering_model")


def test_function_target_conf_wraps_without_typeerror() -> None:
    """A function-target model conf wraps cleanly (the isinstance guard)."""
    # SmokeModel targets a *function* — a bare issubclass(t, ModelAdapter)
    # would raise TypeError, so the wrap path must guard with isinstance.
    t = get_target(_smoke_model())
    assert not isinstance(t, type)  # precondition: target is a function
    exp = _exp_conf()  # must not raise
    assert get_target(exp.model) is DDSSMAdapter


def test_adapter_target_conf_curries_config_not_rewrapped() -> None:
    """An adapter-target model conf gets ``config`` curried, not re-wrapped."""
    cfg = hydra_zen.builds(_StubConfig, populate_full_signature=True)()
    adapter_conf = hydra_zen.builds(_StubAdapter, populate_full_signature=True)(
        config=cfg
    )
    exp = experiment(
        data=NonlinBimodalLift1D,
        model=adapter_conf,
        hparams=cfg,
        training=Training(steps=5, log_every=1),
    )
    # Still a _StubAdapter (NOT re-wrapped in a DDSSMAdapter).
    assert get_target(exp.model) is _StubAdapter
    # ``config`` curried onto the existing adapter conf.
    assert exp.model.config is cfg


def test_function_target_adapter_detection_no_typeerror() -> None:
    """``_targets_adapter`` on a function-target conf returns False, no raise."""
    from experiments._make import _targets_adapter

    assert _targets_adapter(_smoke_model()) is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
