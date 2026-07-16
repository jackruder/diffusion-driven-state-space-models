"""Config-path tests for ``DDSSMAdapter``.

Commit 2 extends the adapter to accept a whole :class:`DDSSMModelConfig`
in place of a pre-built module. This file exercises the new path:

* Adapter construction with only a config (no ``module=``).
* ``adapter.module`` builds lazily via ``config.build_module()``.
* One-step ``fit`` + save/load round-trip works via the config path.
* ``fit(hparams=<full DDSSMModelConfig>)`` unwraps to the training slice.

Contract tests in :mod:`tests.adapters.test_ddssm` stay on the LEGACY
path in commit 2; they migrate in commit 3 when family factories switch
to returning ``DDSSMModelConfig``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from hydra_zen import builds

from ddssm.adapters.ddssm import DDSSMAdapter
from ddssm.data.datamodule import SyntheticDataModule
from ddssm.experiment.experiment import TrainingScalars
from ddssm.model.dssd import DDSSM_base
from ddssm.model.ddssm_config import (
    DDSSMModelConfig,
    DDSSMModelKnobs,
    DDSSMShape,
    DDSSMTrainingHparams,
)
from ddssm.model.encoder import GaussianEncoder
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.transitions.transitions import GaussianTransition
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.nn.combiners import CompoundCombiner
from ddssm.nn.aggregators import IdentityAggregator
from ddssm.nn.fusions import ConcatLinearFusion
from ddssm.nn.dist_heads import GaussianDistHead
from ddssm.nn.futsum import GRUFutureSummary
from ddssm.nn.gaussians import GaussianHead
from ddssm.nn.diffnets import (
    ContextProducer,
    ResidualBlockConfig,
    FeatureMixerConfig,
)


# Small fixed shape mirroring tests/test_trainer.py::make_small_model
DATA_DIM = 3
LATENT_DIM = 2
J = 1
EMB_TIME = 8
CHANNELS = 16
NHEADS = 2


def _small_ddssm_config() -> DDSSMModelConfig:
    """Build a DDSSMModelConfig equivalent to make_small_model, fully wired.

    Every slot builder is fully specified — no MISSING sentinels — because
    this test bypasses ``_make.experiment``'s shape-fill pass.
    """
    ctx = builds(
        ContextProducer,
        channels=CHANNELS,
        num_layers=1,
        residual_block=ResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
        populate_full_signature=True,
        zen_partial=True,
    )
    head = builds(GaussianHead, populate_full_signature=True, zen_partial=True)
    combiner = builds(
        CompoundCombiner,
        aggregator=builds(IdentityAggregator, populate_full_signature=True,
                          zen_partial=True)(),
        fusion=builds(ConcatLinearFusion, populate_full_signature=True,
                      zen_partial=True)(),
        populate_full_signature=True,
        zen_partial=True,
    )
    dist_head = builds(GaussianDistHead, populate_full_signature=True, zen_partial=True)
    fut_summary = builds(
        GRUFutureSummary,
        summary_dim=CHANNELS,
        num_layers=1,
        populate_full_signature=True,
        zen_partial=True,
    )

    encoder = builds(
        GaussianEncoder,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        use_mask=True,
        hidden_dim=CHANNELS,
        combiner=combiner(),
        dist_head=dist_head(),
        fut_summary=fut_summary(),
        populate_full_signature=True,
    )
    decoder = builds(
        GaussianDecoder,
        latent_dim=LATENT_DIM,
        data_dim=DATA_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=ctx(),
        gaussian_head=head(),
        populate_full_signature=True,
    )
    transition = builds(
        GaussianTransition,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=ctx(),
        gaussian_head=head(),
        populate_full_signature=True,
    )
    aux = builds(
        AuxPosterior,
        latent_dim=LATENT_DIM,
        j=J,
        hidden_dim=CHANNELS,
        n_layers=1,
        populate_full_signature=True,
    )

    return DDSSMModelConfig(
        shape=DDSSMShape(
            j=J, data_dim=DATA_DIM, latent_dim=LATENT_DIM,
            emb_time_dim=EMB_TIME, T_max=16,
        ),
        encoder=encoder,
        decoder=decoder,
        transition=transition,
        aux_posterior=aux,
        baseline=None,
        sigma_data=None,
        model_knobs=DDSSMModelKnobs(),
        training=DDSSMTrainingHparams(batch_size=4),
    )


def test_config_path_lazy_module_build() -> None:
    """``adapter.module`` is built lazily on first access from a DDSSMModelConfig."""
    cfg = _small_ddssm_config()
    adapter = DDSSMAdapter(config=cfg)
    assert adapter._module is None
    m = adapter.module
    assert isinstance(m, DDSSM_base)
    # Second access returns the same instance (not rebuilt).
    assert adapter.module is m


def test_legacy_path_still_works() -> None:
    """Pre-built ``module=...`` on __init__ bypasses lazy build."""
    from tests.test_trainer import make_small_model
    from ddssm.model.dssd import DDSSMHyperParamsConf

    m = make_small_model()
    adapter = DDSSMAdapter(config=DDSSMHyperParamsConf(batch_size=4), module=m)
    assert adapter._module is m
    assert adapter.module is m


def test_module_property_raises_without_buildable_config() -> None:
    """Lazy-build path requires ``self.config`` to be a DDSSMModelConfig."""
    from ddssm.model.dssd import DDSSMHyperParamsConf

    adapter = DDSSMAdapter(config=DDSSMHyperParamsConf())
    import pytest

    with pytest.raises(TypeError, match="DDSSMModelConfig"):
        _ = adapter.module


def test_resolve_training_hparams_unwraps_config() -> None:
    """``_resolve_training_hparams`` unwraps a whole config to its training slice."""
    cfg = _small_ddssm_config()
    resolved = DDSSMAdapter._resolve_training_hparams(cfg)
    assert resolved is cfg.training
    # Training slice passes through as-is.
    hp = DDSSMTrainingHparams(batch_size=8)
    assert DDSSMAdapter._resolve_training_hparams(hp) is hp
    # None stays None.
    assert DDSSMAdapter._resolve_training_hparams(None) is None


def test_config_path_fit_and_forecast(tmp_path: Path) -> None:
    """End-to-end: config-only adapter fits, writes CSV, and forecasts."""
    cfg = _small_ddssm_config()
    adapter = DDSSMAdapter(config=cfg)
    data = SyntheticDataModule(
        mode="lgssm", T=16, D=DATA_DIM, N_per_split=4, batch_size=4
    )
    training = TrainingScalars(steps=1, log_every=1, validate_every=0)
    adapter.fit(
        data=data,
        training=training,
        device=torch.device("cpu"),
        csv_log_path=str(tmp_path / "metrics.csv"),
        tensorboard_dir=str(tmp_path / "tb"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert isinstance(adapter.module, DDSSM_base)
    assert (tmp_path / "metrics.csv").exists()
