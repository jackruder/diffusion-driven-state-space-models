"""M5 tests: dssd.py forward psi threading, metrics surfacing, hparam hygiene.

Written RED first — these fail until M5 implementation is complete.
"""

from __future__ import annotations

from functools import partial

import torch
import pytest

from ddssm.model.dssd import DDSSM_base, DDSSMHyperParamsConf

# ---------------------------------------------------------------------------
# Tiny architecture pieces (copied from test_dssd_stage2.py helpers)
# ---------------------------------------------------------------------------

from ddssm.nn.futsum import GRUFutureSummary
from ddssm.nn.fusions import ConcatLinearFusion
from ddssm.nn.diffnets import (
    CSDIUnet,
    ContextProducer,
    FeatureMixerConfig,
    ResidualBlockConfig,
    DiffResidualBlockConfig,
)
from ddssm.nn.combiners import CompoundCombiner
from ddssm.nn.gaussians import GaussianHead
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder
from ddssm.nn.dist_heads import GaussianDistHead
from ddssm.nn.aggregators import ContextProducerAggregator
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.model.centering.baselines import ZeroBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)

J = 2
DATA_DIM = 3
LATENT_DIM = 4
EMB_TIME = 8
CHANNELS = 16
NHEADS = 2
T_MAX = 10


_CTX = partial(
    ContextProducer,
    channels=CHANNELS,
    num_layers=1,
    residual_block=ResidualBlockConfig(
        feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
    ),
)
_AGG = partial(
    ContextProducerAggregator,
    channels=CHANNELS,
    num_layers=1,
    residual_block=ResidualBlockConfig(
        feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
    ),
)
_TINY_UNET = partial(
    CSDIUnet,
    channels=CHANNELS,
    n_layers=1,
    embedding_dim=CHANNELS,
    residual_block=DiffResidualBlockConfig(
        feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
    ),
)


def _make_encoder() -> GaussianEncoder:
    return GaussianEncoder(
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        use_mask=True,
        hidden_dim=CHANNELS,
        combiner=partial(
            CompoundCombiner,
            aggregator=_AGG,
            fusion=partial(ConcatLinearFusion),
        ),
        dist_head=partial(GaussianDistHead),
        fut_summary=partial(GRUFutureSummary, summary_dim=CHANNELS, num_layers=1),
    )


def _make_decoder() -> GaussianDecoder:
    return GaussianDecoder(
        latent_dim=LATENT_DIM,
        data_dim=DATA_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX,
        gaussian_head=GaussianHead,
    )


def _make_diffusion_model() -> DDSSM_base:
    """Build a model with DiffusionTransition (psi is real)."""
    baseline = ZeroBaseline(latent_dim=LATENT_DIM, j=J)
    schedule = DiffusionScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=20,
        k_sampling_mode="uniform",
    )
    transition = DiffusionTransition(
        baseline=baseline,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        T_max=T_MAX,
        unet=_TINY_UNET,
        schedule=schedule,
    )
    aux = AuxPosterior(latent_dim=LATENT_DIM, j=J, hidden_dim=8, n_layers=2)
    sigma_data = SigmaDataBuffer(T_max=T_MAX, tracking_mode="fixed")
    model = DDSSM_base(
        encoder=_make_encoder(),
        decoder=_make_decoder(),
        transition=transition,
        j=J,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        emb_time_dim=EMB_TIME,
        aux_posterior=aux,
        baseline=baseline,
        sigma_data=sigma_data,
    )
    return model


# ``_make_gaussian_model`` was removed — BaselineGaussianTransition is gone
# and there is no longer a separate stage-1 transition slot on DDSSM_base.
# The zero-psi assertion moved to
# tests/test_training/test_split_loss.py::test_score_init_step_zero_psi_for_nondiffusion
# (which still covers the plain GaussianTransition path).


def _make_batch(B: int = 2, T: int = 5) -> dict:
    torch.manual_seed(42)
    return {
        "observed_data": torch.randn(B, DATA_DIM, T),
        "observation_mask": torch.ones(B, DATA_DIM, T),
        "timepoints": torch.arange(T).expand(B, T).clone().long(),
    }


# ---------------------------------------------------------------------------
# test_forward_components_carry_real_psi_values
# ---------------------------------------------------------------------------


def test_forward_components_carry_real_psi_values() -> None:
    """DiffusionTransition forward: trans_kl_psi is real (not a zero placeholder).

    Asserts:
    - trans_kl_psi != trans_kl_phith (they differ because weights differ)
    - trans_kl_psi.requires_grad is True (real gradient path, not detach()*0)
    """
    torch.manual_seed(7)
    model = _make_diffusion_model()
    model.train()
    batch = _make_batch()

    components, metrics, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )

    # The psi and phith values should differ (they use different weighting)
    assert not torch.allclose(
        components.trans_kl_psi, components.trans_kl_phith
    ), (
        f"trans_kl_psi ({components.trans_kl_psi.item():.6f}) should differ from "
        f"trans_kl_phith ({components.trans_kl_phith.item():.6f}) — "
        "if they're equal it likely means psi is still the zero placeholder"
    )

    # The psi side must carry a real gradient path (not detach()*0)
    assert components.trans_kl_psi.requires_grad, (
        "trans_kl_psi.requires_grad must be True (real gradient path from score net)"
    )


# ---------------------------------------------------------------------------
# test_forward_zero_psi_for_nondiffusion_transition
# ---------------------------------------------------------------------------


# ``test_forward_zero_psi_for_nondiffusion_transition`` was removed together
# with ``_make_gaussian_model`` / ``BaselineGaussianTransition``. The
# zero-psi guarantee for plain (non-diffusion) transitions is covered by
# tests/test_training/test_split_loss.py.


# ---------------------------------------------------------------------------
# test_metrics_include_kl_phith_and_kl_psi
# ---------------------------------------------------------------------------


def test_metrics_include_kl_phith_and_kl_psi() -> None:
    """Metrics dict must contain loss/rate/trans/kl_phith and loss/rate/trans/kl_psi."""
    model = _make_diffusion_model()
    model.train()
    batch = _make_batch()

    _, metrics, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )

    assert "loss/rate/trans/kl_phith" in metrics, (
        f"Expected 'loss/rate/trans/kl_phith' in metrics, keys: {list(metrics.keys())}"
    )
    assert "loss/rate/trans/kl_psi" in metrics, (
        f"Expected 'loss/rate/trans/kl_psi' in metrics, keys: {list(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# test_metrics_include_loss_init_psi
# ---------------------------------------------------------------------------


def test_metrics_include_loss_init_psi() -> None:
    """Metrics dict must contain loss/rate/init/loss_psi."""
    model = _make_diffusion_model()
    model.train()
    batch = _make_batch()

    _, metrics, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )

    assert "loss/rate/init/loss_psi" in metrics, (
        f"Expected 'loss/rate/init/loss_psi' in metrics, keys: {list(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# test_hyperparams_have_psi_betas_and_clip_grad_norm
# ---------------------------------------------------------------------------


def test_hyperparams_have_psi_betas_and_clip_grad_norm() -> None:
    """DDSSMHyperParamsConf has psi_betas (default None) and clip_grad_norm
    (default 1.0 — restored alongside the non-finite-grad skip guard)."""
    conf = DDSSMHyperParamsConf()

    # psi_betas must exist with default None
    assert hasattr(conf, "psi_betas"), (
        "DDSSMHyperParamsConf must have field 'psi_betas'"
    )
    assert conf.psi_betas is None, (
        f"psi_betas default must be None, got {conf.psi_betas!r}"
    )

    # clip_grad_norm must be present and default to 1.0
    assert hasattr(conf, "clip_grad_norm"), (
        "DDSSMHyperParamsConf must have field 'clip_grad_norm'"
    )
    assert conf.clip_grad_norm == 1.0, (
        f"clip_grad_norm default must be 1.0, got {conf.clip_grad_norm!r}"
    )
