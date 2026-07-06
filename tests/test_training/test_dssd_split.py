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
from ddssm.model.centering.baselines import MLPBaseline
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)
from ddssm.model.transitions.baseline_gaussian import BaselineGaussianTransition

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
    """Build a stage-2 model with DiffusionTransition (psi is real)."""
    baseline = MLPBaseline(latent_dim=LATENT_DIM, j=J, hidden_dim=8, n_layers=2)
    schedule = DiffusionScheduleConfig(
        S_k=1,
        k_chunk=1,
        num_steps=20,
        k_sampling_mode="uniform",
    )
    stage1_transition = BaselineGaussianTransition(
        baseline=baseline,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
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
        stage1_transition=stage1_transition,
    )
    model.stage_selector = "stage_2"
    return model


def _make_gaussian_model() -> DDSSM_base:
    """Build a stage-1 model with BaselineGaussianTransition (psi = 0)."""
    baseline = MLPBaseline(latent_dim=LATENT_DIM, j=J, hidden_dim=8, n_layers=2)
    transition = BaselineGaussianTransition(
        baseline=baseline,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
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
        stage1_transition=transition,
    )
    model.stage_selector = "stage_1"
    return model


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


def test_forward_zero_psi_for_nondiffusion_transition() -> None:
    """Non-diffusion transition: trans_kl_psi == 0 and init_kl_psi == 0."""
    model = _make_gaussian_model()
    model.train()
    batch = _make_batch()

    components, _, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )

    assert components.trans_kl_psi.item() == pytest.approx(0.0, abs=1e-9), (
        f"Expected trans_kl_psi == 0.0 for Gaussian transition, "
        f"got {components.trans_kl_psi.item()}"
    )
    assert components.init_kl_psi.item() == pytest.approx(0.0, abs=1e-9), (
        f"Expected init_kl_psi == 0.0 for Gaussian transition, "
        f"got {components.init_kl_psi.item()}"
    )


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
# test_hyperparams_have_psi_betas_and_no_clip_grad_norm
# ---------------------------------------------------------------------------


def test_hyperparams_have_psi_betas_and_no_clip_grad_norm() -> None:
    """DDSSMHyperParamsConf has psi_betas (default None); clip_grad_norm is gone."""
    conf = DDSSMHyperParamsConf()

    # psi_betas must exist with default None
    assert hasattr(conf, "psi_betas"), (
        "DDSSMHyperParamsConf must have field 'psi_betas'"
    )
    assert conf.psi_betas is None, (
        f"psi_betas default must be None, got {conf.psi_betas!r}"
    )

    # clip_grad_norm must be gone
    assert not hasattr(conf, "clip_grad_norm"), (
        "DDSSMHyperParamsConf must NOT have field 'clip_grad_norm' (M5 removes it)"
    )
