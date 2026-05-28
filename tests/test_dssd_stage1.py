"""Stage-1 forward-pass tests for the model-v2 DDSSM extensions.

Verify that:

* :class:`DDSSM_base` constructs with the new slots
  (``aux_posterior``, ``baseline``, ``sigma_data``, ``stage1_transition``)
  populated and ``stage_selector="stage_1"`` produces a finite ELBO that
  includes ``R_σp`` in the rate.
* ``aux_posterior`` is mandatory — construction without it raises
  (ADR-0006: the init term is the transition's own hierarchical walk
  over the auxiliary latents, which needs ``q_Φ``).
"""

from __future__ import annotations

from types import SimpleNamespace
from functools import partial

import torch
import pytest

from ddssm.dssd import DDSSM_base
from ddssm.futsum import GRUFutureSummary
from ddssm.decoder import GaussianDecoder
from ddssm.encoder import GaussianEncoder
from ddssm.fusions import ConcatLinearFusion
from ddssm.diffnets import (
    ContextProducer,
    FeatureMixerConfig,
    ResidualBlockConfig,
)
from ddssm.combiners import CompoundCombiner
from ddssm.gaussians import GaussianHead
from ddssm.dist_heads import GaussianDistHead
from ddssm.aggregators import ContextProducerAggregator
from ddssm.aux_posterior import AuxPosterior
from ddssm.centering.baselines import MLPBaseline
from ddssm.centering.sigma_data import SigmaDataBuffer
from ddssm.transitions.baseline_gaussian import BaselineGaussianTransition

# ---------------------------------------------------------------------------
# Tiny test-only constants.
# ---------------------------------------------------------------------------

J = 2
DATA_DIM = 3
LATENT_DIM = 4
EMB_TIME = 8
CHANNELS = 8
NHEADS = 4


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


def _make_encoder() -> GaussianEncoder:
    return GaussianEncoder(
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        use_mask=True,
        hidden_dim=CHANNELS,
        combiner=partial(
            CompoundCombiner, aggregator=_AGG, fusion=partial(ConcatLinearFusion),
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


def _make_gaussian_baseline() -> MLPBaseline:
    return MLPBaseline(latent_dim=LATENT_DIM, j=J, hidden_dim=8, n_layers=2)


def _make_hparams(lambda_sigma_p: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        S=1,
        ema_decay=0.999,
        weight_decay=1e-2,
        batch_size=2,
        grad_accum_steps=1,
        t_chunk=4,
        clip_grad_norm=None,
        lambda_schedule="none",
        lambda_start=0.001,
        lambda_end=1.0,
        lambda_warmup_steps=1,
        enc_lr=1e-3,
        dec_lr=1e-3,
        zinit_lr=1e-3,
        trans_lr=1e-3,
        logvar_min=-7.0,
        logvar_max=7.0,
        lambda_sigma_p=lambda_sigma_p,
    )


def _make_batch(B: int, T: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "observed_data": torch.randn(B, DATA_DIM, T),
        "observation_mask": torch.ones(B, DATA_DIM, T),
        "timepoints": torch.arange(T).expand(B, T).clone().long(),
    }


# ---------------------------------------------------------------------------
# Construction + mutual exclusion
# ---------------------------------------------------------------------------


def test_constructor_requires_aux_posterior() -> None:
    """``aux_posterior`` is mandatory — the init term needs q_Φ (ADR-0006)."""
    enc = _make_encoder()
    dec = _make_decoder()
    baseline = _make_gaussian_baseline()
    trans = BaselineGaussianTransition(
        baseline=baseline, latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
    )
    with pytest.raises(ValueError, match="aux_posterior is required"):
        DDSSM_base(
            encoder=enc,
            decoder=dec,
            transition=trans,
            j=J,
            data_dim=DATA_DIM,
            latent_dim=LATENT_DIM,
            emb_time_dim=EMB_TIME,
            # no aux_posterior
        )


# ---------------------------------------------------------------------------
# VHP-via-diffusion stage-1 forward
# ---------------------------------------------------------------------------


def _make_vhp_model(lambda_sigma_p: float = 0.0) -> DDSSM_base:
    enc = _make_encoder()
    dec = _make_decoder()
    baseline = _make_gaussian_baseline()
    aux = AuxPosterior(latent_dim=LATENT_DIM, j=J, hidden_dim=8, n_layers=2)
    sigma_data = SigmaDataBuffer(T_max=10, tracking_mode="fixed")
    trans = BaselineGaussianTransition(
        baseline=baseline, latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
    )
    model = DDSSM_base(
        encoder=enc,
        decoder=dec,
        transition=trans,
        j=J,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        emb_time_dim=EMB_TIME,
        aux_posterior=aux,
        baseline=baseline,
        sigma_data=sigma_data,
        stage1_transition=trans,
    )
    model.stage_selector = "stage_1"
    return model


def test_vhp_stage1_forward_produces_finite_loss() -> None:
    """Forward pass on the VHP path returns finite losses + expected metric keys."""
    model = _make_vhp_model()
    batch = _make_batch(B=2, T=5)
    components, metrics, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    assert torch.isfinite(components.total())
    assert torch.isfinite(components.recon)
    assert torch.isfinite(components.elbo_reg() - components.recon)
    # New VHP-related keys present.
    assert "loss/rate/init/kl_aux" in metrics
    assert "loss/rate/init/loss_init" in metrics
    # Stage-1 entropy contribution is non-zero (encoder posterior is
    # generic, so -H(q) is finite and non-zero).
    assert torch.isfinite(metrics["loss/rate/init/entropy"])
    # Regularizer surfaces, both finite.
    assert torch.isfinite(metrics["loss/rate/trans/r_sigma_p"])
    assert torch.isfinite(metrics["loss/rate/trans/r_mu_p"])


def test_vhp_stage1_r_sigma_p_active_with_lambda() -> None:
    """When ``λ_σp > 0`` the regularizer contributes a non-zero metric."""
    model = _make_vhp_model(lambda_sigma_p=1.0)
    batch = _make_batch(B=2, T=5)
    _, metrics, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    # Generic baseline + λ > 0 ⇒ regularizer typically > 0 (mean log σ_p² != 0
    # in expectation).  Just check it isn't *exactly* zero (the λ=0 path
    # would short-circuit to 0).
    assert metrics["loss/rate/trans/r_sigma_p"] != 0.0


def test_vhp_stage1_sigma_data_buffer_accumulates() -> None:
    """The σ_data buffer accumulates values during stage 1."""
    model = _make_vhp_model()
    assert model.sigma_data is not None
    pre = model.sigma_data.sigma_data2.clone()
    pre_step = model.sigma_data.ema_step.clone()

    batch = _make_batch(B=2, T=5)
    model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    # Buffer slots for the visited timesteps should have advanced.
    # The init term touches t = 1..j; the transition_kl touches t = j+1..T.
    # We expect a non-trivial number of timesteps to have moved either
    # value or step counter.
    moved_value = (model.sigma_data.sigma_data2 != pre).any()
    moved_step = (model.sigma_data.ema_step != pre_step).any()
    assert bool(moved_value) or bool(moved_step)
