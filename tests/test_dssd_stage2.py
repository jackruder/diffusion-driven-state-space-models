"""Stage-2 forward-pass tests for the model-v2 DDSSM extensions.

Verify that:

* :class:`DDSSM_base` forward at ``stage_selector="stage_2"`` runs
  through :class:`DiffusionTransition` end-to-end (centered ESM/EDM
  loss + σ_data buffer updates + VHP init term + ``R_μp`` under the
  Learnable baseline-mode).
* Stage-2 ``L_init`` omits the encoder-entropy term per
  ``model-v2.org`` § Entropy cancellation in stage 2.
* The ``r_mu_p`` regularizer is exactly 0 immediately after the
  ``baseline_anchor`` snapshot (handoff invariant) and non-zero after
  the baseline parameters move.
"""

from __future__ import annotations

from types import SimpleNamespace
from functools import partial

import torch

from ddssm.nn.futsum import GRUFutureSummary
from ddssm.model.dssd import DDSSM_base
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


def _make_hparams() -> SimpleNamespace:
    return SimpleNamespace(
        S=1,
        ema_decay=0.999,
        weight_decay=1e-2,
        batch_size=2,
        grad_accum_steps=1,
        t_chunk=4,
        clip_grad_norm=None,
        enc_lr=1e-3,
        dec_lr=1e-3,
        trans_lr=1e-3,
        logvar_min=-7.0,
        logvar_max=7.0,
    )


def _make_batch(B: int, T: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "observed_data": torch.randn(B, DATA_DIM, T),
        "observation_mask": torch.ones(B, DATA_DIM, T),
        "timepoints": torch.arange(T).expand(B, T).clone().long(),
    }


def _make_stage2_model(
    *,
    baseline_mode: str = "pinned",
    snapshot_anchor: bool = False,
) -> DDSSM_base:
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
    anchor = baseline.snapshot() if snapshot_anchor else None
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
        baseline_anchor=anchor,
        baseline_mode=baseline_mode,
        sigma_data=sigma_data,
        stage1_transition=stage1_transition,
    )
    model.stage_selector = "stage_2"
    return model


def test_stage2_forward_finite() -> None:
    """End-to-end stage-2 forward returns finite losses + expected metrics."""
    model = _make_stage2_model()
    batch = _make_batch(B=2, T=5)
    components, metrics, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    assert torch.isfinite(components.total())
    assert torch.isfinite(components.recon)
    assert torch.isfinite(components.elbo_reg() - components.recon)
    assert "loss/rate/init/kl_aux" in metrics
    assert "loss/rate/init/loss_init" in metrics
    assert "loss/rate/trans/r_mu_p" in metrics


def test_stage2_entropy_term_is_zero() -> None:
    """Stage-2 entropy term cancels per § Entropy cancellation in stage 2."""
    model = _make_stage2_model()
    batch = _make_batch(B=2, T=5)
    _, metrics, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    # stage_selector == "stage_2" causes _init_kl_loss to skip the
    # explicit -H(q_φ) term (it cancels with the ESM expansion).
    assert float(metrics["loss/rate/init/entropy"].item()) == 0.0


def test_stage2_r_mu_p_zero_after_snapshot() -> None:
    """``R_μp = 0`` immediately after the baseline_anchor snapshot."""
    model = _make_stage2_model(
        baseline_mode="learnable",
        snapshot_anchor=True,
    )
    batch = _make_batch(B=2, T=5)
    _, metrics, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    assert torch.isclose(
        metrics["loss/rate/trans/r_mu_p"], torch.tensor(0.0), atol=1e-6
    )


def test_stage2_r_mu_p_positive_after_drift() -> None:
    """After the baseline drifts from the anchor, ``R_μp > 0``."""
    model = _make_stage2_model(
        baseline_mode="learnable",
        snapshot_anchor=True,
    )
    # Drift the baseline.
    with torch.no_grad():
        for p in model.baseline.parameters():
            p.data.add_(0.5)
    batch = _make_batch(B=2, T=5)
    _, metrics, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    assert float(metrics["loss/rate/trans/r_mu_p"].item()) > 0.0


def test_stage2_r_mu_p_zero_under_pinned_mode() -> None:
    """Under ``"pinned"`` mode, ``R_μp`` is always 0 regardless of drift."""
    model = _make_stage2_model(
        baseline_mode="pinned",
        snapshot_anchor=True,
    )
    with torch.no_grad():
        for p in model.baseline.parameters():
            p.data.add_(0.5)
    batch = _make_batch(B=2, T=5)
    _, metrics, _ = model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    assert float(metrics["loss/rate/trans/r_mu_p"].item()) == 0.0


def test_stage2_sigma_data_buffer_updates_across_init_and_transition() -> None:
    """Stage-2 forward updates the σ_data buffer at t = 1..T_used."""
    model = _make_stage2_model()
    assert model.sigma_data is not None
    pre_step = model.sigma_data.ema_step.clone()
    batch = _make_batch(B=2, T=5)
    model(
        batch["observed_data"],
        batch["observation_mask"],
        batch["timepoints"],
    )
    # ema_step zeros only at frozen slots; "fixed" mode is not frozen
    # yet (we only freeze on reset_schedule), so updates should fire.
    # In Phase 4 the sigma_data buffer is still in "fixed" tracking_mode
    # with frozen=False (we haven't called reset_schedule), so updates
    # do happen.  Slots 1..T_used (here 1..5) should have advanced
    # ema_step.
    expected = torch.zeros(T_MAX, dtype=torch.bool)
    expected[:5] = True
    advanced = model.sigma_data.ema_step > pre_step
    assert torch.equal(advanced, expected)
