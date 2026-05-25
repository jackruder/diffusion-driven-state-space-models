"""Shared fixtures for ``test_integration`` — builds small DDSSM models.

Tests in this directory verify mathematical claims from
``model-v2.org`` by training small models for a few hundred steps on
synthetic data.  All tests in this directory should be marked
``slow`` so the fast suite stays fast.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn as nn

from ddssm.aggregators import ContextProducerAggregator
from ddssm.aux_posterior import AuxPosterior
from ddssm.centering.baselines import (
    BaseBaseline,
    IdentityBaseline,
    LinearBaseline,
    MLPBaseline,
    ZeroBaseline,
)
from ddssm.centering.sigma_data import SigmaDataBuffer
from ddssm.combiners import CompoundCombiner
from ddssm.decoder import GaussianDecoder
from ddssm.diffnets import (
    ContextProducer,
    CSDIUnet,
    DiffResidualBlockConfig,
    FeatureMixerConfig,
    ResidualBlockConfig,
)
from ddssm.dist_heads import GaussianDistHead
from ddssm.dssd import DDSSM_base
from ddssm.encoder import GaussianEncoder
from ddssm.fusions import ConcatLinearFusion
from ddssm.futsum import GRUFutureSummary
from ddssm.gaussians import GaussianHead
from ddssm.transitions.baseline_gaussian import BaselineGaussianTransition
from ddssm.transitions.diffusion_v3 import (
    DiffusionV3ScheduleConfig,
    DiffusionV3Transition,
)


DATA_DIM = 1
LATENT_DIM = 4
J = 1
EMB_TIME = 8
T_MAX = 16
CHANNELS = 16
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
        use_mask=False,
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


def _make_hparams(lambda_sigma_p: float = 1e-2, batch_size: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        S=1,
        ema_decay=0.999,
        weight_decay=1e-2,
        batch_size=batch_size,
        grad_accum_steps=1,
        t_chunk=4,
        clip_grad_norm=None,
        lambda_schedule="none",
        lambda_start=1.0,
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


def _make_baseline(form: str) -> BaseBaseline:
    """Construct one of the four baseline forms by name."""
    if form == "zero":
        return ZeroBaseline(latent_dim=LATENT_DIM, j=J)
    if form == "identity":
        return IdentityBaseline(latent_dim=LATENT_DIM, j=J)
    if form == "linear":
        return LinearBaseline(latent_dim=LATENT_DIM, j=J)
    if form == "mlp":
        return MLPBaseline(latent_dim=LATENT_DIM, j=J, hidden_dim=16, n_layers=2)
    raise ValueError(f"Unknown baseline form: {form!r}")


def make_vhp_model(
    *,
    baseline_form: str = "mlp",
    baseline_mode: str = "pinned",
    anchor_lambda: float = 0.0,
    tracking_mode: str = "fixed",
    lambda_sigma_p: float = 1e-2,
    sigma_data_init: float = 1.0,
    snapshot_anchor: bool = False,
) -> DDSSM_base:
    """Build a small DDSSM with the VHP-via-diffusion path wired."""
    baseline = _make_baseline(baseline_form)
    aux_posterior = AuxPosterior(
        latent_dim=LATENT_DIM, j=J, hidden_dim=16, n_layers=2,
    )
    sigma_data = SigmaDataBuffer(
        T_max=T_MAX,
        tracking_mode=tracking_mode,
        init_value=sigma_data_init,
    )
    schedule = DiffusionV3ScheduleConfig(
        S_k=1, k_chunk=1, num_steps=20, k_sampling_mode="uniform",
    )
    stage1_transition = BaselineGaussianTransition(
        baseline=baseline,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
    )
    stage2_transition = DiffusionV3Transition(
        baseline=baseline,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        T_max=T_MAX,
        unet=_TINY_UNET,
        schedule=schedule,
    )
    anchor = baseline.snapshot() if snapshot_anchor else None
    return DDSSM_base(
        encoder=_make_encoder(),
        decoder=_make_decoder(),
        transition=stage2_transition,
        j=J,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        emb_time_dim=EMB_TIME,
        hyperparams=_make_hparams(lambda_sigma_p=lambda_sigma_p),
        use_observation_mask=False,
        aux_posterior=aux_posterior,
        baseline=baseline,
        baseline_anchor=anchor,
        baseline_mode=baseline_mode,
        anchor_lambda=anchor_lambda,
        sigma_data=sigma_data,
        stage1_transition=stage1_transition,
    )


def make_random_walk_data(
    *,
    n_seqs: int = 32,
    T: int = 8,
    noise_std: float = 0.3,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Generate a Gaussian random walk: x_t = x_{t-1} + N(0, noise_std²)."""
    torch.manual_seed(seed)
    eps = noise_std * torch.randn(n_seqs, DATA_DIM, T)
    x = torch.cumsum(eps, dim=-1)
    return {
        "observed_data": x,
        "observation_mask": torch.ones(n_seqs, DATA_DIM, T),
        "timepoints": torch.arange(T).expand(n_seqs, T).clone().long(),
    }


def make_smooth_sine_data(
    *,
    n_seqs: int = 32,
    T: int = 8,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Smooth deterministic time series x_t = sin(2π·t / T + φ)."""
    torch.manual_seed(seed)
    phase = 2 * torch.pi * torch.rand(n_seqs, 1, 1)
    t = torch.arange(T, dtype=torch.float32).view(1, 1, T)
    x = torch.sin(2 * torch.pi * t / T + phase)
    return {
        "observed_data": x,
        "observation_mask": torch.ones(n_seqs, DATA_DIM, T),
        "timepoints": torch.arange(T).expand(n_seqs, T).clone().long(),
    }


def run_stage(
    *,
    model: DDSSM_base,
    stage: str,
    data_factory: Callable[[], dict[str, torch.Tensor]],
    n_steps: int,
    lr: float = 1e-3,
) -> list[dict[str, torch.Tensor]]:
    """Run ``n_steps`` of training in the given stage; return per-step metrics."""
    model.stage_selector = stage
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    metrics_log = []
    model.train()
    for _ in range(n_steps):
        batch = data_factory()
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, metrics, _ = model(
            batch["observed_data"],
            batch["observation_mask"],
            batch["timepoints"],
        )
        loss.backward()
        optimizer.step()
        metrics_log.append({k: v.detach().clone() for k, v in metrics.items()})
    return metrics_log


__all__ = [
    "DATA_DIM",
    "EMB_TIME",
    "J",
    "LATENT_DIM",
    "T_MAX",
    "make_random_walk_data",
    "make_smooth_sine_data",
    "make_vhp_model",
    "run_stage",
]
