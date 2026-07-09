"""Shared fixtures for ``test_integration`` — builds small DDSSM models.

Tests in this directory verify mathematical claims from
``model-v2.org`` by training small models for a few hundred steps on
synthetic data.  All tests in this directory should be marked
``slow`` so the fast suite stays fast.
"""

from __future__ import annotations

from types import SimpleNamespace
from functools import partial
from collections.abc import Callable

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
from ddssm.model.centering.baselines import (
    BaseBaseline,
    ZeroBaseline,
    PersistenceBaseline,
)
from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)

DATA_DIM = 1
LATENT_DIM = 4
J = 1
EMB_TIME = 8
T_MAX = 16
CHANNELS = 16
NHEADS = 2


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


def _make_hparams(batch_size: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        S=1,
        ema_decay=0.999,
        weight_decay=1e-2,
        batch_size=batch_size,
        grad_accum_steps=1,
        t_chunk=4,
        enc_lr=1e-3,
        dec_lr=1e-3,
        trans_lr=1e-3,
        logvar_min=-13.0,
        logvar_max=13.0,
    )


def _make_baseline(form: str) -> BaseBaseline:
    """Construct one of the parameter-free baseline forms by name."""
    if form == "zero":
        return ZeroBaseline(latent_dim=LATENT_DIM, j=J)
    if form == "persistence":
        return PersistenceBaseline(latent_dim=LATENT_DIM, j=J)
    raise ValueError(f"Unknown baseline form: {form!r}")


def make_vhp_model(
    *,
    baseline_form: str = "persistence",
    tracking_mode: str = "fixed",
    sigma_data_init: float = 1.0,
) -> DDSSM_base:
    """Build a small DDSSM with the VHP-via-diffusion path wired.

    Post-refactor: baseline forms are parameter-free (``zero`` /
    ``persistence``); there is no baseline mode, no anchor, no stage-1
    Gaussian transition slot on ``DDSSM_base``.
    """
    baseline = _make_baseline(baseline_form)
    aux_posterior = AuxPosterior(
        latent_dim=LATENT_DIM,
        j=J,
        hidden_dim=16,
        n_layers=2,
    )
    sigma_data = SigmaDataBuffer(
        T_max=T_MAX,
        tracking_mode=tracking_mode,
        init_value=sigma_data_init,
    )
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
    return DDSSM_base(
        encoder=_make_encoder(),
        decoder=_make_decoder(),
        transition=transition,
        j=J,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        emb_time_dim=EMB_TIME,
        use_observation_mask=False,
        aux_posterior=aux_posterior,
        baseline=baseline,
        sigma_data=sigma_data,
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
    data_factory: Callable[[], dict[str, torch.Tensor]],
    n_steps: int,
    lr: float = 1e-3,
    stage: str | None = None,
    lambda_mu_p: float = 0.0,
) -> list[dict[str, torch.Tensor]]:
    """Run ``n_steps`` of training in a single phase; return per-step metrics.

    ``stage`` / ``lambda_mu_p`` are accepted for signature back-compat and
    silently ignored: staged training and the R_μp anchor were removed.
    """
    del stage, lambda_mu_p
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    metrics_log = []
    model.train()
    for _ in range(n_steps):
        batch = data_factory()
        optimizer.zero_grad(set_to_none=True)
        components, metrics, _ = model(
            batch["observed_data"],
            batch["observation_mask"],
            batch["timepoints"],
        )
        loss = components.recon + components.init_kl + components.trans_kl
        loss.backward()
        optimizer.step()
        metrics_log.append({k: v.detach().clone() for k, v in metrics.items()})
    return metrics_log


__all__ = [
    "DATA_DIM",
    "EMB_TIME",
    "LATENT_DIM",
    "T_MAX",
    "J",
    "make_random_walk_data",
    "make_smooth_sine_data",
    "make_vhp_model",
    "run_stage",
]
