"""Composed DDSSM model for the init-centering preset family.

Builds the model via a factory function so that ``baseline`` and
``aux_posterior`` are constructed once and passed by reference to the
:class:`DiffusionTransition` (per ``model-v2.org`` § Generative
baseline).

The factory is parametric over the two remaining ablation-grid axes
from ``init-experiment.org`` § Composition with the ablation grid:

* ``baseline_form`` ∈ {zero, persistence}   — both parameter-free
* ``tracking_mode`` ∈ {fixed, global_ema, per_t}

Default values reproduce the canonical cell — persistence / per-t EMA.
"""

from __future__ import annotations

from typing import Literal
from functools import partial

from hydra_zen import builds

from ddssm.model.dssd import DDSSM_base
from ddssm.nn.diffnets import (
    CSDIUnet,
    FeatureMixerConfig,
    DiffResidualBlockConfig,
)
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.experiment.stores import model_store
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


def _build_baseline(
    *,
    baseline_form: str,
    latent_dim: int,
    j: int,
) -> BaseBaseline:
    """Construct the parameter-free baseline (μ_p) head."""
    if baseline_form == "zero":
        return ZeroBaseline(latent_dim=latent_dim, j=j)
    if baseline_form == "persistence":
        return PersistenceBaseline(latent_dim=latent_dim, j=j)
    raise ValueError(
        f"baseline_form must be one of (zero, persistence); got {baseline_form!r}"
    )


def _build_init_centering_model(
    *,
    # --- Cell axes ---
    baseline_form: Literal["zero", "persistence"] = "persistence",
    tracking_mode: Literal["fixed", "global_ema", "per_t"] = "per_t",
    # --- Shape ---
    j: int = 1,
    T_max: int = 32,
    data_dim: int = 1,
    latent_dim: int = 4,
    use_time_embedding: bool = False,
    emb_time_dim: int = 16,
    channels: int | None = None,
    encoder_hidden_dim: int | None = None,
    decoder_hidden_dim: int | None = None,
    aux_posterior_hidden_dim: int | None = None,
    aux_posterior_n_layers: int = 2,
    sigma_data_ema_decay: float = 0.997,
    # --- Diffusion schedule ---
    diffusion_S_k: int = 1,
    diffusion_k_chunk: int = 1,
    diffusion_num_steps: int = 128,
    diffusion_layers: int = 2,
) -> DDSSM_base:
    """Construct an init-centering DDSSM model parametric over the ablation grid."""
    if not use_time_embedding:
        emb_time_dim = 0

    if channels is None:
        channels = 16 * latent_dim
    if encoder_hidden_dim is None:
        encoder_hidden_dim = 16 * latent_dim
    if decoder_hidden_dim is None:
        decoder_hidden_dim = 16 * latent_dim
    if aux_posterior_hidden_dim is None:
        aux_posterior_hidden_dim = 16 * latent_dim

    baseline = _build_baseline(
        baseline_form=baseline_form,
        latent_dim=latent_dim,
        j=j,
    )
    aux_posterior = AuxPosterior(
        latent_dim=latent_dim,
        j=j,
        hidden_dim=aux_posterior_hidden_dim,
        n_layers=aux_posterior_n_layers,
    )
    sigma_data = SigmaDataBuffer(
        T_max=T_max,
        tracking_mode=tracking_mode,
        init_value=1.0,
        ema_decay=sigma_data_ema_decay,
    )

    unet = partial(
        CSDIUnet,
        channels=channels,
        n_layers=diffusion_layers,
        embedding_dim=channels,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(type="conv", n_layers=1)
        ),
    )
    schedule = DiffusionScheduleConfig(
        S_k=diffusion_S_k,
        k_chunk=diffusion_k_chunk,
        num_steps=diffusion_num_steps,
    )
    transition = DiffusionTransition(
        baseline=baseline,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        T_max=T_max,
        unet=unet,
        schedule=schedule,
    )

    encoder = GaussianEncoder(
        data_dim=data_dim,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        use_mask=False,
        hidden_dim=encoder_hidden_dim,
    )
    decoder = GaussianDecoder(
        data_dim=data_dim,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        hidden_dim=decoder_hidden_dim,
    )

    return DDSSM_base(
        encoder=encoder,
        decoder=decoder,
        transition=transition,
        j=j,
        data_dim=data_dim,
        latent_dim=latent_dim,
        emb_time_dim=emb_time_dim,
        use_observation_mask=False,
        aux_posterior=aux_posterior,
        baseline=baseline,
        sigma_data=sigma_data,
    )


SmokeModel = builds(
    _build_init_centering_model,
    populate_full_signature=True,
)


model_store(SmokeModel, name="init_centering_smoke")

__all__ = ["SmokeModel", "_build_init_centering_model"]
