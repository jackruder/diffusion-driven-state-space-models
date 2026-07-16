"""A small, hand-written DDSSM model factory for the authoring guide.

This is the worked example for ``docs/authoring/model.md``: it builds a
:class:`~ddssm.model.dssd.DDSSM_base` from the underlying runtime classes
**without** going through ``experiments.init_centering.model.SmokeModel``.

The architecture here is deliberately minimal and fixed:

* baseline: ``PersistenceBaseline`` by default (``μ_p = z_{t-1}``, parameter-
  free ``σ_p² = 1``). Override with ``baseline_form="zero"`` for the
  ``ZeroBaseline`` behaviour.
* σ_data²: :class:`SigmaDataBuffer` using the library default (``"per_t"``).
* transition: centered diffusion with a small conv-mixer ``CSDIUnet``.
* encoder/decoder: :class:`GaussianEncoder` / :class:`GaussianDecoder`.

See ``docs/authoring/model.md`` for the menu of alternatives.
"""

from __future__ import annotations

from typing import Literal
from functools import partial

from hydra_zen import builds

from ddssm.model.dssd import DDSSM_base  # kept for backcompat
from ddssm.model.ddssm_config import (
    DDSSMModelConfig,
    DDSSMModelKnobs,
    DDSSMShape,
    DDSSMTrainingHparams,
)
from ddssm.nn.diffnets import CSDIUnet, FeatureMixerConfig, DiffResidualBlockConfig
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder
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


def _make_baseline(form: str, *, latent_dim: int, j: int) -> BaseBaseline:
    if form == "zero":
        return ZeroBaseline(latent_dim=latent_dim, j=j)
    if form == "persistence":
        return PersistenceBaseline(latent_dim=latent_dim, j=j)
    raise ValueError(
        f"baseline_form must be one of zero/persistence; got {form!r}"
    )


def build_synthval_model(
    *,
    data_dim: int = 1,
    latent_dim: int = 1,
    j: int = 1,
    use_time_embedding: bool = False,
    emb_time_dim: int = 16,
    T_max: int = 32,
    hidden_dim: int = 32,
    channels: int = 32,
    diffusion_layers: int = 2,
    diffusion_num_steps: int = 64,
    diffusion_k_sampling_mode: str = "adaptive_is",
    diffusion_S_k: int = 1,
    diffusion_time_chunk_size: int | None = None,
    recon_time_chunk: int | None = None,
    baseline_form: Literal["zero", "persistence"] = "persistence",
    # Training slice curried by ``_make.experiment`` (single source of truth).
    training: DDSSMTrainingHparams | None = None,
) -> DDSSMModelConfig:
    """Compose a minimal DDSSM model from runtime parts."""
    if not use_time_embedding:
        emb_time_dim = 0
    baseline = _make_baseline(baseline_form, latent_dim=latent_dim, j=j)
    aux_posterior = AuxPosterior(
        latent_dim=latent_dim, j=j, hidden_dim=hidden_dim, n_layers=2
    )
    sigma_data = SigmaDataBuffer(T_max=T_max, init_value=1.0)

    unet = partial(
        CSDIUnet,
        channels=channels,
        n_layers=diffusion_layers,
        embedding_dim=channels,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(type="conv", n_layers=1)
        ),
    )
    transition = DiffusionTransition(
        baseline=baseline,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        T_max=T_max,
        unet=unet,
        schedule=DiffusionScheduleConfig(
            num_steps=diffusion_num_steps,
            k_sampling_mode=diffusion_k_sampling_mode,
            S_k=diffusion_S_k,
            time_chunk_size=diffusion_time_chunk_size,
        ),
    )

    encoder = GaussianEncoder(
        data_dim=data_dim,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        use_mask=False,
        hidden_dim=hidden_dim,
    )
    decoder = GaussianDecoder(
        data_dim=data_dim,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        hidden_dim=hidden_dim,
    )

    return DDSSMModelConfig(
        shape=DDSSMShape(
            j=j,
            data_dim=data_dim,
            latent_dim=latent_dim,
            emb_time_dim=emb_time_dim,
            use_observation_mask=False,
            T_max=T_max,
        ),
        encoder=encoder,
        decoder=decoder,
        transition=transition,
        aux_posterior=aux_posterior,
        baseline=baseline,
        sigma_data=sigma_data,
        model_knobs=DDSSMModelKnobs(recon_time_chunk=recon_time_chunk),
        training=training if training is not None else DDSSMTrainingHparams(),
    )


SynthValModel = builds(build_synthval_model, populate_full_signature=True)

__all__ = ["SynthValModel", "build_synthval_model"]
