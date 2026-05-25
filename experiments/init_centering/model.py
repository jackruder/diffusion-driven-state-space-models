"""Composed DDSSM model for the smoke preset.

Builds the model via a factory function so that ``baseline`` and
``aux_posterior`` are constructed *once* and passed by reference to
both the stage-1 :class:`BaselineGaussianTransition` and the stage-2
:class:`DiffusionV3Transition` — ensuring μ_p's parameters are shared
across the handoff (per ``model-v2.org`` § Generative baseline /
§ Stage-1 → stage-2 handoff step 2).

The Small1D encoder/decoder shape from
``experiments.synthetic.model`` is reused unchanged (data_dim=1,
latent_dim=4, j=1, emb_time_dim=16).
"""

from __future__ import annotations

from functools import partial
from typing import Any

from hydra_zen import builds, instantiate
from omegaconf import MISSING

from conf.registry import model_store

from ddssm.aux_posterior import AuxPosterior
from ddssm.centering.baselines import MLPBaseline
from ddssm.centering.sigma_data import SigmaDataBuffer
from ddssm.diffnets import (
    CSDIUnet,
    DiffResidualBlockConfig,
    FeatureMixerConfig,
)
from ddssm.dssd import DDSSM_base
from ddssm.transitions.baseline_gaussian import BaselineGaussianTransition
from ddssm.transitions.diffusion_v3 import (
    DiffusionV3ScheduleConfig,
    DiffusionV3Transition,
)

from experiments.synthetic.model import Small1D


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

DATA_DIM = 1
LATENT_DIM = 4
J = 1
EMB_TIME = 16
T_MAX = 32  # matches Harmonic data's T

CHANNELS = 32
NHEADS = 4


def _build_smoke_model(
    *,
    hyperparams: Any,
    stages: Any,
) -> DDSSM_base:
    """Construct the smoke-preset model with shared baseline + aux instances.

    The factory pattern is required so the *same* Python objects are
    passed to both transitions (so μ_p's parameters are shared and the
    handoff snapshot captures the right state).
    """
    # ---- shared ingredients ----
    baseline = MLPBaseline(
        latent_dim=LATENT_DIM, j=J, hidden_dim=32, n_layers=2,
    )
    aux_posterior = AuxPosterior(
        latent_dim=LATENT_DIM, j=J, hidden_dim=32, n_layers=2,
    )
    sigma_data = SigmaDataBuffer(
        T_max=T_MAX, tracking_mode="fixed", init_value=1.0,
    )

    # ---- stage-1 transition (Gaussian closed-form) ----
    stage1_transition = BaselineGaussianTransition(
        baseline=baseline,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
    )

    # ---- stage-2 transition (centered ESM/EDM) ----
    unet = partial(
        CSDIUnet,
        channels=CHANNELS,
        n_layers=2,
        embedding_dim=CHANNELS,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
        ),
    )
    schedule = DiffusionV3ScheduleConfig(
        S_k=1, k_chunk=1, num_steps=50, k_sampling_mode="uniform",
    )
    stage2_transition = DiffusionV3Transition(
        baseline=baseline,
        latent_dim=LATENT_DIM,
        j=J,
        emb_time_dim=EMB_TIME,
        T_max=T_MAX,
        unet=unet,
        schedule=schedule,
    )

    # ---- encoder / decoder / static slots from Small1D ----
    encoder = instantiate(Small1D.encoder)
    decoder = instantiate(Small1D.decoder)

    return DDSSM_base(
        encoder=encoder,
        decoder=decoder,
        transition=stage2_transition,
        j=J,
        data_dim=DATA_DIM,
        latent_dim=LATENT_DIM,
        emb_time_dim=EMB_TIME,
        hyperparams=hyperparams,
        stages=stages,
        use_observation_mask=False,
        # --- VHP-via-diffusion + baseline-centering ---
        aux_posterior=aux_posterior,
        baseline=baseline,
        baseline_anchor=None,        # populated by the handoff
        baseline_mode="pinned",
        anchor_lambda=0.0,
        sigma_data=sigma_data,
        stage1_transition=stage1_transition,
    )


# hydra-zen wrapper so the smoke preset can plug into experiment(model=...).
SmokeModel = builds(
    _build_smoke_model,
    populate_full_signature=True,
    hyperparams=MISSING,
    stages=MISSING,
)


model_store(SmokeModel, name="init_centering_smoke")

__all__ = ["SmokeModel"]
