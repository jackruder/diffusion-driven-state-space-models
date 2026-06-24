"""Composed DDSSM model for the init-centering preset family.

Builds the model via a factory function so that ``baseline`` and
``aux_posterior`` are constructed *once* and passed by reference to
both the stage-1 :class:`BaselineGaussianTransition` and the stage-2
:class:`DiffusionTransition` — ensuring μ_p's parameters are shared
across the handoff (per ``model-v2.org`` § Generative baseline /
§ Stage-1 → stage-2 handoff step 2).

The factory is parametric over the three ablation-grid axes from
``init-experiment.org`` § Composition with the ablation grid:

* ``baseline_form`` ∈ {zero, persistence, linear, mlp}
* ``baseline_mode`` ∈ {pinned, learnable}  (auto-clamped to pinned
  for the parameter-free zero/persistence forms)
* ``tracking_mode`` ∈ {fixed, global_ema, per_t}

Default values reproduce the canonical cell from
``init-experiment.org:275`` — MLP / Pinned / per-t EMA.
"""

from __future__ import annotations

from typing import Any, Literal
import logging
from functools import partial

from hydra_zen import builds
from omegaconf import MISSING

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
    MLPBaseline,
    BaseBaseline,
    ZeroBaseline,
    LinearBaseline,
    PersistenceBaseline,
)
from ddssm.model.centering.sigma_data import SigmaDataBuffer

# Forms that have no learnable μ_p parameters and therefore degenerate to
# ``baseline_mode="pinned"`` regardless of user input. Single source of truth
# lives in ``cells.py`` (the grid module) so the auto-clamp here and the
# cell-enumeration there can't drift.
from experiments.init_centering.cells import _PARAM_FREE_FORMS
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)
from ddssm.model.transitions.baseline_gaussian import BaselineGaussianTransition

log = logging.getLogger(__name__)


def _build_baseline(
    *,
    baseline_form: str,
    latent_dim: int,
    j: int,
    hidden_dim: int,
    n_layers: int,
) -> BaseBaseline:
    """Construct the baseline (μ_p) head for the requested form."""
    if baseline_form == "zero":
        return ZeroBaseline(
            latent_dim=latent_dim, j=j,
            hidden_dim=hidden_dim, n_layers=n_layers,
        )
    if baseline_form == "persistence":
        return PersistenceBaseline(
            latent_dim=latent_dim, j=j,
            hidden_dim=hidden_dim, n_layers=n_layers,
        )
    if baseline_form == "linear":
        return LinearBaseline(latent_dim=latent_dim, j=j)
    if baseline_form == "mlp":
        return MLPBaseline(
            latent_dim=latent_dim, j=j,
            hidden_dim=hidden_dim, n_layers=n_layers,
        )
    raise ValueError(
        f"baseline_form must be one of (zero, persistence, linear, mlp); "
        f"got {baseline_form!r}"
    )


def _build_init_centering_model(
    *,
    # --- Cell axes (the three grid dimensions) ---
    # baseline_form default is "persistence" rather than "mlp":
    # ``z_t ≈ z_{t-1}`` is the simplest dynamics-capable prior, has
    # no learnable mean params (so baseline_mode auto-clamps to
    # pinned), and is the right inductive bias for smooth latent
    # trajectories. ``mlp`` is still the canonical "high surface"
    # cell — the ablation grid passes baseline_form explicitly.
    baseline_form: Literal["zero", "persistence", "linear", "mlp"] = "persistence",
    baseline_mode: Literal["pinned", "learnable"] = "pinned",
    tracking_mode: Literal["fixed", "global_ema", "per_t"] = "per_t",
    # --- Shape ---
    j: int = 1,
    T_max: int = 32,
    data_dim: int = 1,
    latent_dim: int = 4,
    # Reserved for the future irregular-timestep / relative-time regime;
    # gated off by ``use_time_embedding=False`` for the current uniform-grid
    # regime where ``timepoints`` is just an within-window index and the
    # sinusoidal embedding carries no signal.
    use_time_embedding: bool = False,
    emb_time_dim: int = 16,
    # ``None`` ⇒ derive via the "channels = 16 × latent_dim" scaling rule
    # locked in by CONTEXT.md § "Size axis". Set explicitly to override.
    channels: int | None = None,
    baseline_hidden_dim: int | None = None,  # ``None`` ⇒ 16 × latent_dim
    encoder_hidden_dim: int | None = None,   # ``None`` ⇒ 16 × latent_dim
    decoder_hidden_dim: int | None = None,   # ``None`` ⇒ 16 × latent_dim
    baseline_n_layers: int = 2,
    # σ_data² tracking-EMA decay (used only by global_ema / per_t tracking
    # modes; ignored when tracking_mode="fixed"). Distinct from the
    # transition-weight EMA in ``hparams.ema_decay``.
    sigma_data_ema_decay: float = 0.997,
    # --- Stage-2 diffusion schedule ---
    diffusion_S_k: int = 1,
    diffusion_k_chunk: int = 1,
    diffusion_num_steps: int = 128,
    diffusion_layers: int = 2,
) -> DDSSM_base:
    """Construct an init-centering DDSSM model parametric over the ablation grid.

    The factory pattern is required so the *same* Python objects are
    passed to both transitions (so μ_p's parameters are shared and the
    handoff snapshot captures the right state).

    Capacity scaling: ``channels``, ``baseline_hidden_dim``,
    ``encoder_hidden_dim``, ``decoder_hidden_dim`` all default to
    ``16 × latent_dim`` (CONTEXT.md § Size axis). The score-net's
    feature mixer is convolutional (per ADR-0003); there is no
    ``nheads`` knob anymore since attention is not used at our latent
    dims. Pass explicit values to override the scaling rule for a
    single knob.

    Auto-degeneracy: if ``baseline_form`` is one of the parameter-free
    forms (``zero`` / ``persistence``) and ``baseline_mode`` is
    ``"learnable"``, the mode is clamped to ``"pinned"`` and a warning
    is emitted.  This avoids crashing under Optuna overrides that
    happen to sample the degenerate region.
    """
    if baseline_form in _PARAM_FREE_FORMS and baseline_mode == "learnable":
        log.warning(
            "baseline_form=%r has no learnable mu_p parameters; clamping "
            "baseline_mode from 'learnable' to 'pinned' (auto-degenerate per "
            "init-experiment.org § Composition with the ablation grid). Note: "
            "resolved_config.yaml still records the requested 'learnable'.",
            baseline_form,
        )
        baseline_mode = "pinned"

    # ``use_time_embedding`` is the authoritative on/off switch; ``emb_time_dim``
    # is only consulted when the path is enabled. Forcing 0 here makes every
    # submodule's ``self.emb_time_dim > 0`` guard collapse the time-conditioning
    # ops out of both eager and torch.compile graphs.
    if not use_time_embedding:
        emb_time_dim = 0

    # ---- size-matrix defaults (CONTEXT.md § "Size axis") ----
    if channels is None:
        channels = 16 * latent_dim
    if baseline_hidden_dim is None:
        baseline_hidden_dim = 16 * latent_dim
    if encoder_hidden_dim is None:
        encoder_hidden_dim = 16 * latent_dim
    if decoder_hidden_dim is None:
        decoder_hidden_dim = 16 * latent_dim
    # ---- shared ingredients ----
    baseline = _build_baseline(
        baseline_form=baseline_form,
        latent_dim=latent_dim, j=j,
        hidden_dim=baseline_hidden_dim, n_layers=baseline_n_layers,
    )
    aux_posterior = AuxPosterior(
        latent_dim=latent_dim, j=j,
        hidden_dim=baseline_hidden_dim, n_layers=baseline_n_layers,
    )
    sigma_data = SigmaDataBuffer(
        T_max=T_max, tracking_mode=tracking_mode, init_value=1.0,
        ema_decay=sigma_data_ema_decay,
    )

    # ---- stage-1 transition (Gaussian closed-form) ----
    stage1_transition = BaselineGaussianTransition(
        baseline=baseline,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
    )

    # ---- stage-2 transition (centered ESM/EDM) ----
    # Feature mixer is ``conv`` (not transformer); see
    # docs/adr/0003-score-net-feature-mixer-conv.md for the rationale.
    unet = partial(
        CSDIUnet,
        channels=channels,
        n_layers=diffusion_layers,
        embedding_dim=channels,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(type="conv", n_layers=1)
        ),
    )
    # ``k_sampling_mode`` defaults to ``"lsgm_is"`` on
    # ``DiffusionScheduleConfig`` — the LSGM importance-sampling
    # distribution from model-v2.org § Importance Sampling
    # (p_k ∝ (β / (1 − α²))^γ, with the unbiasing reweight in
    # ``_esm_chunk_loss``). No override needed here.
    schedule = DiffusionScheduleConfig(
        S_k=diffusion_S_k,
        k_chunk=diffusion_k_chunk,
        num_steps=diffusion_num_steps,
    )
    stage2_transition = DiffusionTransition(
        baseline=baseline,
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        T_max=T_max,
        unet=unet,
        schedule=schedule,
    )

    # ---- encoder / decoder built inline (no Small1D dependency) ----
    encoder = GaussianEncoder(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim, use_mask=False,
        hidden_dim=encoder_hidden_dim,
    )
    decoder = GaussianDecoder(
        data_dim=data_dim, latent_dim=latent_dim, j=j,
        emb_time_dim=emb_time_dim,
        hidden_dim=decoder_hidden_dim,
    )

    return DDSSM_base(
        encoder=encoder,
        decoder=decoder,
        transition=stage2_transition,
        j=j,
        data_dim=data_dim,
        latent_dim=latent_dim,
        emb_time_dim=emb_time_dim,
        use_observation_mask=False,
        # --- VHP-via-diffusion + baseline-centering ---
        aux_posterior=aux_posterior,
        baseline=baseline,
        baseline_anchor=None,        # populated by the handoff
        baseline_mode=baseline_mode,
        sigma_data=sigma_data,
        stage1_transition=stage1_transition,
    )


# hydra-zen wrapper so the preset can plug into experiment(model=...).
SmokeModel = builds(
    _build_init_centering_model,
    populate_full_signature=True,
)


model_store(SmokeModel, name="init_centering_smoke")

__all__ = ["SmokeModel", "_build_init_centering_model"]
