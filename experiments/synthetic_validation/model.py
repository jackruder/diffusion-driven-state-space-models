"""A small, hand-written DDSSM model factory for the authoring guide.

This is the worked example for ``docs/authoring/model.md``: it builds a
:class:`~ddssm.model.dssd.DDSSM_base` from the underlying runtime classes
**without** going through ``experiments.init_centering.model.SmokeModel``.

Why a factory *function* (and not nested ``builds(...)`` configs)? The baseline
(μ_p / σ_p head) must be the **same Python object** in the stage-1
:class:`~ddssm.model.transitions.baseline_gaussian.BaselineGaussianTransition`,
the stage-2 :class:`~ddssm.model.transitions.diffusion.DiffusionTransition`, and
the model's own ``baseline`` slot — so its parameters are shared across the
stage-1 → stage-2 handoff. Nested configs would each instantiate a *separate*
baseline, breaking that sharing. A factory instantiates it once and passes it by
reference. (``ddssm.experiment.builders`` exposes per-slot ``builds`` configs;
``experiments/init_centering/model.py`` is the fuller reference factory.)

The architecture here is deliberately minimal and fixed:

* baseline: ``PersistenceBaseline`` by default (μ_p = z_{t-1}; σ_p
  from a small MLP), frozen. Override with ``baseline_form="zero"``
  for the legacy ZeroBaseline behaviour.
* σ_data²: :class:`SigmaDataBuffer` using the library default
  (``"per_t"``) — each timestep tracks its own running EMA so the
  centered-ESM target preconditioning stays calibrated as the
  encoder's residual distribution evolves through stage 2.
* stage-1 transition: closed-form Gaussian centered on the baseline.
* stage-2 transition: centered diffusion with a small conv-mixer ``CSDIUnet``.
* encoder/decoder: the default :class:`GaussianEncoder` / :class:`GaussianDecoder`.

See ``docs/authoring/model.md`` for the menu of alternatives (other baselines,
mixers, aggregators, fut-summaries) you can swap in.
"""

from __future__ import annotations

from typing import Literal
from functools import partial

from hydra_zen import builds

from ddssm.model.dssd import DDSSM_base
from ddssm.nn.diffnets import CSDIUnet, FeatureMixerConfig, DiffResidualBlockConfig
from ddssm.model.decoder import GaussianDecoder
from ddssm.model.encoder import GaussianEncoder
from ddssm.nn.aux_posterior import AuxPosterior
from ddssm.model.centering.baselines import (
    MLPBaseline,
    BaseBaseline,
    ZeroBaseline,
    LinearBaseline,
    PersistenceBaseline,
)


def _make_baseline(
    form: str, *, latent_dim: int, j: int, hidden_dim: int
) -> BaseBaseline:
    if form == "zero":
        return ZeroBaseline(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=hidden_dim,
            n_layers=2,
        )
    if form == "persistence":
        return PersistenceBaseline(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=hidden_dim,
            n_layers=2,
        )
    if form == "linear":
        return LinearBaseline(latent_dim=latent_dim, j=j)
    if form == "mlp":
        return MLPBaseline(
            latent_dim=latent_dim,
            j=j,
            hidden_dim=hidden_dim,
            n_layers=2,
        )
    raise ValueError(
        f"baseline_form must be one of zero/persistence/linear/mlp; got {form!r}"
    )


from ddssm.model.centering.sigma_data import SigmaDataBuffer
from ddssm.model.transitions.diffusion import (
    DiffusionTransition,
    DiffusionScheduleConfig,
)
from ddssm.model.transitions.baseline_gaussian import BaselineGaussianTransition


def build_synthval_model(
    *,
    data_dim: int = 1,
    latent_dim: int = 1,
    j: int = 1,
    # Reserved for the future irregular-timestep / relative-time regime;
    # gated off by ``use_time_embedding=False`` for the current uniform-grid
    # regime (matches the init-centering factory default).
    use_time_embedding: bool = False,
    emb_time_dim: int = 16,
    T_max: int = 32,
    hidden_dim: int = 32,
    channels: int = 32,
    diffusion_layers: int = 2,
    diffusion_num_steps: int = 64,
    # Stage-2 ESM/EDM schedule knobs. ``k_sampling_mode`` defaults
    # to ``"adaptive_is"`` on ``DiffusionScheduleConfig`` — the
    # loss-aware optimal IS density per-t built from the live
    # ``σ_d²`` running estimate (see ``importance-sampling.org``
    # § Mean-dominated regime). The parameter is plumbed through so
    # ablations can flip to ``"uniform"``, ``"lsgm_is"``, or
    # ``"adaptive_is_full"`` if needed. ``S_k=1`` matches the
    # dataclass default — bump it up for tighter per-step gradient
    # estimates on small-data overfits.
    diffusion_k_sampling_mode: str = "adaptive_is",
    diffusion_S_k: int = 1,
    # Baseline form for the shared centering head. Persistence
    # (``μ_p = z_{t-1}``) is the library default since it's the
    # simplest dynamics-capable prior. Use ``"zero"`` for the legacy
    # stationary-prior behaviour, ``"mlp"`` / ``"linear"`` for richer
    # parametric baselines.
    baseline_form: Literal["zero", "persistence", "linear", "mlp"] = "persistence",
) -> DDSSM_base:
    """Compose a minimal DDSSM model from runtime parts.

    Args:
        data_dim: Observed channel count ``D``.
        latent_dim: Latent dimension ``d``.
        j: Latent history window.
        use_time_embedding: When ``False`` (default), force ``emb_time_dim=0``
            and disable the absolute-time conditioning path.
        emb_time_dim: Time-embedding width (consulted only when
            ``use_time_embedding=True``).
        T_max: Max sequence length (must cover the data's ``T``); sizes the
            σ_data² buffer.
        hidden_dim: Width for the baseline / aux-posterior / encoder / decoder.
        channels: Score-network width.
        diffusion_layers: Score-network depth.
        diffusion_num_steps: Denoising steps for the stage-2 diffusion transition.

    Returns:
        An assembled :class:`~ddssm.model.dssd.DDSSM_base`.
    """
    if not use_time_embedding:
        emb_time_dim = 0
    # --- shared ingredients: built once, passed by reference ---
    baseline = _make_baseline(
        baseline_form,
        latent_dim=latent_dim,
        j=j,
        hidden_dim=hidden_dim,
    )
    aux_posterior = AuxPosterior(
        latent_dim=latent_dim, j=j, hidden_dim=hidden_dim, n_layers=2
    )
    # tracking_mode defaults to "per_t" on SigmaDataBuffer — keeps the
    # per-timestep variance tracking through stage 2 so the centered-ESM
    # target stays calibrated as the encoder evolves.
    sigma_data = SigmaDataBuffer(T_max=T_max, init_value=1.0)

    # --- stage-1 transition: closed-form Gaussian centered on the baseline ---
    stage1_transition = BaselineGaussianTransition(
        baseline=baseline, latent_dim=latent_dim, j=j, emb_time_dim=emb_time_dim
    )

    # --- stage-2 transition: centered diffusion (small conv-mixer score net) ---
    unet = partial(
        CSDIUnet,
        channels=channels,
        n_layers=diffusion_layers,
        embedding_dim=channels,
        residual_block=DiffResidualBlockConfig(
            feature=FeatureMixerConfig(type="conv", n_layers=1)
        ),
    )
    stage2_transition = DiffusionTransition(
        baseline=baseline,  # SAME instance as stage 1
        latent_dim=latent_dim,
        j=j,
        emb_time_dim=emb_time_dim,
        T_max=T_max,
        unet=unet,
        schedule=DiffusionScheduleConfig(
            num_steps=diffusion_num_steps,
            k_sampling_mode=diffusion_k_sampling_mode,
            S_k=diffusion_S_k,
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

    # Pre-allocate ``baseline_anchor`` so the state_dict has the same
    # keys before and after the centering handoff. Under
    # ``baseline_mode="pinned"`` the anchor is never read by the loss
    # (see ``DDSSM_base.forward``: the r_mu_p term only fires when
    # ``baseline_mode == "learnable"``), so this is a state-dict-shape
    # alignment knob, nothing more. Without it, ``ddssm.visualize`` /
    # ``ddssm.evaluate`` fail to load a post-handoff checkpoint with
    # "Unexpected key(s) in state_dict: baseline_anchor.*".
    baseline_anchor = baseline.snapshot()

    return DDSSM_base(
        encoder=encoder,
        decoder=decoder,
        transition=stage2_transition,
        j=j,
        data_dim=data_dim,
        latent_dim=latent_dim,
        emb_time_dim=emb_time_dim,
        use_observation_mask=False,
        aux_posterior=aux_posterior,  # mandatory: owns the init-state term
        baseline=baseline,
        baseline_anchor=baseline_anchor,
        baseline_mode="pinned",  # μ_p frozen
        sigma_data=sigma_data,
        stage1_transition=stage1_transition,
    )


# hydra-zen wrapper so the preset can plug into ``experiment(model=...)``.
SynthValModel = builds(build_synthval_model, populate_full_signature=True)

__all__ = ["SynthValModel", "build_synthval_model"]
