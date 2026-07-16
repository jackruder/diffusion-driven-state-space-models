"""``DDSSMModelConfig`` — the full build+train+eval description of a DDSSM.

Every model family carries a :class:`~ddssm.model.config.ModelConfig` subclass
that fully describes what's needed to build and train it. For the native DDSSM
family that means the composition tree (encoder / decoder / transition /
auxiliary posterior / baseline / sigma_data slots) plus the shape spine, the
scalar model-side knobs, and the trainer/optimiser hparams. This module hosts
the nested dataclasses; the adapter (`ddssm.adapters.ddssm.DDSSMAdapter`) uses
them to build the module lazily via :meth:`DDSSMModelConfig.build_module`.

Design:
- Slots for submodules (encoder / decoder / transition / aux_posterior /
  baseline / sigma_data) are typed ``Any`` and expected to be hydra-zen
  ``builds()`` instances — the ``_target_`` in each slot IS the discriminator
  between family variants (e.g. ``GaussianEncoder`` vs ``ARFlowEncoder``).
  ``build_module`` walks the tree via :func:`hydra_zen.instantiate`.
- Duplication resolved: ``S``, ``logvar_min``, ``logvar_max`` live on
  :class:`DDSSMModelKnobs` (model side); the trainer never reads them.
- ``T_max`` lives in :class:`DDSSMShape` — a single source of truth shared by
  the diffusion transition and the sigma_data buffer.
"""

from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

from ddssm.model.config import ModelConfig
from ddssm.training.stages import LambdaRampConf, LrScheduleGroupConf


@dataclass
class DDSSMShape:
    """Topology invariants of a DDSSM instance.

    These are the values that must agree across the submodule slots
    (encoder / decoder / transition / baseline / aux_posterior / sigma_data),
    so we lift them once here and fill any ``MISSING`` sentinels in the
    slot confs from this record at build time.
    """

    j: int = 1
    data_dim: int = 1
    latent_dim: int = 1
    emb_time_dim: int = 16
    covariate_dim: int = 0
    static_embed_dim: int = 0
    num_classes_per_static: list[int] | None = None
    use_observation_mask: bool = True
    # Max sequence length. Constrains the diffusion transition and the
    # sigma_data buffer — historically each carried its own copy, which
    # could silently drift. Owned here as the single source of truth.
    T_max: int = 32


@dataclass
class DDSSMModelKnobs:
    """Scalar knobs stored on the ``DDSSM_base`` instance itself.

    These are model-side (they parametrise the ELBO forward pass), not
    trainer-side. Keeping them here — and NOT on ``DDSSMTrainingHparams``
    — resolves today's duplicate storage between ``DDSSM_base.__init__``
    and ``DDSSMHyperParamsConf`` (``S``, ``logvar_min``, ``logvar_max``).
    """

    S: int = 1
    logvar_min: float = -13.0
    logvar_max: float = 13.0
    mask_emb_dim: int = 8
    recon_time_chunk: int | None = None
    recon_grad_checkpoint: bool = False


@dataclass
class DDSSMTrainingHparams(ModelConfig):
    """Trainer / optimiser hparams for DDSSM.

    Mirrors today's :class:`~ddssm.model.dssd.DDSSMHyperParamsConf` MINUS the
    duplicated model-side scalars (``S``, ``logvar_min``, ``logvar_max``) and
    the dead ``t_chunk`` field the trainer never reads. Subclasses
    :class:`ModelConfig` so it can also serve directly as
    :attr:`Experiment.hparams` when a caller only wants the training slice.
    """

    batch_size: int = 16
    ema_decay: float = 0.999
    # AdamW weight decay for the single optimizer / φθ side. ``weight_decay_psi``
    # overrides ψ-side independently; ``None`` falls back to ``weight_decay``.
    weight_decay: float = 1e-4
    weight_decay_psi: float | None = None
    grad_accum_steps: int = 4
    # Global grad-norm clip, applied after the non-finite-grad skip check
    # in ``DDSSMTrainer._optimizer_step``. ``None`` disables clipping.
    clip_grad_norm: float | None = 1.0
    # Optional Adam betas for the score-net (ψ) param groups in single-loss
    # mode (list, not tuple, for OmegaConf). ``None`` keeps today's topology.
    psi_betas: list[float] | None = None

    enc_lr: float = 5e-4
    dec_lr: float = 5e-4
    trans_lr: float = 5e-4

    # Optional λ-ramp for the FullELBO rate weight on the φθ-side KL terms.
    lambda_ramp: LambdaRampConf | None = None
    # Optional per-role LR schedule (φθ / ψ). Enabling requires ``lambda_ramp``.
    lr_schedule: LrScheduleGroupConf | None = None
    # Enable split-loss training: separate optimisers + objectives for ψ vs φθ.
    use_split_loss: bool = False


@dataclass
class DDSSMModelConfig(ModelConfig):
    """The full DDSSM family config: build + train + eval in one nested tree.

    Slot dataclass. Submodule slots are typed ``Any`` because they carry
    hydra-zen ``builds()`` confs whose ``_target_`` discriminates between
    family variants (Gaussian vs ARFlow vs Identity encoder, Diffusion vs
    Gaussian transition, …). ``build_module`` walks the tree.
    """

    shape: DDSSMShape = field(default_factory=DDSSMShape)
    encoder: Any = None
    decoder: Any = None
    transition: Any = None
    aux_posterior: Any = None
    baseline: Any = None
    sigma_data: Any = None
    model_knobs: DDSSMModelKnobs = field(default_factory=DDSSMModelKnobs)
    training: DDSSMTrainingHparams = field(default_factory=DDSSMTrainingHparams)

    def __post_init__(self) -> None:
        if self.aux_posterior is None:
            raise ValueError(
                "DDSSMModelConfig.aux_posterior is required: DDSSM_base "
                "computes the initial-state term via the transition's "
                "hierarchical VHP walk, which needs q_Φ(z_aux | z_{1:j}). "
                "Provide an AuxPosterior builds() in the aux_posterior slot."
            )

    @property
    def batch_size(self) -> int:
        """DataLoader batch size — delegates to the training slice.

        ``Experiment.train`` reads ``hparams.batch_size`` via ``getattr`` to
        sync the loader; exposing it at the top level keeps that path
        working when :attr:`Experiment.hparams` holds the whole
        :class:`DDSSMModelConfig`.
        """
        return self.training.batch_size

    def build_module(self):
        """Instantiate a ``DDSSM_base`` from this config.

        Uses :func:`hydra_zen.instantiate` to walk the submodule slots.
        Baseline is built FIRST so its instance can be threaded into the
        transition (which takes ``baseline`` as a pre-instantiated arg —
        see ``DiffusionTransition.__init__``; the DDSSM_base and transition
        must share the same baseline instance by reference).
        """
        # Local imports so this leaf module doesn't force torch import on
        # anyone touching the config schema.
        from inspect import signature

        from hydra_zen import instantiate, get_target

        from ddssm.model.dssd import DDSSM_base

        baseline = instantiate(self.baseline) if self.baseline is not None else None
        aux_posterior = instantiate(self.aux_posterior)
        sigma_data = (
            instantiate(self.sigma_data) if self.sigma_data is not None else None
        )
        encoder = instantiate(self.encoder)
        decoder = instantiate(self.decoder)
        # Thread the shared baseline instance into the transition IFF the
        # target's constructor takes ``baseline`` (DiffusionTransition does;
        # GaussianTransition doesn't). Introspect the target — catching a
        # TypeError from instantiate doesn't work because hydra-zen wraps it
        # in an InstantiationException.
        trans_target = get_target(self.transition)
        if "baseline" in signature(trans_target).parameters:
            transition = instantiate(self.transition, baseline=baseline)
        else:
            transition = instantiate(self.transition)

        shape = self.shape
        knobs = self.model_knobs
        return DDSSM_base(
            encoder=encoder,
            decoder=decoder,
            transition=transition,
            j=shape.j,
            data_dim=shape.data_dim,
            latent_dim=shape.latent_dim,
            emb_time_dim=shape.emb_time_dim,
            covariate_dim=shape.covariate_dim,
            static_embed_dim=shape.static_embed_dim,
            num_classes_per_static=shape.num_classes_per_static,
            use_observation_mask=shape.use_observation_mask,
            mask_emb_dim=knobs.mask_emb_dim,
            logvar_min=knobs.logvar_min,
            logvar_max=knobs.logvar_max,
            S=knobs.S,
            aux_posterior=aux_posterior,
            baseline=baseline,
            sigma_data=sigma_data,
            recon_time_chunk=knobs.recon_time_chunk,
            recon_grad_checkpoint=knobs.recon_grad_checkpoint,
        )


__all__ = [
    "DDSSMModelConfig",
    "DDSSMModelKnobs",
    "DDSSMShape",
    "DDSSMTrainingHparams",
]
