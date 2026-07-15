"""``DDSSMAdapter`` — the native DDSSM family behind the ``ModelAdapter`` seam.

Wraps a *pre-composed* :class:`~ddssm.model.dssd.DDSSM_base` (built externally
by the ``SmokeModel`` factory & co — the module is never rebuilt here) and
exposes it through the uniform fit / forecast / checkpoint surface. The ``fit``
body is the trainer-construction + fit-call block lifted verbatim from
:meth:`ddssm.experiment.experiment.Experiment.train` (post module-3, with the
trainable-mask handling already removed) so the native path keeps bit-for-bit
parity with today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

import torch

from ddssm.adapters.base import ModelAdapter
from ddssm.training.train import DDSSMTrainer

if TYPE_CHECKING:  # annotation-only — keep this module cycle-safe / import-light
    from ddssm.model.dssd import DDSSM_base
    from ddssm.model.config import ModelConfig
    from ddssm.data.datamodule import TimeSeriesDataModule
    from ddssm.experiment.experiment import TrainingScalars


class DDSSMAdapter(ModelAdapter):
    """Integrate the native DDSSM family with the ``Experiment`` workflow.

    The owned module is a pre-composed :class:`DDSSM_base`; ``hparams`` (when
    supplied to :meth:`fit`) governs trainer / optimizer knobs only and never
    rebuilds topology, so :meth:`load_checkpoint`'s ``hparams`` is unused here
    (the module is already built). ``build_trainer`` is the ``TrainerPartial``
    slot: a curried :class:`DDSSMTrainer` factory whose curried ``hparams`` is
    superseded by the ``hparams=`` keyword :meth:`fit` passes through.
    """

    def __init__(
        self,
        config: ModelConfig,
        module: DDSSM_base,
        build_trainer: Callable[..., DDSSMTrainer] | None = None,
    ) -> None:
        """Store the pre-composed module + (optional) curried trainer factory."""
        super().__init__(config)
        self._module = module  # composed externally — DDSSM_base untouched
        self._build_trainer = build_trainer or DDSSMTrainer
        self.trainer: DDSSMTrainer | None = None

    @property
    def module(self) -> DDSSM_base:
        """The raw, checkpointable ``DDSSM_base`` this adapter owns."""
        return self._module

    def fit(
        self,
        *,
        data: TimeSeriesDataModule,
        training: TrainingScalars,
        device: torch.device,
        csv_log_path: str,
        tensorboard_dir: str,
        checkpoint_dir: str,
        hparams: ModelConfig | None = None,  # Experiment.hparams; wins over self.config
        wandb_config: dict | None = None,
        model_config_yaml: str | None = None,
    ) -> None:
        """Train the owned module (trainer-construction + fit-call from Experiment).

        Mirrors :meth:`Experiment.train`'s post-path-setup block: builds the
        trainer with the exact same kwargs (``hparams`` passed as a keyword so
        it supersedes ``TrainerPartial``'s curried value), stores it on
        ``self.trainer``, drives ``trainer.fit`` with a val loader iff
        ``validate_every > 0``, and closes the metric sinks in ``finally``.
        ``resume_from`` needs no code here — it rides ``fit_kwargs()`` into
        ``trainer.fit``. No trainable-mask handling. When
        ``data.train_loader()`` is ``None`` (``NullDataModule``) this no-ops
        WITHOUT building a trainer or writing a CSV.
        """
        train_loader = data.train_loader()
        if train_loader is None:
            # NullDataModule: no data attached — no-op (no trainer, no CSV).
            return

        # ``hparams=hparams or self.config`` as a keyword so it supersedes the
        # TrainerPartial's curried hparams (ADR-0004). Surface parity with
        # Experiment.train: do NOT expose the trainer's optimizer/quiet params.
        trainer_kwargs: dict = dict(
            model=self._module,
            device=device,
            csv_log_path=csv_log_path,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            wandb_config=wandb_config,
            hparams=hparams or self.config,
        )
        if model_config_yaml is not None:
            trainer_kwargs["model_config_yaml"] = model_config_yaml
        trainer = self._build_trainer(**trainer_kwargs)
        self.trainer = trainer

        val_loader = data.val_loader() if training.validate_every > 0 else None

        try:
            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                batch_transform=data.batch_transform,
                **training.fit_kwargs(),
            )
        finally:
            # Logger lifecycle is owned by the run, not by fit(). Close the
            # CSV/TB/W&B sinks exactly once, after fit (or on the exception
            # path).
            trainer.metrics.close()

    def forecast(
        self,
        x_hist: torch.Tensor,
        x_mask: torch.Tensor,
        past_time: torch.Tensor,
        future_time: torch.Tensor,
        past_covariates: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
        static_covariates: torch.Tensor | None,
        *,
        num_samples: int,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Delegate to ``DDSSM_base.forecast``, forwarding extra sampling knobs.

        ``**kwargs`` preserves ``DDSSM_base.forecast``'s extra kwonly sampling
        controls (``use_vp_init`` / ``s_churn`` / ``s_noise`` / ``s_tmin`` /
        ``s_tmax``); eval passes only ``num_samples`` but the surface stays open.
        """
        return self._module.forecast(
            x_hist=x_hist,
            x_mask=x_mask,
            past_time=past_time,
            future_time=future_time,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            static_covariates=static_covariates,
            num_samples=num_samples,
            **kwargs,
        )

    def log_prob(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Delegate to ``DDSSM_base.log_prob`` (overrides the base ABC raise)."""
        return self._module.log_prob(*args, **kwargs)

    def save_checkpoint(self, path: str) -> None:
        """Serialize via the trainer's public ``save_checkpoint`` (v3 schema).

        Raises ``RuntimeError`` when :meth:`fit` has not run (no trainer to
        snapshot optimizer / EMA / step state from) — same wording shape as
        ``Experiment.objective_value``'s "``.train()`` ran first" guard.
        """
        if self.trainer is None:
            raise RuntimeError(
                "DDSSMAdapter.save_checkpoint requires that .fit() ran first "
                "to populate self.trainer."
            )
        self.trainer.save_checkpoint(path)

    def load_checkpoint(
        self,
        path: str,
        *,
        device: torch.device,
        hparams: ModelConfig | None = None,  # unused: DDSSM module is pre-composed
        load_ema: bool = True,
        expected_model_config_yaml: str | None = None,
        strict: bool = False,
    ) -> None:
        """Restore state into the pre-composed module (trainer-free).

        ``hparams`` is unused — the DDSSM module is already built, so there is
        no topology to (re)construct on load; old checkpoints load
        bit-identically. A cross-format payload raises ``ValueError``:
        :func:`ddssm.training.checkpoint.load_into_model` only *warns* on an
        unknown ``_format`` (and would then silently partial-load into the
        module), so we pre-read the payload's ``_format`` tag and reject any
        non-DDSSM format explicitly before delegating.
        """
        self._reject_foreign_format(path, device=device)
        from ddssm.training.checkpoint import load_into_model

        load_into_model(
            self._module,
            path,
            device=device,
            expected_model_config_yaml=expected_model_config_yaml,
            load_ema=load_ema,
            strict=strict,
        )

    @staticmethod
    def _reject_foreign_format(path: str, *, device: torch.device) -> None:
        """Raise ``ValueError`` when ``path`` is not a DDSSM checkpoint payload.

        The DDSSM checkpoint loader tolerates an unknown ``_format`` (warn +
        best-effort load) and treats any dict lacking ``model_state`` as a
        legacy raw ``state_dict``; the adapter contract instead requires a hard
        ``ValueError`` so a foreign payload never silently partial-loads. Two
        cases are rejected:

        * a payload dict carrying a ``_format`` tag not in
          :data:`_SUPPORTED_FORMATS` (a versioned foreign format), and
        * a payload dict with neither a ``_format`` tag nor a ``model_state``
          key (an unrecognized mapping — e.g. some other framework's dump).

        A legacy pre-payload DDSSM checkpoint is a bare ``state_dict`` whose
        values are all tensors and whose keys are module parameter names — it
        carries neither ``_format`` nor ``model_state`` but IS loadable, so we
        only reject the no-tag/no-model_state case when the mapping doesn't look
        like a tensor state_dict (i.e. it has at least one non-tensor value).
        """
        from ddssm.training.checkpoint import _SUPPORTED_FORMATS

        payload = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(payload, dict):
            return
        fmt = payload.get("_format")
        if fmt is not None:
            if fmt not in _SUPPORTED_FORMATS:
                raise ValueError(
                    f"Cannot load checkpoint {path!r}: foreign _format={fmt!r} "
                    f"(DDSSMAdapter accepts {sorted(_SUPPORTED_FORMATS)})."
                )
            return
        if "model_state" in payload:
            return
        # No _format, no model_state: only a bare tensor state_dict is a valid
        # legacy DDSSM payload. Any non-tensor value marks it as foreign.
        if any(not isinstance(v, torch.Tensor) for v in payload.values()):
            raise ValueError(
                f"Cannot load checkpoint {path!r}: payload is neither a DDSSM "
                f"checkpoint (no '_format'/'model_state') nor a bare tensor "
                f"state_dict."
            )
