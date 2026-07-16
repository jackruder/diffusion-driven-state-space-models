"""``ModelAdapter`` ABC + ``MetricNotSupported`` — the model-family seam.

An adapter integrates one model family (DDSSM, baseline forecasters, …) with the
single :class:`~ddssm.experiment.experiment.Experiment` orchestrator. The
adapter OWNS a plain :class:`torch.nn.Module`; the ``Experiment`` never touches
that module directly, only the adapter's fit / forecast / checkpoint surface.

Import-cycle note
-----------------
This module must NOT acquire a *runtime* import of ``ddssm.experiment`` — a
later refactor makes ``experiment.py`` import ``ddssm.adapters``, and a runtime
edge back the other way would cycle. So ``TrainingScalars`` and
``TimeSeriesDataModule`` are pulled in only under :data:`typing.TYPE_CHECKING`
(they are annotation-only here). ``ModelConfig`` (a stdlib-only leaf) and
``torch`` are safe to import at runtime.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

from ddssm.model.config import ModelConfig

if TYPE_CHECKING:  # annotation-only — never imported at runtime (cycle guard)
    from ddssm.data.datamodule import TimeSeriesDataModule
    from ddssm.experiment.experiment import TrainingScalars


class MetricNotSupported(NotImplementedError):  # noqa: N818 (name fixed by design §2)
    """Raised ONLY at adapter / ``require_module`` gating boundaries.

    Subclasses :class:`NotImplementedError` so callers that only care about
    "this metric isn't available for this family" can catch the narrow type —
    but the metric runner catches ONLY ``MetricNotSupported``, never the broad
    base. The codebase raises bare ``NotImplementedError`` as a *load-bearing*
    signal deep inside DDSSM internals (``encoder.py``, ``transitions.py``,
    ``dist_heads.py``, ``diffnets.py``); catching the base class there would
    silently skip metrics on real bugs. This dedicated subtype keeps the
    "unsupported metric" path distinguishable from those genuine failures.
    """


class ModelAdapter(abc.ABC):
    """Integrate one model family (fit/forecast/checkpoint) with the workflow.

    The underlying model is a plain ``nn.Module`` OWNED by the adapter; the
    single :class:`~ddssm.experiment.experiment.Experiment` orchestrator drives
    families only through this surface.

    Method-lifting rule
    -------------------
    Lift a method onto this ABC only when **≥2 families** can meaningfully
    implement it (today: :meth:`forecast`, :meth:`log_prob`). Family-specific
    capabilities stay on the concrete adapter; shared-but-optional methods live
    here with a base that raises :class:`MetricNotSupported`, and each family
    overrides when applicable.

    hparams vs. self.config
    -----------------------
    ``self.config`` is the *pre-fit default only*. Once :meth:`fit` /
    :meth:`load_checkpoint` receive ``hparams``, that wins (ADR-0004);
    ``self.config`` must never be read mid-fit. After a hydra ``instantiate``,
    ``config`` / ``Experiment.hparams`` / ``TrainerPartial``'s curried hparams
    are THREE distinct instances — ``Experiment.train`` forwarding
    ``self.hparams`` into :meth:`fit` is the single reconciliation point.

    For DDSSM specifically the module is *pre-composed*, so ``hparams`` governs
    trainer / optimizer knobs only and can never rebuild topology;
    :meth:`load_checkpoint`'s ``hparams`` is consequently unused by
    ``DDSSMAdapter``. Re-vendored families that (re)construct their module from
    config do read ``hparams`` in :meth:`load_checkpoint`.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Store ``config`` as the pre-fit default (see class docstring).

        Pre-fit default ONLY. Once fit/load_checkpoint receive ``hparams``,
        that wins (ADR-0004); ``self.config`` must never be read mid-fit. After
        a hydra instantiate, ``config`` / ``Experiment.hparams`` /
        ``TrainerPartial``'s curried hparams are THREE distinct instances —
        ``Experiment.train`` forwarding ``self.hparams`` into :meth:`fit` is the
        single reconciliation point.
        """
        self.config = config

    @property
    @abc.abstractmethod
    def module(self) -> torch.nn.Module:
        """The raw, checkpointable ``nn.Module`` this adapter owns."""
        ...

    @abc.abstractmethod
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
        """Train the owned module, logging ``loss/total`` for train & val splits.

        ``hparams``, when supplied, overrides ``self.config`` for every knob;
        ``self.config`` is not read here. When ``data.train_loader()`` is
        ``None`` (``NullDataModule``) fit must no-op.
        """
        ...

    @abc.abstractmethod
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
    ) -> dict[str, torch.Tensor]:
        """Roll out a probabilistic forecast in NORMALIZED space.

        Returns the existing eval signature verbatim: ``pred_mean`` of shape
        ``(B, D, L2)`` and ``pred_samples`` of shape ``(B, S, D, L2)``.
        """
        ...

    @abc.abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Serialize the owned module's state to ``path``."""
        ...

    @abc.abstractmethod
    def load_checkpoint(
        self,
        path: str,
        *,
        device: torch.device,
        # ``hparams`` wins over self.config for module (re)construction:
        hparams: ModelConfig | None = None,
        load_ema: bool = True,
        expected_model_config_yaml: str | None = None,
        strict: bool = False,
    ) -> None:
        """Restore state from ``path``.

        ``hparams`` wins over ``self.config`` for any module (re)construction
        the family performs on load (unused by ``DDSSMAdapter``, whose module is
        pre-composed). A cross-format checkpoint must raise ``ValueError``.
        """
        ...

    # ---- Shared-but-optional methods: base raises, families override. --------

    def log_prob(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Exact log-density of the family, when defined.

        Base implementation raises :class:`MetricNotSupported` naming the
        concrete adapter class; families that can compute a log-density
        override this.
        """
        raise MetricNotSupported(f"{type(self).__name__} does not implement log_prob")
