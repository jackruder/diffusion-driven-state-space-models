"""``CSDIAdapter`` — the re-vendored CSDI forecaster behind the adapter seam.

Wraps :class:`ddssm.adapters._csdi_vendor.main_model.CSDI_Forecasting` (module 6's
byte-identical copy) so a CSDI baseline trains / forecasts / checkpoints through
the same :class:`~ddssm.adapters.base.ModelAdapter` surface the native DDSSM model
uses. The vendored forward / sampler / loss are called untouched (fidelity to the
standalone Solar-forecasting baseline); the adapter only owns the plumbing:

* lazy module construction from the *winning* config (``hparams`` beats
  ``self.config``, ADR-0004) with the ``_ensure_device`` repair the vendored code
  needs (its ``.device`` string and ``alpha_torch`` tensor are plain attrs that
  ``nn.Module.to()`` never moves);
* the batch mapping onto CSDI's ``(B, L, K)`` convention, deriving the forecast
  ``gt_mask`` from the data module's ``forecast_split`` (a WINDOWED module is
  required — Synthetic / Null leave it ``None``);
* a minimal fit loop (Adam + ``MultiStepLR``, batch-size-weighted val mean,
  periodic + latest checkpoints, resume / interrupt / preempt mirroring
  ``train.py``) that logs ``loss/total`` through :class:`MetricStore`;
* a self-contained ``csdi_ckpt_v1`` checkpoint schema (via the shared
  ``atomic_save`` / ``check_model_config_drift`` helpers).

Covariates are ignored (documented limitation); ``log_prob`` is unsupported and
inherits the base :class:`~ddssm.adapters.base.MetricNotSupported` default.
"""

from __future__ import annotations

import os
import math
import random
import signal
from typing import TYPE_CHECKING
import logging
import datetime
import contextlib
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import OmegaConf

from ddssm.model.config import ModelConfig
from ddssm.adapters.base import ModelAdapter
from ddssm.training.loggers import (
    CSVLogger,
    MetricSpec,
    MetricStore,
    WandbLogger,
    TensorBoardLogger,
)
from ddssm.training.checkpoint import (
    NoUsableCheckpointError,
    atomic_save,
    check_model_config_drift,
)
from ddssm.adapters._csdi_vendor.main_model import CSDI_Forecasting

if TYPE_CHECKING:
    from ddssm.data.datamodule import TimeSeriesDataModule
    from ddssm.experiment.experiment import TrainingScalars

log = logging.getLogger(__name__)

_CSDI_FORMAT = "csdi_ckpt_v1"


class _PreemptError(RuntimeError):
    """Mirror of :class:`ddssm.training.train.PreemptError` for the CSDI fit loop.

    Carries the path to the checkpoint written immediately before the raise so a
    preempt-aware launcher can resume mid-fit. A local subtype avoids importing
    ``train.py`` (whose module-level cost + optimizer-compile machinery the CSDI
    adapter has no need for).
    """

    def __init__(self, resume_from: str) -> None:
        super().__init__(f"training preempted; resume_from={resume_from}")
        self.resume_from = resume_from


@dataclass
class CSDIConfig(ModelConfig):
    """Config for the re-vendored CSDI forecaster (``CSDI_Forecasting``).

    Architecture fields mirror the config dict ``csdi_transition.py`` assembles;
    optimizer / runtime fields carry the upstream paper defaults. ``batch_size``
    is inherited from :class:`ModelConfig`.
    """

    # architecture (vendored CSDI_Forecasting construction, cf. csdi_transition.py)
    target_dim: int = 1
    layers: int = 4
    channels: int = 64
    nheads: int = 8
    diffusion_embedding_dim: int = 128
    num_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.5
    schedule: str = "quad"
    timeemb: int = 128
    featureemb: int = 16
    # resolved None -> target_dim in _build (vendored code crashes on None:
    # main_model.py compares int > None)
    num_sample_features: int | None = None
    is_unconditional: bool = False
    # REQUIRED: CSDI_base.__init__ reads it; "test" = fixed forecast-pattern
    # training (upstream Solar behaviour)
    target_strategy: str = "test"
    # REQUIRED by diff_CSDI (csdi_transition.py sets it)
    is_linear: bool = False
    # optimizer / runtime (paper defaults)
    lr: float = 1e-3
    weight_decay: float = 1e-6
    lr_decay_milestones: tuple[float, ...] = (0.75, 0.9)  # fractions of total steps
    lr_decay_gamma: float = 0.1
    clip_grad_norm: float | None = None


class CSDIAdapter(ModelAdapter):
    """Integrate the re-vendored CSDI forecaster with the ``Experiment`` workflow."""

    def __init__(self, config: CSDIConfig) -> None:
        """Store ``config`` as the pre-fit default; the module builds lazily."""
        super().__init__(config)
        # Built lazily from the WINNING config (hparams or self.config) at fit /
        # load time — the arch depends on knobs that only arrive then.
        self.csdi: CSDI_Forecasting | None = None
        self._device: torch.device = torch.device("cpu")
        # Set on build so save/resume can persist the exact winning config.
        self._built_config: CSDIConfig | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._global_step: int = 0
        self._model_config_yaml: str | None = None
        self._preempt_pending: bool = False

    # ---- ModelAdapter surface ------------------------------------------------
    @property
    def module(self) -> torch.nn.Module:
        """The raw, checkpointable ``nn.Module`` (the vendored CSDI forecaster)."""
        if self.csdi is None:
            raise RuntimeError(
                "CSDIAdapter module accessed before build; call fit / "
                "load_checkpoint first."
            )
        return self.csdi

    # ---- lazy module construction -------------------------------------------
    def _config_dict(self, cfg: CSDIConfig) -> dict:
        """Assemble the vendored config dict the way ``csdi_transition.py`` does.

        Mirrors that mapping (including ``target_strategy`` / ``is_linear`` and the
        ``num_sample_features=None -> target_dim`` resolution) but targets the new
        adapters vendor copy.
        """
        num_sample_features = (
            int(cfg.num_sample_features)
            if cfg.num_sample_features is not None
            else int(cfg.target_dim)
        )
        return {
            "diffusion": {
                "layers": int(cfg.layers),
                "channels": int(cfg.channels),
                "nheads": int(cfg.nheads),
                "diffusion_embedding_dim": int(cfg.diffusion_embedding_dim),
                "beta_start": float(cfg.beta_start),
                "beta_end": float(cfg.beta_end),
                "num_steps": int(cfg.num_steps),
                "schedule": str(cfg.schedule),
                "is_linear": bool(cfg.is_linear),
            },
            "model": {
                "is_unconditional": bool(cfg.is_unconditional),
                "timeemb": int(cfg.timeemb),
                "featureemb": int(cfg.featureemb),
                "target_strategy": str(cfg.target_strategy),
                "num_sample_features": num_sample_features,
            },
        }

    def _build(self, cfg: CSDIConfig, device: torch.device) -> None:
        """Construct ``CSDI_Forecasting`` from ``cfg`` and move it to ``device``.

        Called at fit start and inside ``load_checkpoint`` (both use ``hparams or
        self.config``). Applies the ``_ensure_device`` repair the vendored code
        needs (``.device`` / ``alpha_torch`` are plain attrs ``.to()`` misses).
        """
        config_dict = self._config_dict(cfg)
        self.csdi = CSDI_Forecasting(
            config_dict, device, target_dim=int(cfg.target_dim)
        )
        self.csdi.to(device)
        self._device = device
        self._built_config = cfg
        self._ensure_device(device)

    def _ensure_device(self, device: torch.device) -> None:
        """Point the vendored CSDI at ``device`` (``alpha_torch`` isn't a buffer).

        ``time_embedding`` / ``get_side_info`` allocate via the ``.device`` attr,
        and ``alpha_torch`` is a plain tensor attr, so ``nn.Module.to()`` never
        moves them — repair both here (mirrors ``csdi_transition._ensure_device``).
        """
        if self.csdi is None:
            return
        self.csdi.device = device
        if self.csdi.alpha_torch.device != device:
            self.csdi.alpha_torch = self.csdi.alpha_torch.to(device)

    # ---- batch mapping -------------------------------------------------------
    def _forecast_split(self, data: TimeSeriesDataModule) -> int:
        """The past/future boundary; raise a clear error on non-windowed modules."""
        L1 = data.metadata.forecast_split
        if L1 is None:
            raise ValueError(
                "CSDIAdapter requires a WINDOWED data module: "
                "data.metadata.forecast_split is None (Synthetic / Null modules "
                "leave it unset). Provide a module whose metadata sets it (== L1)."
            )
        return int(L1)

    def _make_csdi_batch(
        self, raw: dict, data: TimeSeriesDataModule, device: torch.device
    ) -> dict:
        """Map a raw loader batch to CSDI's ``(B, L, K)`` batch dict.

        First applies ``data.batch_transform`` (== ``parse_batch``), then permutes
        ``observed_data`` / ``observation_mask`` from ``(B, D, T)`` to
        ``(B, L, K)`` (CSDI's ``process_data`` permutes ``(B,L,K)->(B,K,L)``
        internally); ``timepoints`` pass as-is; ``gt_mask`` = observation mask with
        the forecast window (``t >= L1``) zeroed. Covariates are ignored.
        """
        L1 = self._forecast_split(data)
        batch = data.batch_transform(raw, device)
        observed = batch["observed_data"]  # (B, D, T)
        mask = batch["observation_mask"]  # (B, D, T)

        # (B, D, T) -> (B, L, K) with L=T, K=D (process_data re-permutes to (B,K,L)).
        observed_data = observed.permute(0, 2, 1).contiguous()
        observed_mask = mask.permute(0, 2, 1).contiguous()

        gt = observed_mask.clone()
        gt[:, L1:, :] = 0.0  # forecast window is imputed (not conditioned on)

        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "timepoints": batch["timepoints"],
            "gt_mask": gt,
        }

    # ---- fit -----------------------------------------------------------------
    def fit(
        self,
        *,
        data: TimeSeriesDataModule,
        training: TrainingScalars,
        device: torch.device,
        csv_log_path: str,
        tensorboard_dir: str,
        checkpoint_dir: str,
        hparams: ModelConfig | None = None,
        wandb_config: dict | None = None,
        model_config_yaml: str | None = None,
    ) -> None:
        """Train the vendored CSDI forecaster, logging ``loss/total`` (train+val)."""
        train_loader = data.train_loader()
        if train_loader is None:
            # NullDataModule: build-only, no fit (mirrors Experiment.train).
            log.info("[csdi] train_loader() is None (NullDataModule); fit no-op.")
            return

        cfg: CSDIConfig = hparams if hparams is not None else self.config
        if not isinstance(cfg, CSDIConfig):
            raise TypeError(f"CSDIAdapter.fit expects a CSDIConfig, got {type(cfg)!r}")

        # Enforce a windowed module up front (fails before any expensive work).
        self._forecast_split(data)

        if getattr(training, "amp", False):
            log.warning("[csdi] amp requested but ignored (not implemented).")
        if getattr(training, "profile_steps", 0):
            log.warning("[csdi] profile_steps requested but ignored (not implemented).")

        self._model_config_yaml = model_config_yaml
        self._build(cfg, device)

        steps = int(training.steps)
        log_every = int(training.log_every)
        if log_every <= 0:
            # log_every=0 would leave the CSV empty -> the objective reads +inf.
            raise ValueError("CSDIAdapter.fit requires log_every > 0.")
        validate_every = int(getattr(training, "validate_every", 0) or 0)
        checkpoint_every = getattr(training, "checkpoint_every", None)
        resume_from = getattr(training, "resume_from", None)

        self._optimizer = torch.optim.Adam(
            self.csdi.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
        milestones = sorted({
            max(1, math.floor(m * steps)) for m in cfg.lr_decay_milestones
        })
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, milestones=milestones, gamma=float(cfg.lr_decay_gamma)
        )
        self._global_step = 0

        # Resume BEFORE building loggers so a restored step count / rng is in
        # place; mirrors train.py's _safe_resume + _FRESH_START_FALLBACK marker.
        self._safe_resume(resume_from, device=device)

        metrics = self._build_metric_store(csv_log_path, tensorboard_dir, wandb_config)
        self._install_preempt_handlers()

        os.makedirs(checkpoint_dir, exist_ok=True)
        val_loader = data.val_loader()

        try:
            data_iter = iter(train_loader)
            self.csdi.train()
            while self._global_step < steps:
                try:
                    raw = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    raw = next(data_iter)

                csdi_batch = self._make_csdi_batch(raw, data, device)
                B = int(csdi_batch["observed_data"].shape[0])

                self._optimizer.zero_grad(set_to_none=True)
                loss = self.csdi(csdi_batch, is_train=1)
                loss.backward()
                if cfg.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.csdi.parameters(), float(cfg.clip_grad_norm)
                    )
                self._optimizer.step()
                self._scheduler.step()
                self._global_step += 1

                # Pass 0-d tensors (never .item()); MetricStore batches the sync.
                metrics.update("train", {"loss/total": loss.detach()}, weight=B)
                if self._global_step % log_every == 0:
                    metrics.step_end("train", self._global_step)

                if validate_every and (self._global_step % validate_every == 0):
                    self._validate(metrics, val_loader, data, device)

                if checkpoint_every and (self._global_step % checkpoint_every == 0):
                    self._save_ckpt(
                        os.path.join(
                            checkpoint_dir, f"ckpt_step{self._global_step}.pth"
                        )
                    )
                    self._save_ckpt(os.path.join(checkpoint_dir, "ckpt_latest.pth"))

                if self._preempt_pending:
                    path = os.path.join(checkpoint_dir, "ckpt_latest.pth")
                    self._save_ckpt(path)
                    raise _PreemptError(resume_from=str(path))

            # Final latest checkpoint so resume/eval always has a handle.
            self._save_ckpt(os.path.join(checkpoint_dir, "ckpt_latest.pth"))
        except KeyboardInterrupt:
            try:
                path = os.path.join(checkpoint_dir, "ckpt_interrupted_latest.pth")
                self._save_ckpt(path)
                log.warning("[csdi][interrupt] saved emergency checkpoint to %s", path)
            except Exception as e:  # pragma: no cover - best effort
                log.warning("[csdi][interrupt] failed to save checkpoint: %s", e)
            raise
        finally:
            metrics.close()

    def _build_metric_store(
        self, csv_log_path: str, tensorboard_dir: str, wandb_config: dict | None
    ) -> MetricStore:
        """MetricStore mirroring train.py: train=last, val=batch-weighted mean."""
        loggers: list = [
            TensorBoardLogger(log_dir=tensorboard_dir),
            CSVLogger(csv_log_path),
        ]
        if wandb_config:
            loggers.append(WandbLogger(**wandb_config))
        return MetricStore(
            spec=[MetricSpec("loss/*", "last")],
            split_spec={"val": [MetricSpec("loss/*", "mean")]},
            loggers=loggers,
        )

    def _validate(
        self,
        metrics: MetricStore,
        val_loader: object,
        data: TimeSeriesDataModule,
        device: torch.device,
    ) -> None:
        """Accumulate val loss over ALL batches, then one ``epoch_end``."""
        if val_loader is None:
            return
        self.csdi.eval()
        with torch.no_grad():
            for raw in val_loader:
                csdi_batch = self._make_csdi_batch(raw, data, device)
                B = int(csdi_batch["observed_data"].shape[0])
                vloss = self.csdi(csdi_batch, is_train=0)
                metrics.update("val", {"loss/total": vloss.detach()}, weight=B)
        metrics.epoch_end("val", self._global_step)
        self.csdi.train()

    def _install_preempt_handlers(self) -> None:
        """SIGUSR1/SIGTERM flip a flag (mirrors train.py's preempt handler)."""
        self._preempt_pending = False

        def _handler(signum: int, frame: object) -> None:
            self._preempt_pending = True

        for sig in (signal.SIGUSR1, signal.SIGTERM):
            # Non-main thread / unsupported platform — skip silently.
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, _handler)

    # ---- resume --------------------------------------------------------------
    def _safe_resume(self, resume_from: str | None, *, device: torch.device) -> None:
        """Restore model+opt+sched+step+rng, mirroring train.py's fallback path."""
        if resume_from is None:
            return
        try:
            self._restore_from_checkpoint(resume_from, device=device)
            log.info("[csdi][resume] global_step=%d", self._global_step)
        except (FileNotFoundError, IsADirectoryError, NoUsableCheckpointError) as e:
            self._global_step = 0
            log.error(
                "[csdi][resume] FALLBACK TO FRESH START — failed to load "
                "checkpoint %r: %s: %s. global_step reset to 0.",
                resume_from,
                type(e).__name__,
                e,
            )
            try:
                marker = os.path.join(
                    os.path.dirname(resume_from) or ".", "_FRESH_START_FALLBACK.txt"
                )
                with open(marker, "a") as f:
                    f.write(
                        f"{datetime.datetime.now().isoformat()}\t"
                        f"resume_from={resume_from}\t{type(e).__name__}: {e}\n"
                    )
            except OSError:
                pass

    def _restore_from_checkpoint(self, path: str, *, device: torch.device) -> None:
        """Restore full training state (model/opt/sched/step/rng) from a ckpt."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path!r}")
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except (EOFError, RuntimeError, OSError) as e:
            raise NoUsableCheckpointError(
                f"failed to read checkpoint {path!r}: {type(e).__name__}: {e}"
            ) from e
        self._reject_foreign_format(payload)

        self.csdi.load_state_dict(payload["model_state"])
        self._ensure_device(device)
        if self._optimizer is not None and payload.get("optimizer_state") is not None:
            self._optimizer.load_state_dict(payload["optimizer_state"])
        if self._scheduler is not None and payload.get("scheduler_state") is not None:
            self._scheduler.load_state_dict(payload["scheduler_state"])
        self._global_step = int(payload.get("global_step", 0))
        self._restore_rng(payload.get("rng_state"))

    @staticmethod
    def _restore_rng(rng_state: dict | None) -> None:
        if not rng_state:
            return
        try:
            torch.set_rng_state(rng_state["torch_cpu"])
            if torch.cuda.is_available() and rng_state.get("torch_cuda"):
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            np.random.set_state(rng_state["numpy"])  # noqa: NPY002 (legacy global RNG, mirrors checkpoint.py)
            random.setstate(rng_state["python"])
        except (KeyError, TypeError, ValueError) as e:  # pragma: no cover - defensive
            log.warning("[csdi][resume] could not restore RNG state: %s", e)

    # ---- forecast ------------------------------------------------------------
    @torch.no_grad()
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
        """Roll out a probabilistic forecast in NORMALIZED space (no re-scaling).

        ``x_hist`` is ``(B, D, L1)``; the future is a zeroed tail. Returns
        ``pred_mean`` ``(B, D, L2)`` (median over samples) and ``pred_samples``
        ``(B, S, D, L2)``.
        """
        del (
            past_covariates,
            future_covariates,
            static_covariates,
        )  # ignored (limitation)

        L1 = int(x_hist.shape[-1])
        L2 = int(future_time.shape[-1])
        if L1 == 0:
            raise ValueError("CSDIAdapter.forecast: forecast_split is None / L1==0.")
        if self.csdi is None:
            raise RuntimeError(
                "CSDIAdapter.forecast called before build; fit / load_checkpoint first."
            )
        device = x_hist.device
        self._device = device
        self._ensure_device(device)
        self.csdi.eval()

        B, D, _ = x_hist.shape
        # Full window = history + zeroed future.
        future_zeros = x_hist.new_zeros(B, D, L2)
        observed = torch.cat([x_hist, future_zeros], dim=-1)  # (B, D, T)
        # (B, D, T) -> (B, L, K) for process_data.
        observed_data = observed.permute(0, 2, 1).contiguous()

        # Observed mask: history real, future all-ones (CSDI's evaluate slices the
        # tail through gt_mask; observed_mask only gates the target_mask return).
        hist_mask = x_mask.permute(0, 2, 1).contiguous()  # (B, L1, K)
        future_mask = observed_data.new_ones(B, L2, D)
        observed_mask = torch.cat([hist_mask, future_mask], dim=1)  # (B, T, K)

        timepoints = torch.cat([past_time, future_time], dim=1)  # (B, T)

        # gt_mask history-only (forecast window is imputed).
        gt_mask = observed_mask.clone()
        gt_mask[:, L1:, :] = 0.0

        csdi_batch = {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "timepoints": timepoints,
            "gt_mask": gt_mask,
        }
        # evaluate returns (B, S, K, L); slice the forecast time tail.
        samples, *_ = self.csdi.evaluate(csdi_batch, int(num_samples))
        pred_samples = samples[..., L1:]  # (B, S, K=D, L2)
        pred_mean = pred_samples.median(dim=1).values  # (B, D, L2)
        return {"pred_mean": pred_mean, "pred_samples": pred_samples}

    # ---- checkpointing -------------------------------------------------------
    def _payload(self) -> dict:
        """Build the ``csdi_ckpt_v1`` payload (model/opt/sched/step/rng/config)."""
        cfg_yaml = self._model_config_yaml
        if cfg_yaml is None and self._built_config is not None:
            cfg_yaml = OmegaConf.to_yaml(OmegaConf.structured(self._built_config))
        return {
            "_format": _CSDI_FORMAT,
            "model_state": self.csdi.state_dict(),
            "optimizer_state": (
                self._optimizer.state_dict() if self._optimizer is not None else None
            ),
            "scheduler_state": (
                self._scheduler.state_dict() if self._scheduler is not None else None
            ),
            "global_step": int(self._global_step),
            "rng_state": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
                "numpy": np.random.get_state(),  # noqa: NPY002 (legacy global RNG, mirrors checkpoint.py)
                "python": random.getstate(),
            },
            "model_config_yaml": cfg_yaml,
        }

    def _save_ckpt(self, path: str) -> None:
        atomic_save(self._payload(), path)

    def save_checkpoint(self, path: str) -> None:
        """Serialize the owned CSDI module + training state to ``path``."""
        if self.csdi is None:
            raise RuntimeError(
                "CSDIAdapter.save_checkpoint before build (csdi is None); "
                "call fit / load_checkpoint first."
            )
        self._save_ckpt(path)

    @staticmethod
    def _reject_foreign_format(payload: object) -> None:
        """Reject any payload whose ``_format`` isn't ``csdi_ckpt_v1``."""
        if not isinstance(payload, dict):
            raise ValueError(
                f"CSDIAdapter checkpoint must be a dict payload; got {type(payload)!r}."
            )
        fmt = payload.get("_format")
        if fmt != _CSDI_FORMAT:
            raise ValueError(
                f"CSDIAdapter refuses to load checkpoint with _format={fmt!r}; "
                f"expected {_CSDI_FORMAT!r}."
            )

    def load_checkpoint(
        self,
        path: str,
        *,
        device: torch.device,
        hparams: ModelConfig | None = None,
        load_ema: bool = True,
        expected_model_config_yaml: str | None = None,
        strict: bool = False,
    ) -> None:
        """Restore state from ``path``.

        Order: ``_build`` -> ``load_state_dict`` -> ``.to(device)`` -> re-apply
        ``_ensure_device``. Rebuilds the module from ``hparams or self.config``
        (hparams wins for arch). A cross-format payload raises ``ValueError``;
        ``load_ema=True`` warns that CSDI has no EMA state.
        """
        if load_ema:
            log.warning("[csdi] load_ema=True requested but CSDI has no EMA state.")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path!r}")
        payload = torch.load(path, map_location=device, weights_only=False)
        self._reject_foreign_format(payload)

        cfg: CSDIConfig = hparams if hparams is not None else self.config
        if not isinstance(cfg, CSDIConfig):
            raise TypeError(
                f"CSDIAdapter.load_checkpoint expects a CSDIConfig, got {type(cfg)!r}"
            )

        check_model_config_drift(
            payload.get("model_config_yaml"), expected_model_config_yaml
        )

        # _build -> load_state_dict -> .to(device) -> re-apply _ensure_device.
        self._build(cfg, device)
        self.csdi.load_state_dict(payload["model_state"], strict=strict)
        self.csdi.to(device)
        self._ensure_device(device)
        self._global_step = int(payload.get("global_step", 0))
        self._model_config_yaml = payload.get("model_config_yaml")
