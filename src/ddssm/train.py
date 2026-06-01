"""DDSSMTrainer: training loop, EMA, logging, and the fit lifecycle.

The ``.pth`` payload schema lives in :mod:`ddssm.checkpoint`; the
trainer's ``save_checkpoint`` / ``restore_from_checkpoint`` delegate
there.
"""

import os
import math
import pickle
import shutil
import signal
from typing import TYPE_CHECKING, Any, Callable, final
import logging
import tempfile
from contextlib import nullcontext, contextmanager
from collections import deque

import torch

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .stages import EarlyStopSpec

from torch import optim
from hydra_zen import builds
from omegaconf import MISSING
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader

from .dssd import DDSSM_base
from .loggers import (
    CSVLogger,
    MetricSpec,
    MetricStore,
    WandbLogger,
    ConsoleLogger,
    TensorBoardLogger,
)
from .train_utils import (
    param_groups_for_adamw,
)


class PreemptError(RuntimeError):
    """Raised by :class:`DDSSMTrainer.fit` when a preempt signal was caught.

    Carries the absolute path to the checkpoint written immediately before
    the raise so a downstream caller (e.g. ``ddssm.app``) can stash it on
    the Optuna trial's ``user_attrs["resume_from"]`` and the retry can
    resume mid-trial from that ckpt.

    See ADR-0009 for the full preemption-aware launch design.
    """

    def __init__(self, resume_from: str):
        super().__init__(f"training preempted; resume_from={resume_from}")
        self.resume_from = resume_from


class EMA:
    """Exponential moving average of a module's parameters.

    Maintains a shadow copy of all parameter tensors and blends them toward
    the live weights after each update step.  The ``swap`` context manager
    temporarily applies the EMA weights for inference, then restores the live
    weights on exit.

    Args:
        module: The ``nn.Module`` whose parameters to track.
        decay: EMA decay factor (closer to 1 → slower update).
    """

    def __init__(self, module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in module.state_dict().items()}
        self._module = module

    @torch.no_grad()
    def update(self):
        msd = self._module.state_dict()
        for k, v in msd.items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    @contextmanager
    def swap(self):
        msd = self._module.state_dict()
        backup = {k: v.detach().clone() for k, v in msd.items()}
        self._module.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            self._module.load_state_dict(backup, strict=True)


@final
class DDSSMTrainer:
    """Training harness for ``DDSSM_base``.

    Handles the full training lifecycle: gradient accumulation, optional AMP,
    EMA tracking, scheduled λ weighting, step-level metric logging to
    CSV and TensorBoard, and atomic checkpoint save/restore.

    Args:
        model: The ``DDSSM_base`` model to train.
        device: Device on which the model and batches live.
        optimizer: Optional pre-built optimizer.  If ``None``, an AdamW
            optimizer with per-component learning rates is created from the
            model config.
        csv_log_path: If given, step metrics are appended to this CSV file.
        tensorboard_dir: Directory for TensorBoard ``SummaryWriter`` output.
        wandb_config: When provided, a :class:`WandbLogger` is added to the
            metric store.  Pass a dict with any subset of the keyword
            arguments accepted by :class:`WandbLogger` (``project``,
            ``entity``, ``name``, ``tags``, ``config``, ``base_url``,
            ``enabled``).  Example::

                wandb_config = {
                    "project": "ddssm",
                    "entity": "my-team",
                    "base_url": "https://wandb.example.com",
                }

        quiet: If ``True``, suppress console logging.
    """

    def __init__(
        self,
        model: "DDSSM_base",
        device: torch.device,
        hparams: Any = None,
        optimizer: optim.Optimizer | None = None,
        csv_log_path: str | None = None,
        tensorboard_dir: str = "runs/ddssm",
        checkpoint_dir: str = "./checkpoints",
        wandb_config: dict | None = None,
        quiet: bool = False,
        model_config_yaml: str | None = None,
    ):
        self.model = model.to(device)
        self.device = device

        # Per ADR-0004: hparams flow directly into the trainer (no
        # longer routed through ``model.config.hyperparams``). When
        # ``None``, fall back to the project defaults — keeps direct
        # ``DDSSMTrainer(model, device)`` in tests and notebooks
        # working without spelling out optimisation knobs.
        if hparams is None:
            from .dssd import _default_hyperparams
            hparams = _default_hyperparams()
        self.hparams = hparams

        # ADR-0005: resolved YAML of the model's hydra-zen builds()
        # config. Persisted into checkpoints so the load path can warn
        # on silent semantic drift (shapes preserved, builder semantics
        # changed). ``None`` for ad-hoc constructions (tests, notebooks)
        # that have no Hydra config to serialise.
        self._model_config_yaml: str | None = model_config_yaml

        self.global_step = 0
        # Per-stage λ schedule installed by ``StageOrchestrator`` before
        # each stage. Retained for backwards-compat introspection by
        # tests; the loss object now owns scheduling shape per
        # ADR-0004. Single-fit runs use the constant λ inside the
        # default ``FullELBO``.
        self._stage_lambda_fn: Callable[[int], float] | None = None
        self._stage_start_step: int = 0
        # 1-based index of the stage currently running (set by
        # StageOrchestrator); 0 for single-fit runs with no stages. Logged as
        # ``stage/idx`` so multi-stage curves are interpretable from the CSV.
        self._current_stage_idx: int = 0
        # Opt-in fail-fast: when True, a non-finite optimized loss raises
        # instead of silently NaN-poisoning the weights. Off by default so
        # existing runs are unchanged; the logger always counts it regardless.
        self.abort_on_nonfinite_loss: bool = False
        # ADR-0004: active loss object (installed by orchestrator per
        # stage; constructed lazily at fit() start otherwise).
        from .losses import Loss
        self._active_loss: Loss | None = None

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                param_groups_for_adamw(
                    self.model,
                    enc_lr=self.hparams.enc_lr,
                    dec_lr=self.hparams.dec_lr,
                    trans_lr=self.hparams.trans_lr,
                    weight_decay=self.hparams.weight_decay,
                ),
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        self.scheduler = None
        # Hoisted out of fit() so save_checkpoint can persist its state via
        # ``Checkpoint.from_trainer``. bf16 autocast doesn't need scaling, so
        # the default scaler stays disabled and ``scaler.state_dict()`` is
        # NOT written to disk (see ``Checkpoint.from_trainer``); enable it
        # externally if you wire in fp16 AMP.
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.weight_decay = self.hparams.weight_decay

        self.grad_accum_steps = self.hparams.grad_accum_steps
        self.clip_grad_norm = self.hparams.clip_grad_norm

        self.ema_decay = self.hparams.ema_decay
        # EMA on the denoiser (used at sampling time)
        self.ema = EMA(self.model.transition, decay=self.ema_decay)

        loggers = [
            TensorBoardLogger(log_dir=tensorboard_dir),
        ]  # epoch-only by default
        if csv_log_path:
            loggers.append(CSVLogger(path=csv_log_path))
        if wandb_config is not None:
            wandb_logger = WandbLogger(**wandb_config)
            # ``wandb.watch`` is opt-in via ``watch_log`` in wandb_config;
            # when set the logger registers grad / param hooks on the
            # live model so W&B records histograms next to the scalars.
            wandb_logger.watch_model(self.model)
            loggers.append(wandb_logger)
        if not quiet:
            loggers.append(
                ConsoleLogger(every_steps=0),
            )
        self.metrics = MetricStore(
            spec=[
                MetricSpec("loss/*", "last"),
                MetricSpec("time/*", "last"),
                MetricSpec("optim/*", "last"),
                MetricSpec("mem/*", "last"),
                MetricSpec("stage/*", "last"),
            ],
            # Validation accumulates over the whole val set within one
            # ``epoch_end`` flush, so losses are mean-reduced (weighted by
            # batch size) rather than reported from the last batch.
            split_spec={
                "val": [
                    MetricSpec("loss/*", "mean"),
                    MetricSpec("optim/*", "last"),
                    MetricSpec("stage/*", "last"),
                    MetricSpec("*", "mean"),
                ],
            },
            loggers=loggers,
        )

        self.checkpoint_dir = checkpoint_dir

        # Preempt-aware signal handling (ADR-0009). The flag is set by the
        # signal handler and checked from the fit() step loop; on set, fit()
        # saves a checkpoint and raises PreemptError. signal.signal() must
        # be called from the main thread, so we wrap in try/except to
        # tolerate trainer construction in worker threads (e.g. notebooks,
        # parallel test runners).
        self._preempt_pending: bool = False
        for _sig in (signal.SIGUSR1, signal.SIGTERM):
            try:
                signal.signal(_sig, self._handle_preempt_signal)
            except (ValueError, OSError):
                # Non-main thread, or platform without this signal — silently
                # skip. The preempt path only activates under the SLURM
                # launcher which guarantees a main-thread trainer.
                pass
        # Under DDSSM_PREEMPTIVE=1 the launcher's bash preamble forwards
        # SIGINT to the worker too (single-job preempt + Ctrl-C parity for
        # the --local path). Without the env var, leave SIGINT alone so
        # Python's default KeyboardInterrupt path stays intact.
        if os.environ.get("DDSSM_PREEMPTIVE") == "1":
            try:
                signal.signal(signal.SIGINT, self._handle_preempt_signal)
            except (ValueError, OSError):
                pass

    def _handle_preempt_signal(self, signum, frame):
        """Async-signal-safe preempt handler: flip a flag and return.

        MUST NOT call into torch, do any logging, or touch CUDA — the
        actual checkpoint save + raise happens from the fit() loop, where
        we're between optimizer steps and it's safe to do real work.
        """
        self._preempt_pending = True

    def _set_trainable(self, t):
        """Toggle ``requires_grad`` per submodule from a stage trainable mask.

        This is the single gradient-suppression mechanism: the forward pass
        always computes every ELBO term, but frozen submodules accumulate no
        gradients. ``static_embeddings`` and ``aux_posterior`` are part of the
        encoder family and share ``t.encoder``.

        Args:
            t: A ``StageTrainable``-like object with ``encoder`` / ``decoder``
                / ``transition`` / ``baseline`` boolean flags.
        """

        def maybe_flag(mod, flag: bool):
            if mod is None:
                return
            for p in mod.parameters():
                p.requires_grad = flag

        # Expect these attributes to exist (guard with hasattr)
        maybe_flag(getattr(self.model, "encoder", None), t.encoder)
        maybe_flag(getattr(self.model, "decoder", None), t.decoder)
        maybe_flag(getattr(self.model, "transition", None), t.transition)
        maybe_flag(getattr(self.model, "static_embeddings", None), t.encoder)
        # aux_posterior is part of the encoder family (q_Φ in the doc).
        maybe_flag(getattr(self.model, "aux_posterior", None), t.encoder)
        # Baseline μ_p — declarative per-stage flag.  Stage 1 trains it,
        # stage 2 freezes it under Pinned mode (matches the imperative
        # freeze in :func:`perform_centering_handoff`).  Default ``True``
        # keeps legacy models that lack a baseline a no-op.
        baseline_flag = getattr(t, "baseline", True)
        maybe_flag(getattr(self.model, "baseline", None), baseline_flag)

    def _rebuild_optimizer(
        self,
        lrs,
    ):
        """Rebuild the AdamW optimizer with per-component stage learning rates.

        Args:
            lrs: A ``StageLrs``-like object with ``enc_lr`` / ``dec_lr`` /
                ``trans_lr`` learning rates.
        """
        groups = param_groups_for_adamw(
            self.model,
            enc_lr=lrs.enc_lr,
            dec_lr=lrs.dec_lr,
            trans_lr=lrs.trans_lr,
            weight_decay=self.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    # ------------------------
    # Serialization / Checkpoint  (schema owned by ddssm.checkpoint)
    # ------------------------
    def save_checkpoint(
        self, path: str, *, stage_prefix: str | None = None,
    ) -> None:
        """Persist trainer state via :mod:`ddssm.checkpoint`.

        ``stage_prefix`` is forwarded to the payload so multi-stage resume
        (ADR-0009) can identify the originating stage on retry.
        """
        from .checkpoint import save as _save
        _save(self, path, stage_prefix=stage_prefix)

    def restore_from_checkpoint(self, path: str, strict: bool = True) -> dict:
        """Resume: load model weights + optimiser + EMA tracker + step.

        Loads *live* weights into the model (``load_ema=False``); the
        EMA shadows go back into the trainer's EMA tracker, not into the
        transition, so training continues exactly where it left off.

        Also restores GradScaler and LR-scheduler state when present.
        The save/restore sides must agree on whether each subsystem is
        live — see the contract guards below. v1 checkpoints predate
        these fields, so a missing entry on disk only fails the guard
        when the live trainer has the corresponding subsystem enabled.
        """
        from .checkpoint import load_into_model

        ckpt = load_into_model(
            self.model, path, device=self.device, strict=strict, load_ema=False,
        )
        if ckpt.optimizer_state is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt.optimizer_state)
        else:
            print(
                "[restore] Warning: optimizer state not found in checkpoint "
                "or optimizer is None."
            )
        if ckpt.ema_state is not None and hasattr(self, "ema"):
            self.ema.shadow = ckpt.ema_state
            if ckpt.ema_decay is not None:
                self.ema_decay = ckpt.ema_decay
        # GradScaler contract guard. A non-None saved state means the
        # producer was running fp16 AMP; silently dropping it on a
        # disabled-scaler live trainer would bias the restart, so raise.
        # The inverse — live scaling enabled but no saved state — would
        # restart from default scale factors mid-run and is just as bad.
        live_scaler = getattr(self, "scaler", None)
        live_scaler_enabled = live_scaler is not None and live_scaler.is_enabled()
        if ckpt.scaler_state is not None and not live_scaler_enabled:
            raise RuntimeError(
                "Checkpoint carries GradScaler state but the live trainer "
                "has scaling disabled; refusing to drop state silently. "
                "Enable AMP/scaling on the live trainer before resuming."
            )
        if ckpt.scaler_state is None and live_scaler_enabled:
            raise RuntimeError(
                "Live trainer has GradScaler enabled but the checkpoint "
                "carries no scaler state; refusing to restart scale factors "
                "from defaults mid-run."
            )
        if ckpt.scaler_state is not None and live_scaler is not None:
            live_scaler.load_state_dict(ckpt.scaler_state)
        # LR-scheduler contract guard — same logic as the scaler.
        live_scheduler = getattr(self, "scheduler", None)
        if ckpt.scheduler_state is not None and live_scheduler is None:
            raise RuntimeError(
                "Checkpoint carries LR scheduler state but the live trainer "
                "has no scheduler; refusing to drop state silently."
            )
        if ckpt.scheduler_state is None and live_scheduler is not None:
            raise RuntimeError(
                "Live trainer has an LR scheduler but the checkpoint carries "
                "no scheduler state; refusing to restart the schedule from "
                "step 0 mid-run."
            )
        if ckpt.scheduler_state is not None and live_scheduler is not None:
            live_scheduler.load_state_dict(ckpt.scheduler_state)
        self.global_step = ckpt.global_step
        return {"grad_accum_steps": ckpt.grad_accum_steps}

    def _build_default_loss(self):
        """Default loss for single-fit runs: full ELBO with no rate ramp.

        Multi-stage runs receive per-stage loss objects from
        ``StageOrchestrator``; a single ``fit()`` with no declared loss
        falls back here. Post-ADR-0004 all λ-shape config lives on loss
        objects (not ``Hparams``), so the single-fit default is simply
        the unramped full ELBO.
        """
        from .losses import FullELBO

        return FullELBO(rate_lambda=lambda _step: 1.0)

    def _move_batch_to_device(self, batch: dict, device: torch.device) -> dict:
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _prepare_batch(
        self,
        batch: dict,
        device: torch.device,
        batch_transform: Callable[[dict, torch.device], dict] | None,
    ) -> dict:
        if batch_transform is not None:
            return batch_transform(batch, device)
        return self._move_batch_to_device(batch, device)

    def _compute_loss_and_metrics(
        self,
        batch: dict,
        amp: bool,
    ):
        observed = batch["observed_data"]
        observed_mask = batch["observation_mask"]
        timepoints = batch["timepoints"]
        covariates = batch.get("covariates")
        static_covariates = batch.get("static_covariates")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            components, metrics, _stats = self.model(
                observed,
                observed_mask,
                timepoints,
                covariates=covariates,
                static_covariates=static_covariates,
                train=True,
            )

        # Per ADR-0004: the active loss object owns the rate-λ schedule;
        # the trainer just drives the step. Within an orchestrator stage
        # the step counts from the stage boundary; in single-fit runs
        # the orchestrator never sets ``_stage_start_step``, so it stays
        # at 0 and the loss sees the global step directly.
        step_within_stage = self.global_step - self._stage_start_step + 1
        assert self._active_loss is not None, (
            "Trainer.fit must install a default loss object before training"
        )
        loss = self._active_loss(components, step_within_stage)
        # Surface the rate-λ for logging for ANY loss with a schedule (not
        # just FullELBO) — the loss reports its own λ via ``lambda_at``.
        lam = self._active_loss.lambda_at(step_within_stage)
        if lam is not None:
            metrics["optim/lambda"] = torch.tensor(lam)

        return loss, metrics, observed.size(0)

    def _backward_loss(self, loss: torch.Tensor, scaler, amp: bool):
        if amp:
            scaler.scale(loss / self.grad_accum_steps).backward()
        else:
            (loss / self.grad_accum_steps).backward()

    def _accumulate_metrics(self, accum_metrics, metrics: dict):
        if accum_metrics is None:
            return {k: v.detach() for k, v in metrics.items()}
        for k, v in metrics.items():
            accum_metrics[k] = accum_metrics[k] + v.detach()
        return accum_metrics

    def _optimizer_step(self, scaler, amp: bool):
        # Optional global grad-norm clip (``hparams.clip_grad_norm``).
        # Follows the AMP unscale→clip→step order; ``scaler`` is disabled
        # in this trainer so ``unscale_`` is a no-op but keeps the order
        # correct if scaling is ever re-enabled.
        if self.clip_grad_norm is not None:
            if amp:
                scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.clip_grad_norm)
            )
        if amp:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        if hasattr(self, "ema") and self.ema is not None:
            with torch.no_grad():
                self.ema.update()

    def _finalize_accum_metrics(self, accum_metrics):
        if accum_metrics is None:
            return {}
        for k in list(accum_metrics.keys()):
            accum_metrics[k] = accum_metrics[k] / self.grad_accum_steps
        return accum_metrics

    def _log_train_step(
        self,
        step: int,
        log_every: int,
        accum_loss: float,
        accum_metrics: dict,
        accum_weight: int,
        device: torch.device,
    ):
        import time as _time

        self.global_step += 1
        # ``accum_metrics`` carries the model's *unweighted* ELBO under
        # "loss/total" (distortion + rate). Spread it first, then override
        # "loss/total" with the actually-optimized objective
        # (distortion + λ·rate) — the value the early-stop window tracks, so
        # the logged curve and the stop criterion agree. The unweighted ELBO
        # is preserved under "loss/total_unweighted".
        log_values = dict(accum_metrics)
        if "loss/total" in log_values:
            log_values["loss/total_unweighted"] = log_values["loss/total"]
        log_values["loss/total"] = torch.tensor(
            accum_loss / self.grad_accum_steps, device=device
        )
        # Wall-clock elapsed since the metric store was created
        # (typically equals trainer-start time).  Surfaces ``time/elapsed_s``
        # as a CSV column so the eval ``wallclock_to_target`` metric can
        # find the time at which a metric first crossed a threshold.
        log_values["time/elapsed_s"] = torch.tensor(
            _time.time() - self.metrics._t0, device=device
        )
        # Stage markers so a multi-stage curve is interpretable from the CSV
        # alone: which stage a row belongs to, and the stage-relative step
        # (where the λ-ramp reset / centering handoff fired). 0 for single-fit.
        log_values["stage/idx"] = torch.tensor(
            float(self._current_stage_idx), device=device
        )
        log_values["stage/step_within"] = torch.tensor(
            float(self.global_step - self._stage_start_step), device=device
        )
        self.metrics.update(split="train", values=log_values, weight=accum_weight)
        if log_every and (step % log_every == 0):
            self.metrics.step_end("train", self.global_step)

    def _run_validation(
        self,
        val_loader: DataLoader,
        batch_transform: Callable[[dict, torch.device], dict] | None,
        device: torch.device,
    ):
        self.model.eval()
        # Validate on the EMA model — the same transition weights the
        # sampling path uses (ADR-0005). ``swap`` loads the EMA shadows
        # for the duration of the loop and restores the live weights on
        # exit so training continues unperturbed.
        ema_ctx = (
            self.ema.swap()
            if getattr(self, "ema", None) is not None
            else nullcontext()
        )
        with torch.no_grad(), ema_ctx:
            for vbatch in val_loader:
                vbatch = self._prepare_batch(vbatch, device, batch_transform)

                vwin = vbatch["observed_data"]
                vmask = vbatch["observation_mask"]
                vtime = vbatch["timepoints"]
                vcov = vbatch.get("covariates", None)
                vstatic_cov = vbatch.get("static_covariates", None)

                vcomponents, vmetrics, _ = self.model(
                    vwin,
                    vmask,
                    vtime,
                    covariates=vcov,
                    static_covariates=vstatic_cov,
                    train=False,
                )
                # Validation reports the loss object's scalar at the
                # current step (per ADR-0004).  Single-fit ``_stage_start_step``
                # is 0, so this collapses to the global step.
                vstep = self.global_step - self._stage_start_step + 1
                assert self._active_loss is not None
                vloss = self._active_loss(vcomponents, vstep)
                # Mirror _log_train_step: report the optimized objective as
                # loss/total and keep the model's unweighted ELBO under
                # loss/total_unweighted (vmetrics carries the latter).
                vlog = dict(vmetrics)
                if "loss/total" in vlog:
                    vlog["loss/total_unweighted"] = vlog["loss/total"]
                vlog["loss/total"] = vloss
                self.metrics.update("val", values=vlog, weight=vwin.size(0))

    def _maybe_run_validation(
        self,
        step: int,
        val_loader: DataLoader | None,
        validate_every: int,
        batch_transform: Callable[[dict, torch.device], dict] | None,
        device: torch.device,
    ):
        if val_loader is None or not validate_every or (step % validate_every != 0):
            return
        self._run_validation(
            val_loader=val_loader,
            batch_transform=batch_transform,
            device=device,
        )
        self.metrics.epoch_end("val", self.global_step)

    def _save_periodic_checkpoint(
        self, step: int, checkpoint_prefix: str | None
    ) -> str:
        """Persist step-N + latest checkpoints. Returns the latest path.

        The returned path is the absolute path to ``ckpt_<prefix>_latest.pth``
        (or ``ckpt_latest.pth`` for the unprefixed case), suitable for
        passing to ``restore_from_checkpoint`` / ``fit(resume_from=...)``.
        ADR-0009's preempt path relies on this return value as the
        ``PreemptError.resume_from`` payload.
        """
        if checkpoint_prefix is None:
            ckpt_name = f"ckpt_step{step}.pth"
            latest_name = "ckpt_latest.pth"
        else:
            ckpt_name = f"ckpt_{checkpoint_prefix}_step{step}.pth"
            latest_name = f"ckpt_{checkpoint_prefix}_latest.pth"

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_name = os.path.join(self.checkpoint_dir, ckpt_name)
        latest_name = os.path.join(self.checkpoint_dir, latest_name)
        # ADR-0009: stamp the originating stage prefix into the payload so a
        # preempt-retry's StageOrchestrator can identify which stage produced
        # this ckpt and resume into the right one.
        self.save_checkpoint(ckpt_name, stage_prefix=checkpoint_prefix)
        # The pair (step-N, latest) must be atomic at the FS level: a SIGKILL
        # between the two writes would otherwise leave ``ckpt_step{N}`` on disk
        # while ``ckpt_latest`` still points to the previous N-K snapshot, and
        # a preempt-retry would silently lose progress. We mirror step-N to
        # latest by copying into a same-dir tmp file then ``os.replace`` —
        # which is atomic on POSIX, and unlike ``os.link`` is safe on the
        # shared FS (Lustre/NFS) the Tempest cluster runs on.
        d = os.path.dirname(latest_name) or "."
        f = tempfile.NamedTemporaryFile(
            prefix="tmp_latest_", suffix=".pth", dir=d, delete=False,
        )
        tmppath = f.name
        f.close()
        try:
            shutil.copyfile(ckpt_name, tmppath)
            os.replace(tmppath, latest_name)
        except Exception:
            try:
                os.remove(tmppath)
            except OSError:
                pass
            raise
        return latest_name

    def _maybe_save_checkpoint(
        self,
        step: int,
        checkpoint_every: int | None,
        checkpoint_prefix: str | None,
    ):
        if checkpoint_every and (step % checkpoint_every == 0):
            self._save_periodic_checkpoint(
                step=step, checkpoint_prefix=checkpoint_prefix
            )

    # Narrow set of exceptions that legitimately mean "no usable checkpoint
    # on disk" — anything outside this set is a real bug that must surface
    # loudly rather than silently restart from step 0 (ADR-0009 preempt
    # path: a 6-hour stage-2 trial that hits a schema-incompatible ckpt
    # would otherwise lose all progress without a trace).
    _RESUME_NO_CKPT_EXCEPTIONS: tuple[type[BaseException], ...] = (
        FileNotFoundError,    # missing file
        IsADirectoryError,    # path points to a directory
        EOFError,             # truncated pickle stream
        pickle.UnpicklingError,  # malformed pickle
        RuntimeError,         # torch.load corrupt-zip / state_dict shape drift
        KeyError,             # payload missing expected keys
        AttributeError,       # payload structure unexpected (e.g. not a dict)
    )

    def _safe_resume(self, resume_from: str | None):
        if resume_from is None:
            return
        try:
            self.restore_from_checkpoint(resume_from, strict=True)
            log.info("[resume] global_step=%d", self.global_step)
        except self._RESUME_NO_CKPT_EXCEPTIONS as e:
            # Falling back to fresh start MUST be loud — a silent restart
            # in a preempt-retry means hours of training are gone with no
            # trace in the run logs.
            self.global_step = 0
            log.warning(
                "[resume] FALLBACK TO FRESH START — failed to load "
                "checkpoint %r: %s: %s. global_step reset to 0.",
                resume_from, type(e).__name__, e,
            )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        total_steps: int = 10_000,
        validate_every: int = 1_000,
        log_every: int = 10,  # step-level logging
        checkpoint_every: int | None = None,
        checkpoint_prefix: str | None = None,
        amp: bool = False,
        # resume controls
        resume_from: str | None = None,  # path to ckpt to resume from
        batch_transform: Callable[[dict, torch.device], dict] | None = None,
        profile_steps: int = 0,
        early_stop: "EarlyStopSpec | None" = None,
    ) -> int:
        """Run the training loop. One optimizer step counts as one step.

        Validation, checkpoints, and logs are triggered by step counts. Uses
        gradient accumulation and optional AMP for memory efficiency.

        Args:
            train_loader: Training data loader (re-iterated on exhaustion).
            val_loader: Optional validation loader; skipped when ``None``.
            total_steps: Cumulative max global step to stop at (not per-stage).
            validate_every: Run validation every N steps (0 disables).
            log_every: Flush step-level metrics every N steps.
            checkpoint_every: Save a periodic checkpoint every N steps.
            checkpoint_prefix: Filename prefix for periodic / stage checkpoints.
            amp: Enable bf16 autocast (the GradScaler stays disabled for bf16).
            resume_from: Path to a checkpoint to resume from; restores
                ``global_step``, optimizer, and EMA when present.
            batch_transform: Optional per-batch transform applied on device.
            profile_steps: Profile up to this many optimizer steps when > 0.
            early_stop: When an enabled :class:`EarlyStopSpec`, the loop exits
                early once the rolling-window improvement of ``loss/total``
                falls below ``min_improvement``.

        Returns:
            The global step at which the loop exited.

        Raises:
            PreemptError: A SIGUSR1/SIGTERM (or SIGINT under
                ``DDSSM_PREEMPTIVE=1``) was caught; a checkpoint is saved and
                its path is carried on ``resume_from``.
            FloatingPointError: ``abort_on_nonfinite_loss`` is set and the
                accumulated loss is non-finite.
        """
        device = self.device
        self.model.to(device)

        # Per ADR-0004: ensure an active loss object exists. The
        # orchestrator installs one per stage; for single-fit runs we
        # build a default ``FullELBO`` from hparams here.
        if self._active_loss is None:
            self._active_loss = self._build_default_loss()

        # Rolling window for the ELBO-plateau early-stop check.
        es_active = bool(early_stop and early_stop.enabled)
        loss_window: deque[float] = (
            deque(maxlen=int(early_stop.window)) if es_active else deque()
        )
        early_stop_triggered = False

        # Make the console logger print every `log_every` steps
        for lg in self.metrics.loggers:
            if isinstance(lg, ConsoleLogger):
                lg.every_steps = log_every

        # see if we should resume
        self._safe_resume(resume_from)

        data_iter = iter(train_loader)
        # The autocast above uses bf16, which has fp32's exponent range and
        # needs no gradient scaling — so the scaler stays disabled even under
        # AMP. (GradScaler exists for fp16 underflow; enabling it for bf16 is
        # dead work and bit-identical to disabled.) The amp branches in
        # _backward_loss / _optimizer_step then pass through correctly.
        # The scaler lives on ``self`` so its state (when enabled) round-trips
        # through ``save_checkpoint`` / ``restore_from_checkpoint``.
        scaler = self.scaler
        start_step = self.global_step

        do_profile = profile_steps > 0
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        if do_profile:
            trace_dir = os.path.join(self.checkpoint_dir, "profiler_tb")
            os.makedirs(trace_dir, exist_ok=True)
            profiler_cm = profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
                with_modules=True,
                profile_memory=True,
                schedule=schedule(wait=0, warmup=1, active=profile_steps, repeat=1),
                on_trace_ready=tensorboard_trace_handler(trace_dir),
            )
        else:
            profiler_cm = nullcontext()

        try:
            with profiler_cm as prof:
                for step in range(start_step + 1, total_steps + 1):
                    self.model.train()
                    self.optimizer.zero_grad(set_to_none=True)

                    accum_loss = 0.0
                    accum_metrics = None
                    accum_weight = 0

                    for _ in range(self.grad_accum_steps):
                        try:
                            batch = next(data_iter)
                        except StopIteration:
                            data_iter = iter(train_loader)
                            batch = next(data_iter)

                        batch = self._prepare_batch(batch, device, batch_transform)
                        loss, metrics, weight = self._compute_loss_and_metrics(
                            batch=batch,
                            amp=amp,
                        )

                        self._backward_loss(loss, scaler=scaler, amp=amp)

                        accum_loss += float(loss.detach())
                        accum_metrics = self._accumulate_metrics(accum_metrics, metrics)
                        accum_weight += weight

                    if self.abort_on_nonfinite_loss and not math.isfinite(accum_loss):
                        raise FloatingPointError(
                            f"Non-finite training loss ({accum_loss}) at step "
                            f"{step}; aborting (abort_on_nonfinite_loss=True)."
                        )
                    self._optimizer_step(scaler=scaler, amp=amp)
                    accum_metrics = self._finalize_accum_metrics(accum_metrics)

                    self._log_train_step(
                        step=step,
                        log_every=log_every,
                        accum_loss=accum_loss,
                        accum_metrics=accum_metrics,
                        accum_weight=accum_weight,
                        device=device,
                    )

                    if es_active:
                        loss_window.append(accum_loss / self.grad_accum_steps)
                        if (
                            len(loss_window) == loss_window.maxlen
                            and self.global_step >= early_stop.warmup_steps
                        ):
                            half = loss_window.maxlen // 2
                            old_mean = sum(list(loss_window)[:half]) / max(half, 1)
                            new_mean = sum(list(loss_window)[half:]) / max(
                                loss_window.maxlen - half, 1
                            )
                            denom = max(abs(old_mean), 1e-12)
                            rel_drop = (old_mean - new_mean) / denom
                            if rel_drop < early_stop.min_improvement:
                                print(
                                    f"[early-stop] loss/total plateaued at "
                                    f"step {self.global_step} "
                                    f"(rel_drop={rel_drop:.3e} < "
                                    f"{early_stop.min_improvement:.3e})",
                                    flush=True,
                                )
                                early_stop_triggered = True

                    self._maybe_run_validation(
                        step=step,
                        val_loader=val_loader,
                        validate_every=validate_every,
                        batch_transform=batch_transform,
                        device=device,
                    )

                    self._maybe_save_checkpoint(
                        step=step,
                        checkpoint_every=checkpoint_every,
                        checkpoint_prefix=checkpoint_prefix,
                    )

                    # ADR-0009: a SIGUSR1/SIGTERM (or SIGINT under
                    # DDSSM_PREEMPTIVE=1) sets the flag from the signal
                    # handler; here — between optimizer steps and after
                    # the periodic save — is the safe point to write a
                    # fresh ckpt and raise PreemptError out of fit().
                    if self._preempt_pending:
                        ckpt_path = self._save_periodic_checkpoint(
                            step=step,
                            checkpoint_prefix=checkpoint_prefix,
                        )
                        raise PreemptError(resume_from=str(ckpt_path))

                    if do_profile:
                        prof.step()

                    if early_stop_triggered:
                        break

                self.metrics.step_end("train", self.global_step)

                if do_profile:
                    print(
                        prof.key_averages().table(
                            sort_by="self_cpu_time_total", row_limit=50
                        ),
                        flush=True,
                    )
                    print(
                        f"[profiler] tensorboard traces saved to: {trace_dir}",
                        flush=True,
                    )

        except KeyboardInterrupt:
            # Save an emergency/latest checkpoint on interrupt
            if checkpoint_prefix is None:
                latest_name = "ckpt_interrupted_latest.pth"
            else:
                latest_name = f"ckpt_{checkpoint_prefix}_interrupted_latest.pth"
            try:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                latest_name = os.path.join(self.checkpoint_dir, latest_name)

                self.save_checkpoint(latest_name)
                print(f"[interrupt] Saved emergency checkpoint to {latest_name}")
            except Exception as e:
                print(f"[interrupt] Failed to save emergency checkpoint: {e}")
            raise
        finally:
            if hasattr(self, "metrics"):
                self.metrics.close()

        return int(self.global_step)


# ---------------------------------------------------------------------------
# Hydra-zen config for DDSSMTrainer, co-located with the class.
# ``model`` and ``device`` are runtime-supplied (typically by app.py) and
# left MISSING here. Other fields inherit defaults from the constructor.
# ---------------------------------------------------------------------------

DDSSMTrainerConf = builds(
    DDSSMTrainer,
    populate_full_signature=True,
    model=MISSING,
    device=MISSING,
)
