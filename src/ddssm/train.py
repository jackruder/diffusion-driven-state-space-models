"""DDSSMTrainer: training loop, checkpointing, and config I/O for DDSSM models."""

import os
import math
import yaml
from typing import Any, Callable, final
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import asdict

import torch

from torch import optim
from torch.utils.data import DataLoader
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)

from hydra_zen import builds, instantiate
from omegaconf import MISSING

from .dssd import DDSSM_base
from .loggers import (
    CSVLogger,
    MetricSpec,
    MetricStore,
    ConsoleLogger,
    TensorBoardLogger,
    WandbLogger,
)
from .train_utils import (
    param_groups_for_adamw,
)


def _namespace_to_dict(obj: Any) -> Any:
    """Recursively convert SimpleNamespace / objects to plain dicts for YAML serialisation."""
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _namespace_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_namespace_to_dict(v) for v in obj)
    return obj


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
    EMA tracking, REWO / scheduled λ weighting, step-level metric logging to
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

                wandb_config={
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
        optimizer: optim.Optimizer | None = None,
        csv_log_path: str | None = None,
        tensorboard_dir: str = "runs/ddssm",
        wandb_config: dict | None = None,
        quiet: bool = False,
    ):
        self.model = model.to(device)
        self.device = device

        self.global_step = 0

        # Ensure config is attached
        if not hasattr(self.model, "config"):
            raise AttributeError("Model must have a `.config` attribute.")
        self.config = self.model.config

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                param_groups_for_adamw(
                    self.model,
                    enc_lr=self.config.hyperparams.enc_lr,
                    dec_lr=self.config.hyperparams.dec_lr,
                    trans_lr=self.config.hyperparams.trans_lr,
                    zinit_lr=self.config.hyperparams.zinit_lr,
                    weight_decay=self.config.hyperparams.weight_decay,
                ),
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        self.scheduler = None
        self.weight_decay = self.config.hyperparams.weight_decay

        self.grad_accum_steps = self.config.hyperparams.grad_accum_steps

        self.ema_decay = self.config.hyperparams.ema_decay
        # EMA on the denoiser (used at sampling time)
        self.ema = EMA(self.model.transition, decay=self.ema_decay)

        loggers = [
            TensorBoardLogger(log_dir=tensorboard_dir),
        ]  # epoch-only by default
        if csv_log_path:
            loggers.append(CSVLogger(path=csv_log_path))
        if wandb_config is not None:
            loggers.append(WandbLogger(**wandb_config))
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
            ],
            loggers=loggers,
        )

        self.checkpoint_dir = model.config.checkpoint_dir

        # initialize REWO state
        if self.config.hyperparams.lambda_schedule == "rewo":
            self.rewo_lambda = 1.0
            self.rewo_D_bar = None
            self.rewo_initial_phase = True
            self.rewo = True
        else:
            self.rewo = False

    def get_batch_size(self) -> int:
        return self.model.config.hyperparams.batch_size

    def _set_trainable(self, t):
        """t: StageTrainable"""

        def maybe_flag(mod, flag: bool):
            if mod is None:
                return
            for p in mod.parameters():
                p.requires_grad = flag

        # Expect these attributes to exist (guard with hasattr)
        maybe_flag(getattr(self.model, "encoder", None), t.encoder)
        maybe_flag(getattr(self.model, "decoder", None), t.decoder)
        maybe_flag(getattr(self.model, "zinit", None), t.z_init)
        maybe_flag(getattr(self.model, "transition", None), t.transition)
        maybe_flag(getattr(self.model, "static_embeddings", None), t.encoder)

    def _rebuild_optimizer(
        self,
        lrs,
    ):
        """lrs: StageLrs"""
        groups = param_groups_for_adamw(
            self.model,
            enc_lr=lrs.enc_lr,
            dec_lr=lrs.dec_lr,
            trans_lr=lrs.trans_lr,
            zinit_lr=lrs.zinit_lr,
            weight_decay=self.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    # ------------------------
    # Serialization / Checkpoint
    # ------------------------
    def save_config(self, path: str):
        """Dump current config to YAML (supports Pydantic, dataclasses, or SimpleNamespace)."""
        cfg = self.model.config
        if hasattr(cfg, "model_dump"):
            cfg_dict = cfg.model_dump()
        elif hasattr(cfg, "dict"):
            cfg_dict = cfg.dict()
        elif hasattr(cfg, "__dict__"):
            cfg_dict = {k: _namespace_to_dict(v) for k, v in vars(cfg).items()}
        else:
            try:
                cfg_dict = asdict(cfg)
            except Exception:
                cfg_dict = {}
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(cfg_dict, f)

    @classmethod
    def load_from_yaml(
        cls,
        yaml_path: str,
        device: torch.device,
        optimizer: optim.Optimizer | None = None,
        **kwargs,
    ) -> "DDSSMTrainer":
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(yaml_path)
        model = instantiate(cfg).to(device)

        return cls(model, device, optimizer=optimizer, **kwargs)

    @staticmethod
    def _atomic_save(obj, path: str) -> None:
        """Write to a temp file in the same dir, then atomically replace."""
        path = str(path)
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)

        f = tempfile.NamedTemporaryFile(
            prefix="tmp_save_", suffix=".pth", dir=d, delete=False
        )
        tmppath = f.name
        f.close()
        try:
            torch.save(obj, tmppath)
            os.replace(tmppath, path)
        except Exception:
            try:
                os.remove(tmppath)
            except OSError:
                pass
            raise

    def save_checkpoint(
        self,
        path: str,
    ):
        cfg = self.model.config
        if hasattr(cfg, "model_dump"):
            cfg_dict = cfg.model_dump()
        elif hasattr(cfg, "dict"):
            cfg_dict = cfg.dict()
        elif hasattr(cfg, "__dict__"):
            cfg_dict = {k: _namespace_to_dict(v) for k, v in vars(cfg).items()}
        else:
            try:
                cfg_dict = asdict(cfg)
            except Exception:
                cfg_dict = {}

        payload = {
            "_format": "ddssm_ckpt_v1",
            "config": cfg_dict,
            "model_state": self.model.state_dict(),
            "optimizer_state": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "ema_decay": self.ema_decay,
            "ema_state": getattr(self.ema, "shadow", None),
            "global_step": int(self.global_step),
            "grad_accum_steps": int(self.grad_accum_steps),
        }
        if self.rewo:
            payload["rewo_state"] = {
                "lambda": self.rewo_lambda,
                "D_bar": self.rewo_D_bar,
                "initial_phase": self.rewo_initial_phase,
            }
        self._atomic_save(payload, path)

    def restore_from_checkpoint(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"], strict=strict)

        if ckpt.get("optimizer_state") is not None and self.optimizer is not None:
            # works if param_groups match param_groups_for_adamw ordering
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print(
                "[restore] Warning: optimizer state not found in checkpoint or optimizer is None."
            )

        if ckpt.get("ema_state") is not None and hasattr(self, "ema"):
            # restore EMA shadow and (optionally) decay
            self.ema.shadow = ckpt["ema_state"]
            self.ema_decay = ckpt.get("ema_decay", self.ema_decay)

        self.global_step = int(ckpt.get("global_step", 0))

        # Restore REWO state
        if self.rewo:
            rewo = ckpt.get("rewo_state", {})
            self.rewo_lambda = rewo.get("lambda", 1.0)
            self.rewo_D_bar = rewo.get("D_bar", None)
            self.rewo_initial_phase = rewo.get("initial_phase", True)

        return {
            "grad_accum_steps": ckpt.get("grad_accum_steps", self.grad_accum_steps),
        }

    def _build_lambda_schedule(self):
        def linear_sched(start, end, warmup_steps, step):
            if step >= warmup_steps:
                return end
            pct = step / float(warmup_steps)
            return start + (end - start) * pct

        def cosine_sched(start, end, warmup_steps, step):
            if step >= warmup_steps:
                return end
            pct = step / float(warmup_steps)
            cosine_factor = 0.5 * (1.0 - math.cos(math.pi * pct))
            return start + (end - start) * cosine_factor

        if self.config.hyperparams.lambda_schedule == "linear":
            return lambda step: linear_sched(
                start=self.config.hyperparams.lambda_start,
                end=self.config.hyperparams.lambda_end,
                warmup_steps=self.config.hyperparams.lambda_warmup_steps,
                step=step,
            )
        if self.config.hyperparams.lambda_schedule == "cosine":
            return lambda step: cosine_sched(
                start=self.config.hyperparams.lambda_start,
                end=self.config.hyperparams.lambda_end,
                warmup_steps=self.config.hyperparams.lambda_warmup_steps,
                step=step,
            )
        if self.config.hyperparams.lambda_schedule == "rewo":
            return None
        return lambda step: 1.0

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
        lambda_schedule,
        compute_recon: bool,
        compute_trans: bool,
        rewo,
    ):
        observed = batch["observed_data"]
        observed_mask = batch["observation_mask"]
        timepoints = batch["timepoints"]
        covariates = batch.get("covariates", None)
        static_covariates = batch.get("static_covariates", None)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            _elbo, distortion, rate, metrics, _stats = self.model(
                observed,
                observed_mask,
                timepoints,
                covariates=covariates,
                static_covariates=static_covariates,
                train=True,
                compute_recon=compute_recon,
                compute_trans=compute_trans,
                report_scaled=False,
            )

        if not self.rewo:
            assert lambda_schedule is not None
            sched_step = self.global_step + 1
            current_lambda = lambda_schedule(sched_step)
            loss = distortion + current_lambda * rate
            metrics["optim/lambda"] = torch.tensor(current_lambda)
        else:
            assert rewo is not None
            d_val = distortion.detach()
            if self.rewo_D_bar is None:
                self.rewo_D_bar = d_val
            else:
                self.rewo_D_bar = (
                    1 - rewo.alpha
                ) * d_val + rewo.alpha * self.rewo_D_bar

            delta = self.rewo_D_bar - rewo.D0
            H_delta = 1.0 if delta > 0 else 0.0
            lam_safe = max(self.rewo_lambda, 1e-6)
            term1 = (1 - H_delta) * math.tanh(rewo.tau1 * (1.0 / lam_safe - 1.0))
            term2 = rewo.tau2 * H_delta
            f_lambda = term1 - term2

            self.rewo_lambda = self.rewo_lambda * math.exp(
                -rewo.nu * f_lambda * float(delta)
            )

            if self.rewo_D_bar <= rewo.D0:
                self.rewo_initial_phase = False

            if self.rewo_initial_phase:
                loss = distortion
            else:
                loss = rate + self.rewo_lambda * distortion

            metrics["rewo/lambda"] = torch.tensor(self.rewo_lambda)
            metrics["rewo/D_bar"] = self.rewo_D_bar
            metrics["rewo/initial_phase"] = torch.tensor(
                1.0 if self.rewo_initial_phase else 0.0
            )

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
        self.global_step += 1
        log_values = {
            "loss/total": torch.tensor(
                accum_loss / self.grad_accum_steps, device=device
            ),
            **accum_metrics,
        }
        self.metrics.update(split="train", values=log_values, weight=accum_weight)
        if log_every and (step % log_every == 0):
            self.metrics.step_end("train", self.global_step)

    def _run_validation(
        self,
        val_loader: DataLoader,
        batch_transform: Callable[[dict, torch.device], dict] | None,
        device: torch.device,
        compute_recon: bool,
        compute_trans: bool,
    ):
        self.model.eval()
        with torch.no_grad():
            for vbatch in val_loader:
                vbatch = self._prepare_batch(vbatch, device, batch_transform)

                vwin = vbatch["observed_data"]
                vmask = vbatch["observation_mask"]
                vtime = vbatch["timepoints"]
                vcov = vbatch.get("covariates", None)
                vstatic_cov = vbatch.get("static_covariates", None)

                vloss, vmetrics, _ = self.model(
                    vwin,
                    vmask,
                    vtime,
                    covariates=vcov,
                    static_covariates=vstatic_cov,
                    train=False,
                    compute_recon=compute_recon,
                    compute_trans=compute_trans,
                )
                self.metrics.update(
                    "val",
                    values={"loss/total": vloss, **vmetrics},
                    weight=vwin.size(0),
                )

    def _maybe_run_validation(
        self,
        step: int,
        val_loader: DataLoader | None,
        validate_every: int,
        batch_transform: Callable[[dict, torch.device], dict] | None,
        device: torch.device,
        compute_recon: bool,
        compute_trans: bool,
    ):
        if val_loader is None or not validate_every or (step % validate_every != 0):
            return
        self._run_validation(
            val_loader=val_loader,
            batch_transform=batch_transform,
            device=device,
            compute_recon=compute_recon,
            compute_trans=compute_trans,
        )
        self.metrics.epoch_end("val", self.global_step)

    def _save_periodic_checkpoint(self, step: int, checkpoint_prefix: str | None):
        if checkpoint_prefix is None:
            ckpt_name = f"ckpt_step{step}.pth"
            latest_name = "ckpt_latest.pth"
        else:
            ckpt_name = f"ckpt_{checkpoint_prefix}_step{step}.pth"
            latest_name = f"ckpt_{checkpoint_prefix}_latest.pth"

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_name = os.path.join(self.checkpoint_dir, ckpt_name)
        latest_name = os.path.join(self.checkpoint_dir, latest_name)
        self.save_checkpoint(ckpt_name)
        self.save_checkpoint(latest_name)

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

    def _safe_resume(self, resume_from: str | None):
        if resume_from is None:
            return
        try:
            self.restore_from_checkpoint(resume_from, strict=True)
            print(f"[resume] global_step={self.global_step}")
        except Exception as e:
            print(f"[resume] Failed to restore from {resume_from}: {e}")

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
        compute_recon: bool = True,
        compute_trans: bool = True,
        profile_steps: int = 0,
    ):
        """One optimizer step == one 'step'.
        Validation / checkpoints / logs are triggered by step counts.
        - Resumes from `resume_from` if provided (restores global_step, optimizer, EMA).
        - Uses grad accumulation and optional AMP for memory efficiency.
        - Profiles up to `profile_steps` optimizer steps if > 0.
        """

        device = self.device
        self.model.to(device)

        # --- REWO Initialization ---
        rewo = self.model.config.hyperparams.rewo if self.rewo else None

        lambda_schedule = self._build_lambda_schedule()

        # Make the console logger print every `log_every` steps
        for lg in self.metrics.loggers:
            if isinstance(lg, ConsoleLogger):
                lg.every_steps = log_every

        # see if we should resume
        self._safe_resume(resume_from)

        data_iter = iter(train_loader)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
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
                            lambda_schedule=lambda_schedule,
                            compute_recon=compute_recon,
                            compute_trans=compute_trans,
                            rewo=rewo,
                        )

                        self._backward_loss(loss, scaler=scaler, amp=amp)

                        accum_loss += float(loss.detach())
                        accum_metrics = self._accumulate_metrics(accum_metrics, metrics)
                        accum_weight += weight

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

                    self._maybe_run_validation(
                        step=step,
                        val_loader=val_loader,
                        validate_every=validate_every,
                        batch_transform=batch_transform,
                        device=device,
                        compute_recon=compute_recon,
                        compute_trans=compute_trans,
                    )

                    self._maybe_save_checkpoint(
                        step=step,
                        checkpoint_every=checkpoint_every,
                        checkpoint_prefix=checkpoint_prefix,
                    )

                    if do_profile:
                        prof.step()

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
