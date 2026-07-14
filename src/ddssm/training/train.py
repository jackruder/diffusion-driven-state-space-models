"""DDSSMTrainer: training loop, EMA, logging, and the fit lifecycle.

The ``.pth`` payload schema lives in :mod:`ddssm.training.checkpoint`; the
trainer's ``save_checkpoint`` / ``restore_from_checkpoint`` delegate
there.
"""

import os
import math
import random
import shutil
import signal
from typing import TYPE_CHECKING, Any, final
import logging
import datetime
import tempfile
from contextlib import nullcontext, contextmanager
from collections import deque
from collections.abc import Callable

import numpy as np
import torch

from ddssm.model.losses import FullELBO, SplitLoss

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ddssm.training.stages import EarlyStopSpec

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

from ddssm.model.dssd import DDSSM_base
from ddssm.training.loggers import (
    CSVLogger,
    MetricSpec,
    MetricStore,
    WandbLogger,
    ConsoleLogger,
    TensorBoardLogger,
)
from ddssm.training.checkpoint import NoUsableCheckpointError
from ddssm.training.train_utils import (
    param_groups_psi,
    param_groups_phith,
    param_groups_for_adamw,
    split_params_phith_psi,
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
        self._float_keys: tuple[str, ...] = tuple(
            k for k, v in self.shadow.items() if torch.is_floating_point(v)
        )
        self._int_keys: tuple[str, ...] = tuple(
            k for k in self.shadow if k not in set(self._float_keys)
        )
        self._float_shadow: list[torch.Tensor] = [
            self.shadow[k] for k in self._float_keys
        ]
        # Cache the live-tensor lists once so per-step ``update()`` does not
        # call ``state_dict()`` — the returned dict is dynamo-opaque and
        # would graph-break inside a compiled train step.
        msd = self._module.state_dict()
        self._float_live: list[torch.Tensor] = [msd[k] for k in self._float_keys]
        self._int_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
            (self.shadow[k], msd[k]) for k in self._int_keys
        ]

    def _recache_live(self) -> None:
        """Refresh the live-tensor cache — call after any state_dict swap."""
        msd = self._module.state_dict()
        self._float_live = [msd[k] for k in self._float_keys]
        self._int_pairs = [(self.shadow[k], msd[k]) for k in self._int_keys]

    @torch.no_grad()
    def update(self):
        torch._foreach_mul_(self._float_shadow, self.decay)
        torch._foreach_add_(
            self._float_shadow, self._float_live, alpha=1.0 - self.decay
        )
        for shadow_t, live_t in self._int_pairs:
            shadow_t.copy_(live_t)

    @contextmanager
    def swap(self):
        msd = self._module.state_dict()
        backup = {k: v.detach().clone() for k, v in msd.items()}
        self._module.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            self._module.load_state_dict(backup, strict=True)
            # Defensive: refresh live-tensor cache in case load_state_dict
            # replaces param/buffer tensors.
            self._recache_live()


_COMPILE_OFF_VALUES = {"0", "false", "no", "off"}


def _compiled_step_enabled() -> bool:
    """Whether the compiled two-region train step is opt-in (``DDSSM_COMPILE_STEP=1``).

    Applies only when the trainer is in single-loss mode with
    ``grad_accum_steps=1``; falls back to the per-microstep eager path
    otherwise.
    """
    v = os.environ.get("DDSSM_COMPILE_STEP", "0").strip().lower()
    return v not in _COMPILE_OFF_VALUES


def _make_split_step(
    model,
    active_loss,
    optimizer,
    params_flat,
    max_norm: float,
    ema: "EMA | None" = None,
) -> "tuple[callable, callable]":
    """Build the two-region train-step callables per torch's docs pattern.

    Returns ``(fwd_bwd_fn, opt_ema_fn)`` — two callables the caller
    compiles independently via ``torch.compile``. Matches the pattern
    from PyTorch's Compiled Autograd tutorial + "Compiling the optimizer"
    recipe: forward+backward as one region (compiled_autograd hook makes
    backward a fullgraph), optimizer.step + EMA as a separate region
    (never fullgraph — ``_use_grad`` graph-breaks otherwise).

    * ``fwd_bwd_fn(batch, lam)``: bf16-autocast raw forward core +
      single-loss ``recon + λ·(init_kl_phith + trans_kl_phith)`` +
      ``loss.backward()``. Returns a flat tuple of detached metric
      tensors — ``(loss, recon, init_kl_phith, init_kl_psi,
      trans_kl_phith, trans_kl_psi, init_vhp, init_entropy,
      init_kl_aux, init_loss_init, recon_calib, *trans_extras)`` —
      so AOT autograd allocates a tangent slot only for ``loss``.
    * ``opt_ema_fn()``: clip + ``optimizer.step()`` +
      ``zero_grad(set_to_none=False)`` + in-graph EMA. Returns the
      pre-clip ``grad_norm``.

    Caller constraints: single-optimizer, ``grad_accum_steps == 1``,
    ``lam`` passed as a 0-d device tensor filled per step, ``fused=True``
    AdamW.
    """
    ema_decay = float(ema.decay) if ema is not None else 0.0
    ema_float_shadow = ema._float_shadow if ema is not None else None
    ema_float_live = ema._float_live if ema is not None else None
    ema_int_pairs = ema._int_pairs if ema is not None else None

    del active_loss  # signature stability; loss composition is inlined below

    def _fwd_bwd(batch, lam):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            core = model._forward_core_raw(
                batch["observed_data"],
                batch["observation_mask"],
                batch["timepoints"],
                covariates=batch.get("covariates"),
                static_covariates=batch.get("static_covariates"),
                train=True,
            )
        # Inline FullELBO single-loss composition so ``lam`` stays a tensor.
        loss = core.recon + lam * (core.init_kl_phith + core.trans_kl_phith)
        loss.backward()
        return (
            loss.detach(),
            core.recon.detach(),
            core.init_kl_phith.detach(),
            core.init_kl_psi.detach(),
            core.trans_kl_phith.detach(),
            core.trans_kl_psi.detach(),
            core.init_vhp.detach(),
            core.init_entropy.detach(),
            core.init_kl_aux.detach(),
            core.init_loss_init.detach(),
            core.recon_calib.detach(),
            *(t.detach() for t in core.trans_extras),
        )

    def _opt_ema():
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params_flat, max_norm, foreach=True,
        )
        optimizer.step()
        # set_to_none=False keeps grad tensor storage stable across steps.
        optimizer.zero_grad(set_to_none=False)
        if ema_float_shadow is not None:
            torch._foreach_mul_(ema_float_shadow, ema_decay)
            torch._foreach_add_(
                ema_float_shadow, ema_float_live, alpha=1.0 - ema_decay
            )
            for shadow_t, live_t in ema_int_pairs:
                shadow_t.copy_(live_t)
        return grad_norm

    return _fwd_bwd, _opt_ema


def _compile_optimizer_step(optimizers) -> None:
    """Wrap each optimizer's ``.step()`` with ``torch.compile``.

    Idempotent via the ``_ddssm_compiled`` sentinel; opt out with
    ``DDSSM_TORCH_COMPILE_OPTIMIZER=0``.
    """
    if os.environ.get(
        "DDSSM_TORCH_COMPILE_OPTIMIZER", "1"
    ).strip().lower() in _COMPILE_OFF_VALUES:
        return
    for opt in optimizers:
        if getattr(opt.step, "_ddssm_compiled", False):
            continue
        opt.step = torch.compile(opt.step, fullgraph=False)
        opt.step._ddssm_compiled = True


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
            from ddssm.model.dssd import _default_hyperparams

            hparams = _default_hyperparams()
        self.hparams = hparams

        # ADR-0005: resolved YAML of the model's hydra-zen builds()
        # config. Persisted into checkpoints so the load path can warn
        # on silent semantic drift (shapes preserved, builder semantics
        # changed). ``None`` for ad-hoc constructions (tests, notebooks)
        # that have no Hydra config to serialise.
        self._model_config_yaml: str | None = model_config_yaml

        self.global_step = 0
        # Global step at which the current fit phase began. 0 for a fresh
        # run; restored from the checkpoint on resume so ``step_within_stage``
        # (which drives the loss λ schedule) counts from the phase origin.
        self._stage_start_step: int = 0
        # Opt-in fail-fast: when True, a non-finite optimized loss raises
        # instead of silently NaN-poisoning the weights. Off by default so
        # existing runs are unchanged; the logger always counts it regardless.
        self.abort_on_nonfinite_loss: bool = False
        # ADR-0004: active loss object (installed by orchestrator per
        # stage; constructed lazily at fit() start otherwise).
        from ddssm.model.losses import Loss

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
                    psi_betas=self._hparams_psi_betas(),
                    weight_decay_psi=getattr(self.hparams, "weight_decay_psi", None),
                    claim_psi=getattr(self.hparams, "lr_schedule", None) is not None,
                ),
                betas=(0.9, 0.999),
                eps=1e-8,
                fused=True,
            )
        # Optimizer topology: single-mode is ``[self.optimizer]``;
        # ``_install_split_topology`` (split-loss mode) replaces this with
        # ``[opt_phith, opt_psi]`` and keeps ``self.optimizer`` aliased to
        # the φθ side. Every step/zero_grad/unscale site loops this list.
        self._optimizers: list[optim.Optimizer] = [self.optimizer]
        self.opt_psi: optim.Optimizer | None = None
        # Cache of per-optimizer flat param lists for clip_grad_norm_ /
        # zero_grad. Invalidated (set to None) any time ``self._optimizers``
        # is reassigned; lazily rebuilt on the next _optimizer_step call.
        self._opt_param_cache: list[list[torch.nn.Parameter]] | None = None
        # Cached (φθ, ψ) parameter lists for the split backward; populated
        # by ``_install_split_topology`` with ``include_frozen=True`` (the
        # full per-side partition). ``_backward_loss`` re-filters them by
        # the live ``requires_grad`` at every call, so per-stage
        # freeze/unfreeze after install is honored.
        self._phith_params: list[torch.nn.Parameter] | None = None
        self._psi_params: list[torch.nn.Parameter] | None = None
        # One-time warning latches for the silent scheduler traps:
        # (a) legacy ``self.scheduler`` fallback stepping only the φθ side
        # under the split topology; (b) rebuilding optimizers while
        # installed schedulers still point at the old optimizer objects.
        self._warned_split_legacy_scheduler: bool = False
        self._warned_stale_schedulers: bool = False

        self.scheduler = None
        # Scheduler topology mirror of ``_optimizers``: populated by
        # ``_install_scheduler`` (one entry per optimizer). When empty but
        # ``self.scheduler`` was assigned directly (legacy call sites),
        # ``_optimizer_step`` falls back to stepping ``self.scheduler``.
        self._schedulers: list = []
        # Hoisted out of fit() so save_checkpoint can persist its state via
        # ``Checkpoint.from_trainer``. bf16 autocast doesn't need scaling, so
        # the default scaler stays disabled and ``scaler.state_dict()`` is
        # NOT written to disk (see ``Checkpoint.from_trainer``); enable it
        # externally if you wire in fp16 AMP.
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.weight_decay = self.hparams.weight_decay
        # ψ-side (score net) override; ``None`` falls back to
        # ``weight_decay``. getattr guard: hparams objects predating the
        # field must keep working unchanged.
        self.weight_decay_psi: float | None = getattr(
            self.hparams, "weight_decay_psi", None
        )

        self.grad_accum_steps = self.hparams.grad_accum_steps
        self.clip_grad_norm = self.hparams.clip_grad_norm
        # Grad-skip bookkeeping (alongside per-optimizer grad-norm
        # clipping): non-finite grad norms discard the macro-batch instead
        # of poisoning Adam state / EMA shadows. ``grad_skip_count`` is
        # cumulative and persisted via checkpoints; ``_last_grad_norm`` is
        # the combined (L2-over-per-optimizer) pre-clip norm of the most
        # recent step (NaN on a skipped step); ``_last_grad_norms_by_opt``
        # holds the individual per-optimizer norms when the split topology
        # is active (``None`` in single-optimizer mode).
        self.grad_skip_count: int = 0
        self._last_grad_norm: float | None = None
        self._last_grad_norms_by_opt: list[float] | None = None

        self.ema_decay = self.hparams.ema_decay
        # EMA on the full model (encoder + decoder + transition). Used at
        # validation (swap in ``_run_validation``) and sampling time. Full-
        # model scope keeps the val snapshot self-consistent: swapping only
        # the transition gave val a Frankenstein model.
        self.ema = EMA(self.model, decay=self.ema_decay)

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
                / ``transition`` boolean flags.
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

    def _hparams_psi_betas(self) -> tuple[float, float] | None:
        """Single-mode ψ betas from hparams, as a tuple (or ``None``).

        ``getattr`` guard: hparams objects predating the ``psi_betas``
        field must keep working unchanged.
        """
        pb = getattr(self.hparams, "psi_betas", None)
        return tuple(pb) if pb else None

    def _psi_weight_decay(self) -> float:
        """Resolved ψ-optimizer weight decay (override or φθ fallback)."""
        if self.weight_decay_psi is None:
            return self.weight_decay
        return self.weight_decay_psi

    def _warn_if_ema_decay_too_high(self, total_steps: int) -> None:
        """Warn if the EMA time constant covers >5% of the training budget.

        The EMA has effective window (time constant) ``τ = 1 / (1 − decay)``
        steps: after τ steps the contribution of the init snapshot has
        decayed to ``1/e``. When ``τ / total_steps > 5%`` the shadow is
        still dominated by initialization at the end of training and the
        EMA weights are worse than the live weights — e.g. ``decay=0.9999``
        (τ = 10_000) with ``total_steps=500`` gives ``τ/budget = 20×``.

        Cutoff (5%) is the "less than 20 time constants of training" rule
        of thumb — well short of the ~40+ often used in diffusion practice
        but tight enough to catch the smoke-preset footgun.
        """
        if total_steps <= 0 or self.ema_decay >= 1.0:
            return
        tau = 1.0 / (1.0 - self.ema_decay)
        ratio = tau / total_steps
        if ratio <= 0.05:
            return
        max_decay = max(0.0, 1.0 - 20.0 / total_steps)
        log.warning(
            "[ema] ema_decay=%.6g gives an effective window of %.0f steps "
            "(%.1f%% of the %d-step budget); the shadow will still be "
            "dominated by initialization at fit end. For this budget, "
            "prefer ema_decay <= %.4g (window <= 5%% of budget).",
            self.ema_decay,
            tau,
            100.0 * ratio,
            total_steps,
            max_decay,
        )

    def _build_split_optimizers(
        self,
        enc_lr: float,
        dec_lr: float,
        trans_lr: float,
        baseline_lr: float | None = None,
    ) -> None:
        """(Re)build the two-optimizer (φθ, ψ) topology and its aliases.

        The ψ side (score net: ``transition.diffmodel`` +
        ``transition.embed_layer``) gets betas ``(0.9, 0.99)`` — faster
        second-moment adaptation for the score-matching objective — while
        the φθ side keeps the default ``(0.9, 0.999)``. Weight decay is
        per-optimizer: φθ uses ``hparams.weight_decay``, ψ uses
        ``hparams.weight_decay_psi`` when set (falling back to
        ``weight_decay``).

        Raises:
            ValueError: If the model has no ψ parameters (split-loss mode
                requires a diffusion transition).
        """
        psi_groups = param_groups_psi(
            self.model,
            trans_lr=trans_lr,
            weight_decay=self._psi_weight_decay(),
        )
        if not psi_groups:
            raise ValueError(
                "split-loss mode requires a diffusion transition: the model "
                "has no ψ parameters (transition.diffmodel / "
                "transition.embed_layer), so param_groups_psi is empty."
            )
        opt_phith = torch.optim.AdamW(
            param_groups_phith(
                self.model,
                enc_lr=enc_lr,
                dec_lr=dec_lr,
                trans_lr=trans_lr,
                weight_decay=self.weight_decay,
                baseline_lr=baseline_lr,
            ),
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )
        opt_psi = torch.optim.AdamW(
            psi_groups, betas=(0.9, 0.99), eps=1e-8, fused=True
        )
        self._optimizers = [opt_phith, opt_psi]
        self._opt_param_cache = None  # invalidated on topology change
        # ``self.optimizer`` stays the φθ alias so existing single-optimizer
        # call sites (checkpointing, external LR pokes) keep a target.
        self.optimizer = opt_phith
        self.opt_psi = opt_psi

    def _install_split_topology(self) -> None:
        """Switch the trainer to the split-loss two-optimizer topology.

        Builds ``opt_phith`` / ``opt_psi`` from the trainer hparams' LRs
        (mirroring the default single-optimizer build) and caches the
        (φθ, ψ) parameter lists used by the split backward. Calling it
        again simply rebuilds (fresh Adam state).
        """
        # The split backward runs two passes over a shared graph (the φθ
        # pass uses retain_graph=True). AOTAutograd's donated-buffer
        # optimization — active inside torch.compile'd regions, which
        # ``maybe_compile`` enables by default — hard-errors on
        # retain_graph=True, so it must be off for split-loss runs.
        try:
            import torch._functorch.config as _functorch_config

            _functorch_config.donated_buffer = False
        except (ImportError, AttributeError):  # pragma: no cover
            pass
        self._build_split_optimizers(
            enc_lr=self.hparams.enc_lr,
            dec_lr=self.hparams.dec_lr,
            trans_lr=self.hparams.trans_lr,
            baseline_lr=getattr(self.hparams, "baseline_lr", None),
        )
        # Cache the UNFILTERED per-side partition (frozen params included):
        # a trainable mask may flip ``requires_grad`` after install, so the
        # backward filters by the live flag instead of a stale snapshot.
        self._phith_params, self._psi_params = split_params_phith_psi(
            self.model, include_frozen=True
        )

    def _install_scheduler(self, sched: torch.optim.lr_scheduler.LambdaLR) -> None:
        """Install the LR scheduler(s) for the current optimizer topology.

        Single mode: ``self.scheduler = sched`` and ``_schedulers = [sched]``
        (today's behavior). Split mode: additionally builds a ψ-side
        ``LambdaLR`` on ``opt_psi`` reusing the φθ scheduler's shape (the
        same ``lr_lambda`` callable — identical warmup/total/floor), so
        both sides decay together; ``_optimizer_step`` steps the list.

        Args:
            sched: A ``LambdaLR`` (e.g. from ``make_warmup_cosine``)
                attached to ``self._optimizers[0]``.
        """
        self.scheduler = sched
        if len(self._optimizers) < 2:
            self._schedulers = [sched]
            return
        opt_psi = self._optimizers[1]
        sched_psi = torch.optim.lr_scheduler.LambdaLR(
            opt_psi,
            lr_lambda=[sched.lr_lambdas[0]] * len(opt_psi.param_groups),
        )
        self._schedulers = [sched, sched_psi]

    def _install_lr_schedule(
        self,
        group_conf,
        total_steps: int,
        lambda_ramp,
    ) -> None:
        """Install per-role LR schedules from a :class:`LrScheduleGroupConf`.

        ``None`` group_conf is a no-op. Otherwise the resolver fills any
        None fields from ``lambda_ramp`` + ``total_steps`` (see
        :func:`resolve_lr_schedule_defaults`) and per-role
        :func:`make_lr_lambda` callables are attached to the current
        optimizer topology:

        - Single-optimizer mode: one ``LambdaLR`` whose per-group
          ``lr_lambda`` list dispatches on each param group's ``role`` tag
          (``role="phith"`` default when absent). Delegates to
          :meth:`_install_scheduler` so the legacy install path is reused.
        - Split-loss mode: two independent ``LambdaLR`` — φθ side on
          ``_optimizers[0]`` with the φθ schedule, ψ side on ``opt_psi``
          with the ψ schedule (NOT a broadcast of φθ's). ``self.scheduler``
          is aliased to the φθ one for the legacy accessor.
        """
        if group_conf is None:
            return
        from ddssm.training.stages import (
            make_lr_lambda,
            resolve_lr_schedule_defaults,
        )

        resolved = resolve_lr_schedule_defaults(
            group_conf, lambda_ramp, total_steps
        )
        phith_fn = make_lr_lambda(resolved.phith)
        psi_fn = make_lr_lambda(resolved.psi)

        def _per_group_lambda(opt):
            return [
                (phith_fn if g.get("role", "phith") == "phith" else psi_fn)
                for g in opt.param_groups
            ]

        if len(self._optimizers) < 2:
            opt = self._optimizers[0]
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=_per_group_lambda(opt)
            )
            self._install_scheduler(sched)
            return

        # Split mode: build both schedulers directly. Do NOT route through
        # ``_install_scheduler`` — it would broadcast φθ's lambda across
        # the ψ optimizer, which is exactly the asymmetry we exist to fix.
        opt_phith = self._optimizers[0]
        opt_psi = self._optimizers[1]
        sched_phith = torch.optim.lr_scheduler.LambdaLR(
            opt_phith, lr_lambda=_per_group_lambda(opt_phith)
        )
        sched_psi = torch.optim.lr_scheduler.LambdaLR(
            opt_psi, lr_lambda=_per_group_lambda(opt_psi)
        )
        self.scheduler = sched_phith
        self._schedulers = [sched_phith, sched_psi]

    def _rebuild_optimizer(
        self,
        lrs,
    ):
        """Rebuild the optimizer(s) with per-component stage learning rates.

        Split-aware: under the two-optimizer topology both sides are
        rebuilt with their structural betas; otherwise the single AdamW is
        rebuilt as before (threading the optional single-mode per-group
        ψ betas from hparams).

        Args:
            lrs: A ``StageLrs``-like object with ``enc_lr`` / ``dec_lr`` /
                ``trans_lr`` learning rates.
        """
        if self._schedulers and not self._warned_stale_schedulers:
            self._warned_stale_schedulers = True
            log.warning(
                "[rebuild] optimizer(s) rebuilt while %d LR scheduler(s) are "
                "installed; they still point at the OLD optimizer objects and "
                "will no longer scale the live LRs. Reinstall via "
                "_install_scheduler after rebuilding.",
                len(self._schedulers),
            )
        if len(self._optimizers) == 2:
            self._build_split_optimizers(
                enc_lr=lrs.enc_lr,
                dec_lr=lrs.dec_lr,
                trans_lr=lrs.trans_lr,
                baseline_lr=getattr(lrs, "baseline_lr", None),
            )
            # Defensive refresh of the split param caches (cheap): the
            # partition is structural, but a rebuild is the natural point
            # to re-derive it should the module set ever change.
            self._phith_params, self._psi_params = split_params_phith_psi(
                self.model, include_frozen=True
            )
            return
        groups = param_groups_for_adamw(
            self.model,
            enc_lr=lrs.enc_lr,
            dec_lr=lrs.dec_lr,
            trans_lr=lrs.trans_lr,
            weight_decay=self.weight_decay,
            psi_betas=self._hparams_psi_betas(),
            weight_decay_psi=self.weight_decay_psi,
            claim_psi=getattr(self.hparams, "lr_schedule", None) is not None,
        )
        self.optimizer = torch.optim.AdamW(
            groups, betas=(0.9, 0.999), eps=1e-8, fused=True
        )
        self._optimizers = [self.optimizer]
        self._opt_param_cache = None  # invalidated on topology change

    # ------------------------
    # Serialization / Checkpoint  (schema owned by ddssm.training.checkpoint)
    # ------------------------
    def save_checkpoint(self, path: str) -> None:
        """Persist trainer state via :mod:`ddssm.training.checkpoint`."""
        from ddssm.training.checkpoint import save as _save

        _save(self, path)

    def restore_from_checkpoint(self, path: str, strict: bool = True) -> None:
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
        from ddssm.training.checkpoint import load_into_model

        ckpt = load_into_model(
            self.model,
            path,
            device=self.device,
            strict=strict,
            load_ema=False,
        )
        # Split-mode contract guard — FIRST, before any optimizer-state
        # handling. The live trainer's intent is DECLARED when either the
        # split topology is already installed or an active loss object has
        # been set (its ``use_split_loss`` flag decides). A declared mode
        # that contradicts the checkpoint would silently drop / fabricate
        # ψ optimizer state, so hard-error. When NEITHER is present —
        # e.g. the orchestrator's preempt-resume restores BEFORE any stage
        # loss is installed — the trainer has expressed no intent yet, so
        # adopt the checkpoint's mode silently (fit() reconciles topology
        # with the stage loss afterwards).
        live_split = len(self._optimizers) > 1
        if live_split:
            declared_split: bool | None = True
        elif self._active_loss is not None:
            declared_split = isinstance(self._active_loss, FullELBO) and getattr(
                self._active_loss, "use_split_loss", False
            )
        else:
            declared_split = None  # undeclared: adopt the checkpoint's mode
        if declared_split is not None and ckpt.split_loss != declared_split:
            raise ValueError(
                f"Checkpoint was produced in "
                f"{'split' if ckpt.split_loss else 'single'}-loss mode but the "
                f"live trainer is in "
                f"{'split' if declared_split else 'single'}-loss "
                "mode; refusing to mix optimizer topologies across resume."
            )
        live_mode = ckpt.split_loss if declared_split is None else declared_split
        # Split-mode restore may run before fit() installs the two-optimizer
        # topology (e.g. a direct restore_from_checkpoint call, or the
        # undeclared adopt-the-checkpoint path above); install it now so
        # the ψ optimizer state has a live target.
        if live_mode and not live_split:
            self._install_split_topology()
        # A v3 split payload must carry both optimizer states or neither —
        # exactly one present means a corrupt / hand-mangled payload.
        if live_mode and (ckpt.optimizer_state is None) != (
            ckpt.optimizer_state_psi is None
        ):
            raise RuntimeError(
                "Split-mode checkpoint carries exactly one of optimizer_state "
                "/ optimizer_state_psi; a split-mode resume requires both (or "
                "neither) — the payload is corrupt."
            )
        if ckpt.optimizer_state is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt.optimizer_state)
        else:
            print(
                "[restore] Warning: optimizer state not found in checkpoint "
                "or optimizer is None."
            )
        if ckpt.optimizer_state_psi is not None and self.opt_psi is not None:
            self.opt_psi.load_state_dict(ckpt.optimizer_state_psi)
        if hasattr(self, "ema"):
            if ckpt.ema_state is not None:
                self.ema.shadow = ckpt.ema_state
                if ckpt.ema_decay is not None:
                    self.ema_decay = ckpt.ema_decay
            else:
                # Live weights were just overwritten from the checkpoint,
                # but ``self.ema.shadow`` still holds the *init* snapshot
                # from ``EMA.__init__`` (which ran before restore). Re-
                # snapshot from live so ``EMA.swap`` doesn't validate an
                # untrained model on the next epoch end.
                self.ema.shadow = {
                    k: p.detach().clone()
                    for k, p in self.model.state_dict().items()
                }
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
        # ψ-side LR-scheduler contract guard — mirrors the φθ guard above.
        live_scheduler_psi = self._schedulers[1] if len(self._schedulers) > 1 else None
        if ckpt.scheduler_state_psi is not None and live_scheduler_psi is None:
            raise RuntimeError(
                "Checkpoint carries ψ-side LR scheduler state but the live "
                "trainer has no second (ψ) scheduler; refusing to drop state "
                "silently."
            )
        if ckpt.scheduler_state_psi is None and live_scheduler_psi is not None:
            raise RuntimeError(
                "Live trainer has a ψ-side LR scheduler but the checkpoint "
                "carries no ψ scheduler state; refusing to restart the ψ "
                "schedule from step 0 mid-run."
            )
        if ckpt.scheduler_state_psi is not None and live_scheduler_psi is not None:
            live_scheduler_psi.load_state_dict(ckpt.scheduler_state_psi)
        # grad_accum_steps contract guard — same logic as scaler/scheduler.
        # Loss is divided by ``self.grad_accum_steps`` (see ``_backward_loss``
        # and the accumulation loop), so silently changing it across resume
        # rescales gradients mid-run — an invisible LR shift.
        if ckpt.grad_accum_steps != self.grad_accum_steps:
            raise RuntimeError(
                f"Checkpoint has grad_accum_steps={ckpt.grad_accum_steps} but "
                f"live trainer has grad_accum_steps={self.grad_accum_steps}; "
                "refusing to rescale gradients silently mid-run."
            )
        self.global_step = ckpt.global_step
        # Always set from the payload (legacy v1/v2 default 0) — skip
        # accounting must reflect the producer's run, not whatever the
        # fresh process accumulated before restore.
        self.grad_skip_count = int(getattr(ckpt, "grad_skip_count", 0))
        # Restore the producing stage's start step alongside global_step so
        # the orchestrator's budget / λ-ramp origin math sees the stage's
        # true start (legacy payloads default to 0, matching the single-fit
        # convention where ``_stage_start_step`` is never set).
        self._stage_start_step = int(ckpt.stage_start_step)
        if ckpt.rng_state is not None:
            # Continue the producer's RNG streams (reparam / diffusion noise,
            # loader shuffles) instead of replaying the fresh process's seed.
            # ``map_location`` may have moved the state tensors; the setters
            # require CPU uint8 tensors.
            torch.set_rng_state(ckpt.rng_state["torch_cpu"].cpu())
            cuda_states = ckpt.rng_state.get("torch_cuda") or []
            if torch.cuda.is_available() and cuda_states:
                if len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
                else:
                    log.warning(
                        "[resume] checkpoint carries %d CUDA RNG states but "
                        "%d devices are visible; skipping CUDA RNG restore.",
                        len(cuda_states),
                        torch.cuda.device_count(),
                    )
            np.random.set_state(ckpt.rng_state["numpy"])
            random.setstate(ckpt.rng_state["python"])

    def _build_default_loss(self, total_steps: int):
        """Default loss when the caller declares none.

        When ``hparams.lambda_ramp`` is set, wrap the FullELBO's rate weight
        in :func:`make_lambda_cosine` anchored to ``total_steps``; otherwise
        return the constant-λ FullELBO.
        """
        from ddssm.model.losses import FullELBO
        from ddssm.training.stages import make_lambda_cosine

        ramp = getattr(self.hparams, "lambda_ramp", None)
        split = bool(getattr(self.hparams, "use_split_loss", False))
        if ramp is None:
            return FullELBO(
                rate_lambda=lambda _step: 1.0, use_split_loss=split
            )
        return FullELBO(
            rate_lambda=make_lambda_cosine(
                ramp, total_steps=total_steps, default_end=1.0
            ),
            use_split_loss=split,
        )

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

    def _backward_loss(self, loss: "torch.Tensor | SplitLoss", scaler, amp: bool):
        if isinstance(loss, SplitLoss):
            # Split backward: route each side's gradients only into its own
            # parameter set. The φθ pass must retain the graph — both sides
            # share the encoder subgraph, and the ψ pass walks it second.
            if self._phith_params is None or self._psi_params is None:
                raise RuntimeError(
                    "SplitLoss backward requires the split topology; call "
                    "_install_split_topology() first (fit() does this when "
                    "the active loss has use_split_loss=True)."
                )
            # The caches hold the FULL per-side partition (frozen params
            # included); filter by the live requires_grad here so per-stage
            # freeze/unfreeze after install is honored — backward(inputs=...)
            # rejects requires_grad=False tensors, and a stale trainable
            # list would silently starve a later-unfrozen module.
            phith_live = [p for p in self._phith_params if p.requires_grad]
            psi_live = [p for p in self._psi_params if p.requires_grad]
            lp = loss.phith / self.grad_accum_steps
            lq = loss.psi / self.grad_accum_steps
            # ψ side can be a graph-free zero (e.g. every ψ term degenerate
            # for the batch) — nothing to backprop then. A fully-frozen
            # side is likewise skipped.
            run_psi = bool(psi_live) and lq.requires_grad
            if phith_live:
                # Retain the shared graph only when the ψ pass will walk
                # it second; when φθ is skipped the graph is still fresh
                # for ψ, so no retain is needed on either path.
                (scaler.scale(lp) if amp else lp).backward(
                    inputs=phith_live, retain_graph=run_psi
                )
            if run_psi:
                (scaler.scale(lq) if amp else lq).backward(inputs=psi_live)
        elif amp:
            scaler.scale(loss / self.grad_accum_steps).backward()
        else:
            (loss / self.grad_accum_steps).backward()

    def _accumulate_metrics(self, accum_metrics, metrics: dict):
        if accum_metrics is None:
            return {k: v.detach() for k, v in metrics.items()}
        for k, v in metrics.items():
            accum_metrics[k] = accum_metrics[k] + v.detach()
        return accum_metrics

    def _optimizer_step(self, scaler, amp: bool) -> bool:
        """Clip grads per-optimizer and step optimizer(s)/scheduler(s)/EMA,
        skipping on non-finite grads.

        Returns:
            ``True`` if the step was taken, ``False`` if it was skipped
            because a gradient norm was non-finite (the macro-batch is
            discarded: grads zeroed, no optimizer / scheduler / EMA update,
            ``grad_skip_count`` incremented).
        """
        # AMP: unscale before inspecting grads (unscale→inspect→step order;
        # the scaler is disabled for bf16 so this is a no-op today, but the
        # order stays correct if fp16 scaling is ever enabled).
        if amp:
            for opt in self._optimizers:
                scaler.unscale_(opt)

        # Clip each optimizer's own parameters independently rather than one
        # norm over the whole model. Under the split topology, opt_phith and
        # opt_psi already run with independent LRs/betas because they
        # optimize disjoint parameter sets against different loss scalars
        # (ELBO recon/rate vs. denoising score-matching — see
        # _backward_loss); a shared norm would let a spike on one side
        # rescale the other side's well-behaved grads. In single-optimizer
        # mode this is one clip over the whole model, same as before.
        # ``clip_grad_norm_`` returns the pre-clip norm regardless of
        # whether clipping was applied, so the non-finite check below still
        # sees the true norm. ``max_norm=inf`` (clipping disabled) never
        # rescales.
        max_norm = (
            float(self.clip_grad_norm)
            if self.clip_grad_norm is not None
            else float("inf")
        )
        # Cache per-optimizer flat param lists to skip the per-step Python
        # comprehension over param_groups. The param membership is fixed
        # after construction (frozen/unfrozen state changes ``requires_grad``
        # but not the list itself); ``clip_grad_norm_`` internally skips
        # tensors whose ``.grad is None``. Pass ``foreach=True`` so the
        # norm/scale use batched ``_foreach`` kernels.
        if getattr(self, "_opt_param_cache", None) is None or len(
            self._opt_param_cache
        ) != len(self._optimizers):
            self._opt_param_cache = [
                [p for group in opt.param_groups for p in group["params"]]
                for opt in self._optimizers
            ]
        norms = [
            torch.nn.utils.clip_grad_norm_(
                params, max_norm, foreach=True,
            )
            for params in self._opt_param_cache
        ]

        # Single batched host sync: stack per-optimizer norms + the
        # aggregated L2-of-norms into one tensor, transfer once. Replaces
        # the previous per-norm ``bool()`` / ``float()`` calls (6+ host
        # syncs per step under split-loss) that broke CPU/GPU pipelining.
        stacked_norms = torch.stack(norms)
        combined = torch.linalg.vector_norm(stacked_norms)
        norm_vals = torch.cat([stacked_norms, combined.reshape(1)]).cpu().tolist()
        per_opt_norm_vals = norm_vals[:-1]
        combined_val = norm_vals[-1]

        if not all(math.isfinite(v) for v in per_opt_norm_vals):
            # Discard the macro-batch on EITHER side going non-finite: both
            # sides backward from the same forward pass on the same
            # (possibly poisoned) batch, so a partial step would poison
            # Adam state / EMA shadows on whichever side(s) look clean too.
            for opt in self._optimizers:
                opt.zero_grad(set_to_none=True)
            self.grad_skip_count += 1
            self._last_grad_norm = float("nan")
            self._last_grad_norms_by_opt = (
                per_opt_norm_vals if len(self._optimizers) > 1 else None
            )
            log.warning(
                "[grad-skip] non-finite grad norm(s) %s at step %d (total skips: %d)",
                per_opt_norm_vals,
                self.global_step + 1,
                self.grad_skip_count,
            )
            if amp:
                # GradScaler contract: after unscale_(), update() must still
                # run once per iteration even when scaler.step() is skipped,
                # so per-optimizer bookkeeping resets for the next step.
                # (No-op while the scaler is disabled, as under bf16.)
                scaler.update()
            return False

        # Aggregate as a single L2 norm over the per-optimizer norms for the
        # ``optim/grad_norm`` metric (equals the one norm in single-optimizer
        # mode); per-optimizer values are logged separately when split.
        self._last_grad_norm = combined_val
        if len(self._optimizers) > 1:
            self._last_grad_norms_by_opt = per_opt_norm_vals
        else:
            self._last_grad_norms_by_opt = None
        if amp:
            for opt in self._optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            for opt in self._optimizers:
                opt.step()

        if self._schedulers:
            for sched in self._schedulers:
                sched.step()
        elif self.scheduler is not None:
            # Back-compat: a scheduler assigned directly to
            # ``trainer.scheduler`` without going through
            # ``_install_scheduler`` still steps.
            if len(self._optimizers) > 1 and not self._warned_split_legacy_scheduler:
                self._warned_split_legacy_scheduler = True
                log.warning(
                    "[split] trainer.scheduler was assigned directly while the "
                    "two-optimizer split topology is active: only the phi-theta "
                    "optimizer is scheduled and opt_psi runs UNSCHEDULED. Use "
                    "_install_scheduler(sched) to schedule both sides."
                )
            self.scheduler.step()

        if hasattr(self, "ema") and self.ema is not None and any(
            p.requires_grad for p in self.model.parameters()
        ):
            # Skip only when the whole model is frozen (live weights don't
            # move, so blending shadows toward them is a no-op that would
            # drain any warm-started EMA lag). In single-fit training this
            # guard is always True; it stays as a defensive check.
            with torch.no_grad():
                self.ema.update()
        return True

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
        # Grad-skip diagnostics: the combined grad norm of the last
        # optimizer step (NaN on a skipped step) and the cumulative skip
        # count. Both fit the existing ``optim/*`` MetricSpec. Under the
        # split topology, also surface each optimizer's own (independently
        # clipped) norm so a spike on one side is visible without being
        # averaged away by the combined figure.
        if self._last_grad_norm is not None:
            log_values["optim/grad_norm"] = torch.tensor(
                self._last_grad_norm, device=device
            )
        if self._last_grad_norms_by_opt is not None:
            log_values["optim/grad_norm_phith"] = torch.tensor(
                self._last_grad_norms_by_opt[0], device=device
            )
            log_values["optim/grad_norm_psi"] = torch.tensor(
                self._last_grad_norms_by_opt[1], device=device
            )
        log_values["optim/grad_skips"] = torch.tensor(
            float(self.grad_skip_count), device=device
        )
        # Per-role LR diagnostics: emit both keys when both roles exist so a
        # scheduled run's asymmetric decay is directly visible in metrics.csv.
        # Split mode reads each optimizer's first group; single mode picks
        # the first group per ``role`` tag (roles present when either an
        # lr_schedule was requested via hparams or the caller opted in via
        # claim_psi/psi_betas/weight_decay_psi).
        if len(self._optimizers) > 1:
            log_values["optim/lr_phith"] = torch.tensor(
                float(self._optimizers[0].param_groups[0]["lr"]), device=device
            )
            log_values["optim/lr_psi"] = torch.tensor(
                float(self._optimizers[1].param_groups[0]["lr"]), device=device
            )
        else:
            phith_lrs = [
                g["lr"]
                for g in self._optimizers[0].param_groups
                if g.get("role", "phith") == "phith"
            ]
            psi_lrs = [
                g["lr"]
                for g in self._optimizers[0].param_groups
                if g.get("role") == "psi"
            ]
            if phith_lrs:
                log_values["optim/lr_phith"] = torch.tensor(
                    float(phith_lrs[0]), device=device
                )
            if psi_lrs:
                log_values["optim/lr_psi"] = torch.tensor(
                    float(psi_lrs[0]), device=device
                )
        self.metrics.update(split="train", values=log_values, weight=accum_weight)
        if log_every and (step % log_every == 0):
            self.metrics.step_end("train", self.global_step)

    def _run_validation(
        self,
        val_loader: DataLoader,
        batch_transform: Callable[[dict, torch.device], dict] | None,
        device: torch.device,
        amp: bool = False,
    ):
        self.model.eval()
        # Validate on the EMA model — the same transition weights the
        # sampling path uses (ADR-0005). ``swap`` loads the EMA shadows
        # for the duration of the loop and restores the live weights on
        # exit so training continues unperturbed.
        ema_ctx = (
            self.ema.swap() if getattr(self, "ema", None) is not None else nullcontext()
        )
        # Mirror the training autocast so val and train losses are computed
        # in the same dtype — numerically comparable and faster under bf16.
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp)
        with torch.no_grad(), ema_ctx, amp_ctx:
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
                # Under split-loss mode the loss object returns a SplitLoss
                # pair; log its scalar total (φθ + ψ).
                vlog["loss/total"] = (
                    vloss.total if isinstance(vloss, SplitLoss) else vloss
                )
                self.metrics.update("val", values=vlog, weight=vwin.size(0))

    def _maybe_run_validation(
        self,
        step: int,
        val_loader: DataLoader | None,
        validate_every: int,
        batch_transform: Callable[[dict, torch.device], dict] | None,
        device: torch.device,
        amp: bool = False,
    ):
        if val_loader is None or not validate_every or (step % validate_every != 0):
            return
        self._run_validation(
            val_loader=val_loader,
            batch_transform=batch_transform,
            device=device,
            amp=amp,
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
        self.save_checkpoint(ckpt_name)
        # The pair (step-N, latest) must be atomic at the FS level: a SIGKILL
        # between the two writes would otherwise leave ``ckpt_step{N}`` on disk
        # while ``ckpt_latest`` still points to the previous N-K snapshot, and
        # a preempt-retry would silently lose progress. We mirror step-N to
        # latest by copying into a same-dir tmp file then ``os.replace`` —
        # which is atomic on POSIX, and unlike ``os.link`` is safe on the
        # shared FS (Lustre/NFS) the Tempest cluster runs on.
        d = os.path.dirname(latest_name) or "."
        f = tempfile.NamedTemporaryFile(
            prefix="tmp_latest_",
            suffix=".pth",
            dir=d,
            delete=False,
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
    # on disk" — anything outside this set (notably load_state_dict shape/
    # key mismatches, which raise RuntimeError) is a schema bug that must
    # surface loudly rather than silently restart from step 0 (ADR-0009
    # preempt path: a 6-hour stage-2 trial that hits a schema-incompatible
    # ckpt would otherwise lose all progress without a trace).
    #
    # File-read failures (corrupt zip, truncated pickle, OSError) are
    # translated to NoUsableCheckpointError at the checkpoint boundary
    # (see ``Checkpoint.load``), so RuntimeError here can only come from
    # ``load_state_dict`` and is correctly excluded.
    _RESUME_NO_CKPT_EXCEPTIONS: tuple[type[BaseException], ...] = (
        FileNotFoundError,
        IsADirectoryError,
        NoUsableCheckpointError,
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
            log.error(
                "[resume] FALLBACK TO FRESH START — failed to load "
                "checkpoint %r: %s: %s. global_step reset to 0.",
                resume_from,
                type(e).__name__,
                e,
            )
            # Persist a marker next to the checkpoint so post-hoc grep
            # across sweep dirs can find silent restarts even if the log
            # stream was lost.
            try:
                marker = os.path.join(
                    os.path.dirname(resume_from) or ".",
                    "_FRESH_START_FALLBACK.txt",
                )
                with open(marker, "a") as f:
                    f.write(
                        f"{datetime.datetime.now().isoformat()}\t"
                        f"resume_from={resume_from}\t"
                        f"{type(e).__name__}: {e}\n"
                    )
            except OSError:
                pass

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

        self._warn_if_ema_decay_too_high(total_steps)

        # Per ADR-0004: ensure an active loss object exists. The
        # orchestrator installs one per stage; for single-fit runs we
        # build a default ``FullELBO`` from hparams here.
        if self._active_loss is None:
            self._active_loss = self._build_default_loss(total_steps)

        # Split-loss topology decision happens here (not __init__): the
        # orchestrator installs the per-stage loss after construction.
        # fit() is re-entered per stage, and the topology must MATCH the
        # active loss both ways. Split direction: the ``len`` guard
        # preserves an existing two-optimizer topology (and its Adam
        # moments) instead of rebuilding it every stage. Single direction:
        # a split topology left over from an earlier stage is downgraded —
        # otherwise a single-loss stage would keep two optimizers (ψ with
        # split betas) and its checkpoints would record split_loss=True.
        loss_wants_split = isinstance(self._active_loss, FullELBO) and getattr(
            self._active_loss, "use_split_loss", False
        )
        if loss_wants_split and len(self._optimizers) < 2:
            self._install_split_topology()
        elif not loss_wants_split and len(self._optimizers) > 1:
            log.warning(
                "[split] active loss is single-mode but the split "
                "two-optimizer topology is installed; downgrading to the "
                "single AdamW (psi Adam state is discarded)."
            )
            self.optimizer = torch.optim.AdamW(
                param_groups_for_adamw(
                    self.model,
                    enc_lr=self.hparams.enc_lr,
                    dec_lr=self.hparams.dec_lr,
                    trans_lr=self.hparams.trans_lr,
                    weight_decay=self.hparams.weight_decay,
                    psi_betas=self._hparams_psi_betas(),
                    weight_decay_psi=self.weight_decay_psi,
                    claim_psi=getattr(self.hparams, "lr_schedule", None) is not None,
                ),
                betas=(0.9, 0.999),
                eps=1e-8,
                fused=True,
            )
            self._optimizers = [self.optimizer]
            self._opt_param_cache = None  # invalidated on topology change
            self.opt_psi = None
            self._phith_params = None
            self._psi_params = None
            # Reconcile the scheduler topology mirror: keep the φθ
            # scheduler when one is installed (it may need reinstalling —
            # _rebuild_optimizer warns on that), drop the ψ-side one.
            self._schedulers = [self.scheduler] if self.scheduler is not None else []

        # Install per-role LR schedules from hparams (opt-in). Runs AFTER
        # topology reconciliation so it sees the final optimizer list, and
        # BEFORE _safe_resume so the checkpoint scheduler-contract guards
        # (in restore_from_checkpoint) see the live schedulers. No-op
        # when hparams.lr_schedule is None or already installed.
        if getattr(self.hparams, "lr_schedule", None) is not None and not self._schedulers:
            self._install_lr_schedule(
                self.hparams.lr_schedule,
                total_steps=int(total_steps),
                lambda_ramp=getattr(self.hparams, "lambda_ramp", None),
            )

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

        # Reset train meters so stage-N values don't leak into stage-(N+1)
        # logged means. Val resets per epoch_end; train has no equivalent
        # flush-and-reset cycle, so we do it explicitly at fit() entry.
        self.metrics._split("train").reset()

        # see if we should resume
        self._safe_resume(resume_from)

        # Compile each optimizer's ``.step()`` — reduces the ~1.5 ms/step
        # AdamW Python dispatch to one compiled kernel launch. Applied
        # here (rather than at optimizer construction) so it runs after
        # topology is settled and LR schedules are installed. Idempotent:
        # ``_ddssm_compiled`` guard prevents re-compilation on repeated
        # fit() calls.
        _compile_optimizer_step(self._optimizers)

        # Two-region compiled step (Compiled Autograd tutorial + "Compiling
        # the optimizer" recipe). Gate: DDSSM_COMPILE_STEP=1 AND
        # single-loss AND grad_accum_steps==1.
        _use_compiled_step = (
            _compiled_step_enabled()
            and not loss_wants_split
            and self.grad_accum_steps == 1
        )
        compiled_fwd_bwd = None
        compiled_opt_ema = None
        lam_t = None
        if _use_compiled_step:
            opt0 = self._optimizers[0]
            _params_flat = [p for g in opt0.param_groups for p in g["params"]]
            _max_norm = (
                float(self.clip_grad_norm)
                if self.clip_grad_norm is not None else float("inf")
            )
            _fwd_bwd_fn, _opt_ema_fn = _make_split_step(
                self.model, self._active_loss, opt0, _params_flat, _max_norm,
                ema=self.ema,
            )
            _fused_mode = os.environ.get(
                "DDSSM_TORCH_COMPILE_MODE", ""
            ).strip() or None
            _fused_kwargs: dict = {"mode": _fused_mode} if _fused_mode else {}
            if _fused_mode in {"reduce-overhead", "max-autotune"}:
                _fused_kwargs["dynamic"] = False
            compiled_fwd_bwd = torch.compile(_fwd_bwd_fn, **_fused_kwargs)
            # ``opt.step`` is compiled standalone by _compile_optimizer_step;
            # wrapping the opt/EMA function in another compile would nest
            # and trigger per-param-group recompiles.
            compiled_opt_ema = _opt_ema_fn
            lam_t = torch.zeros((), device=device)

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

                    if _use_compiled_step:
                        try:
                            batch = next(data_iter)
                        except StopIteration:
                            data_iter = iter(train_loader)
                            batch = next(data_iter)
                        batch = self._prepare_batch(
                            batch, device, batch_transform
                        )
                        step_within_stage = (
                            self.global_step - self._stage_start_step + 1
                        )
                        lam_val = float(
                            self._active_loss.rate_lambda(step_within_stage)
                        )
                        lam_t.fill_(lam_val)
                        fwd_result = compiled_fwd_bwd(batch, lam_t)
                        grad_norm = compiled_opt_ema()
                        metrics = self.model._build_metrics_from_flat(
                            (fwd_result[0], grad_norm, *fwd_result[1:])
                        )
                        if self._active_loss.lambda_at(step_within_stage) is not None:
                            metrics["optim/lambda"] = torch.tensor(lam_val)
                        # Nonfinite policy under compile is detect-and-count
                        # (opt.step already ran); DDSSM_COMPILE_STEP=0
                        # restores the eager pre-step skip semantics.
                        gn_val = float(grad_norm)
                        if not math.isfinite(gn_val):
                            self.grad_skip_count += 1
                            self._last_grad_norm = float("nan")
                            log.warning(
                                "[grad-skip:compiled] non-finite grad norm "
                                "%s at step %d (total skips: %d) — step was "
                                "applied before detection",
                                gn_val, step, self.grad_skip_count,
                            )
                        else:
                            self._last_grad_norm = gn_val
                        if self._schedulers:
                            for sched in self._schedulers:
                                sched.step()
                        elif self.scheduler is not None:
                            self.scheduler.step()
                        accum_loss_t = fwd_result[0]
                        accum_metrics = self._accumulate_metrics(None, metrics)
                        accum_weight = batch["observed_data"].size(0)
                        accum_loss = float(accum_loss_t)
                        if self.abort_on_nonfinite_loss and not math.isfinite(
                            accum_loss
                        ):
                            raise FloatingPointError(
                                f"Non-finite training loss ({accum_loss}) at "
                                f"step {step}; aborting "
                                f"(abort_on_nonfinite_loss=True)."
                            )
                        accum_metrics = self._finalize_accum_metrics(accum_metrics)
                    else:
                        for opt in self._optimizers:
                            opt.zero_grad(set_to_none=True)

                        accum_loss_t: torch.Tensor | None = None
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

                            loss_detached = loss.detach()
                            if isinstance(loss_detached, SplitLoss):
                                # Normalize to a scalar tensor (φθ + ψ) so the
                                # microstep accumulation below stays plain tensor
                                # arithmetic; single-mode passes through untouched.
                                loss_detached = loss_detached.total
                            # Per-microstep guard: check before ``_backward_loss``
                            # so a NaN/Inf micro-batch never poisons ``.grad`` via
                            # backprop. Checking the summed accum_loss post-backward
                            # would already be too late. The host-side read forces
                            # a device sync per microstep, so it only runs when the
                            # guard is enabled; otherwise the loss accumulates
                            # on-device and syncs once per optimizer step.
                            if self.abort_on_nonfinite_loss:
                                loss_scalar = float(loss_detached)
                                if not math.isfinite(loss_scalar):
                                    raise FloatingPointError(
                                        f"Non-finite training loss ({loss_scalar}) "
                                        f"at step {step}; aborting "
                                        f"(abort_on_nonfinite_loss=True)."
                                    )

                            self._backward_loss(loss, scaler=scaler, amp=amp)

                            accum_loss_t = (
                                loss_detached
                                if accum_loss_t is None
                                else accum_loss_t + loss_detached
                            )
                            accum_metrics = self._accumulate_metrics(accum_metrics, metrics)
                            accum_weight += weight

                        self._optimizer_step(scaler=scaler, amp=amp)
                        # Single host sync per optimizer step, queued after the
                        # optimizer kernels so it doesn't stall them.
                        accum_loss = float(accum_loss_t)
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
                        amp=amp,
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

        # NOTE: ``self.metrics`` is deliberately NOT closed here. fit() runs
        # once per stage under the orchestrator, and closing tore down the
        # CSV/TB/W&B sinks after the FIRST stage (W&B even uploaded its
        # final artifacts mid-run). The run owner — ``Experiment.train`` —
        # closes the store once, after all stages.
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
