"""Checkpoint: the single owner of the ``.pth`` payload schema (ADR-0005).

Save and load are symmetric here — the payload keys, format version, and
the model-config cross-check live in one module instead of being split
across a trainer save method, a trainer restore method, and a free load
function.

Two entry points for the two load modes:

* :func:`prepare_model` — what the standalone stages (eval / viz /
  variance) call. Builds the model, loads the checkpoint, cross-checks
  the saved model config against the passed ``experiment=``, and sets
  eval / train mode. ``load_ema`` optionally swaps the diffusion
  transition to its EMA shadows for sampling fidelity.
* :func:`DDSSMTrainer.restore_from_checkpoint` (in ``train.py``) — the
  resume path, which additionally restores optimiser + EMA tracker +
  step counter. It delegates the payload parsing here.
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Any
import difflib
import logging
import tempfile
from dataclasses import dataclass

import numpy as np
import torch

log = logging.getLogger(__name__)


class NoUsableCheckpointError(Exception):
    """The checkpoint file on disk is unreadable (missing / truncated / corrupt).

    Raised at the file-read boundary in :meth:`Checkpoint.load` so that
    the trainer's resume path can distinguish "no ckpt to load" from
    "ckpt loaded but load_state_dict rejected it". The former legitimately
    means fall-back-to-fresh-start (preempt-retry semantics); the latter
    is a schema-drift bug and must surface loudly.
    """


_FORMAT = "ddssm_ckpt_v3"
# Older payloads we can still load (we only ever bumped here). On load, any
# fields a newer schema added (v2: ``scaler_state``, ``scheduler_state``;
# v3: ``optimizer_state_psi``, ``scheduler_state_psi``, ``split_loss``,
# ``grad_skip_count``) default to ``None`` / ``False`` / ``0``.
_SUPPORTED_FORMATS = {"ddssm_ckpt_v1", "ddssm_ckpt_v2", "ddssm_ckpt_v3"}


def _atomic_save(obj: Any, path: str) -> None:
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


@dataclass
class Checkpoint:
    """The parsed ``.pth`` payload — the checkpoint schema in one place."""

    model_state: dict
    model_config_yaml: str | None = None
    optimizer_state: dict | None = None
    ema_decay: float | None = None
    ema_state: dict | None = None
    global_step: int = 0
    grad_accum_steps: int = 1
    # v2 additions. ``None`` means "the producer wasn't using one" (e.g.
    # GradScaler disabled, no LR scheduler); a non-None value carries the
    # corresponding ``state_dict()`` and the trainer's restore path will
    # refuse to silently drop it on the live side.
    scaler_state: dict | None = None
    scheduler_state: dict | None = None
    # The trainer's ``_stage_start_step`` (the global step at which the
    # current fit phase began). Restored together with ``global_step`` so a
    # preempt-retry computes the remaining budget and λ-ramp origin from the
    # phase's true start rather than the fresh process's zeroed counter.
    # Defaults to 0 for legacy payloads and fresh runs.
    stage_start_step: int = 0
    # Global RNG streams (torch CPU / per-device CUDA, numpy, python) at
    # save time, so a resume continues the same noise / shuffle streams
    # instead of replaying the fresh process's seed. The dataloader's
    # position within its epoch is NOT captured: on resume the epoch
    # iterator restarts, reshuffled from the restored torch state.
    # ``None`` for legacy payloads (restore skips it).
    rng_state: dict | None = None
    # v3 additions (split-loss mode, M7). ``optimizer_state_psi`` /
    # ``scheduler_state_psi`` carry the ψ-side optimizer / scheduler
    # ``state_dict()`` when the producer ran the two-optimizer split
    # topology; ``split_loss`` records whether the producing trainer was
    # in split mode (topology, not loss flag, is the source of truth).
    # ``grad_skip_count`` is the cumulative non-finite-gradient skip
    # counter so skip accounting survives preempt-resume. Legacy v1/v2
    # payloads lack all four keys and default them on load.
    optimizer_state_psi: dict | None = None
    scheduler_state_psi: dict | None = None
    split_loss: bool = False
    grad_skip_count: int = 0

    @classmethod
    def from_trainer(cls, trainer) -> Checkpoint:
        """Snapshot a trainer's state into a :class:`Checkpoint`.

        Scaler state is captured only when the GradScaler is enabled (a
        disabled scaler carries nothing worth resuming).
        """
        ema = getattr(trainer, "ema", None)
        scaler = getattr(trainer, "scaler", None)
        scheduler = getattr(trainer, "scheduler", None)
        # v3 split-mode capture. Topology (``_optimizers``) is the source
        # of truth for ``split_loss`` — it works even when the trainer's
        # ``_active_loss`` was never installed (e.g. save before fit()).
        opt_psi = getattr(trainer, "opt_psi", None)
        schedulers = getattr(trainer, "_schedulers", [])
        return cls(
            model_state=trainer.model.state_dict(),
            model_config_yaml=getattr(trainer, "_model_config_yaml", None),
            optimizer_state=(
                trainer.optimizer.state_dict()
                if trainer.optimizer is not None
                else None
            ),
            ema_decay=trainer.ema_decay,
            ema_state=getattr(ema, "shadow", None),
            global_step=int(trainer.global_step),
            grad_accum_steps=int(trainer.grad_accum_steps),
            stage_start_step=int(getattr(trainer, "_stage_start_step", 0)),
            rng_state={
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else []
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            # Only persist scaler state when scaling is actually live —
            # a disabled GradScaler carries no information worth resuming
            # and a non-None entry on disk is the contract guard's signal
            # that the producer was running AMP fp16.
            scaler_state=(
                scaler.state_dict()
                if scaler is not None and scaler.is_enabled()
                else None
            ),
            scheduler_state=(scheduler.state_dict() if scheduler is not None else None),
            optimizer_state_psi=(opt_psi.state_dict() if opt_psi is not None else None),
            scheduler_state_psi=(
                schedulers[1].state_dict() if len(schedulers) > 1 else None
            ),
            split_loss=len(getattr(trainer, "_optimizers", [])) > 1,
            grad_skip_count=int(getattr(trainer, "grad_skip_count", 0)),
        )

    def to_payload(self) -> dict:
        return {
            "_format": _FORMAT,
            "model_config_yaml": self.model_config_yaml,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "ema_decay": self.ema_decay,
            "ema_state": self.ema_state,
            "global_step": self.global_step,
            "grad_accum_steps": self.grad_accum_steps,
            "stage_start_step": self.stage_start_step,
            "rng_state": self.rng_state,
            "scaler_state": self.scaler_state,
            "scheduler_state": self.scheduler_state,
            "optimizer_state_psi": self.optimizer_state_psi,
            "scheduler_state_psi": self.scheduler_state_psi,
            "split_loss": self.split_loss,
            "grad_skip_count": self.grad_skip_count,
        }

    def save(self, path: str) -> None:
        _atomic_save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str, *, device: torch.device) -> Checkpoint:
        """Parse a ``.pth`` payload into a :class:`Checkpoint`.

        Tolerates legacy raw ``state_dict`` payloads (no ``model_state`` key)
        and v1 payloads (missing ``scaler_state`` / ``scheduler_state``, which
        default to ``None``); an unknown ``_format`` is loaded best-effort
        with a warning.

        Read-time failures (missing file, truncated pickle, corrupt zip)
        are translated into :class:`NoUsableCheckpointError` so callers can
        distinguish them from post-read schema errors, which are bugs and
        must propagate.
        """
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except (FileNotFoundError, IsADirectoryError):
            raise
        except (EOFError, pickle.UnpicklingError, RuntimeError, OSError) as e:
            raise NoUsableCheckpointError(
                f"failed to read checkpoint {path!r}: {type(e).__name__}: {e}"
            ) from e
        if not isinstance(payload, dict) or "model_state" not in payload:
            # Legacy raw state_dict (pre-payload checkpoints).
            return cls(model_state=payload)
        fmt = payload.get("_format")
        if fmt is not None and fmt not in _SUPPORTED_FORMATS:
            log.warning(
                "Unknown checkpoint _format=%r (supported: %s); loading best-effort.",
                fmt,
                sorted(_SUPPORTED_FORMATS),
            )
        return cls(
            model_state=payload["model_state"],
            model_config_yaml=payload.get("model_config_yaml"),
            optimizer_state=payload.get("optimizer_state"),
            ema_decay=payload.get("ema_decay"),
            ema_state=payload.get("ema_state"),
            global_step=int(payload.get("global_step", 0)),
            grad_accum_steps=int(payload.get("grad_accum_steps", 1)),
            stage_start_step=int(payload.get("stage_start_step", 0)),
            rng_state=payload.get("rng_state"),
            # v1 payloads never wrote these — ``.get`` defaults to ``None``,
            # which the trainer's contract guard treats as "producer had no
            # scaler/scheduler".
            scaler_state=payload.get("scaler_state"),
            scheduler_state=payload.get("scheduler_state"),
            # v1/v2 payloads never wrote the v3 split-mode fields — default
            # to "producer was single-mode, no skips".
            optimizer_state_psi=payload.get("optimizer_state_psi"),
            scheduler_state_psi=payload.get("scheduler_state_psi"),
            split_loss=bool(payload.get("split_loss", False)),
            grad_skip_count=int(payload.get("grad_skip_count", 0)),
        )


def save(trainer, path: str) -> None:
    """Persist ``trainer`` state to ``path`` (atomic write)."""
    Checkpoint.from_trainer(trainer).save(path)


def load_into_model(
    model: torch.nn.Module,
    path: str,
    *,
    device: torch.device,
    expected_model_config_yaml: str | None = None,
    load_ema: bool = False,
    strict: bool = True,
) -> Checkpoint:
    """Load ``path`` into ``model``, cross-checking the saved model config.

    When ``expected_model_config_yaml`` is given and the checkpoint
    carries ``model_config_yaml`` from training time, the two are diffed;
    any difference emits a WARNING with a unified diff so silent semantic
    drift surfaces loudly (shape mismatches still fail hard in
    ``load_state_dict``).

    With ``load_ema=True`` the diffusion ``transition`` is swapped to its
    EMA shadows after loading — the weights the sampling path used at
    training time.

    Returns the parsed :class:`Checkpoint` so callers (the resume path)
    can also read optimiser / EMA / step state.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path!r}")
    ckpt = Checkpoint.load(path, device=device)

    if ckpt.model_config_yaml is not None and expected_model_config_yaml is not None:
        if ckpt.model_config_yaml.strip() != expected_model_config_yaml.strip():
            diff = "\n".join(
                difflib.unified_diff(
                    ckpt.model_config_yaml.splitlines(),
                    expected_model_config_yaml.splitlines(),
                    fromfile="checkpoint",
                    tofile="experiment=",
                    lineterm="",
                )
            )
            log.warning(
                "Model config drift between checkpoint and passed "
                "experiment=. Loading state from checkpoint anyway "
                "(parameter shapes matched). Diff:\n%s",
                diff,
            )

    model.load_state_dict(ckpt.model_state, strict=strict)

    if load_ema:
        if ckpt.ema_state is not None and hasattr(model, "transition"):
            model.transition.load_state_dict(ckpt.ema_state, strict=strict)
        else:
            log.warning(
                "load_ema requested but checkpoint has no ema_state or model "
                "has no transition; using live weights."
            )
    return ckpt


def prepare_model(
    experiment,
    *,
    checkpoint_path: str | None,
    device: torch.device,
    train: bool = False,
    load_ema: bool = True,
) -> torch.nn.Module:
    """Build + load the experiment's model for a standalone stage.

    Moves the model to ``device``, loads ``checkpoint_path`` with the
    model-config cross-check (so no stage can forget it), and sets eval
    mode — or train mode when ``train=True`` (e.g. a counterfactual
    runner needing train-mode layers).

    ``load_ema`` defaults to ``True``: inference loads the transition's
    EMA shadows — the weights the sampling path used at training time
    (ADR-0005). Pass ``load_ema=False`` for the rare case that wants the
    raw live weights.
    """
    model = experiment.model.to(device)
    if checkpoint_path is None:
        log.warning("No checkpoint provided; using randomly-initialised weights.")
    else:
        # strict=False: a training checkpoint may carry training-only buffers
        # that a freshly-built eval model lacks. Shape mismatches still
        # hard-fail inside load_state_dict; this only tolerates such missing/
        # extra leaf buffers (matches the eval_baselines / probe loaders).
        load_into_model(
            model,
            checkpoint_path,
            device=device,
            expected_model_config_yaml=getattr(
                experiment,
                "model_config_yaml",
                None,
            ),
            load_ema=load_ema,
            strict=False,
        )
        log.info("Loaded checkpoint from %s", checkpoint_path)
    model.train() if train else model.eval()
    return model
