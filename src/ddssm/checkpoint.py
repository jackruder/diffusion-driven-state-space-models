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

import difflib
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import torch

log = logging.getLogger(__name__)

_FORMAT = "ddssm_ckpt_v1"


def _namespace_to_dict(obj: Any) -> Any:
    """Recursively convert SimpleNamespace / objects to plain dicts."""
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _namespace_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_namespace_to_dict(v) for v in obj)
    return obj


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
    # Transitional: asdict dump of ``model.config``. Dead on the read
    # path post ADR-0005, kept one release as a debugging aid.
    config: Any = None

    @classmethod
    def from_trainer(cls, trainer) -> "Checkpoint":
        ema = getattr(trainer, "ema", None)
        return cls(
            model_state=trainer.model.state_dict(),
            model_config_yaml=getattr(trainer, "_model_config_yaml", None),
            optimizer_state=(
                trainer.optimizer.state_dict()
                if trainer.optimizer is not None else None
            ),
            ema_decay=trainer.ema_decay,
            ema_state=getattr(ema, "shadow", None),
            global_step=int(trainer.global_step),
            grad_accum_steps=int(trainer.grad_accum_steps),
            config=_namespace_to_dict(trainer.model.config),
        )

    def to_payload(self) -> dict:
        return {
            "_format": _FORMAT,
            "config": self.config,
            "model_config_yaml": self.model_config_yaml,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "ema_decay": self.ema_decay,
            "ema_state": self.ema_state,
            "global_step": self.global_step,
            "grad_accum_steps": self.grad_accum_steps,
        }

    def save(self, path: str) -> None:
        _atomic_save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str, *, device: torch.device) -> "Checkpoint":
        payload = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or "model_state" not in payload:
            # Legacy raw state_dict (pre-payload checkpoints).
            return cls(model_state=payload)
        return cls(
            model_state=payload["model_state"],
            model_config_yaml=payload.get("model_config_yaml"),
            optimizer_state=payload.get("optimizer_state"),
            ema_decay=payload.get("ema_decay"),
            ema_state=payload.get("ema_state"),
            global_step=int(payload.get("global_step", 0)),
            grad_accum_steps=int(payload.get("grad_accum_steps", 1)),
            config=payload.get("config"),
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
            diff = "\n".join(difflib.unified_diff(
                ckpt.model_config_yaml.splitlines(),
                expected_model_config_yaml.splitlines(),
                fromfile="checkpoint", tofile="experiment=",
                lineterm="",
            ))
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
        log.warning(
            "No checkpoint provided; using randomly-initialised weights."
        )
    else:
        load_into_model(
            model, checkpoint_path, device=device,
            expected_model_config_yaml=getattr(
                experiment, "model_config_yaml", None,
            ),
            load_ema=load_ema,
        )
        log.info("Loaded checkpoint from %s", checkpoint_path)
    model.train() if train else model.eval()
    return model
