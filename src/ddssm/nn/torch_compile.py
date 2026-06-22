"""Opt-in ``torch.compile`` wrapper gated by ``DDSSM_TORCH_COMPILE``."""

from __future__ import annotations

import os
import logging

import torch
from torch import nn
import torch._dynamo

log = logging.getLogger(__name__)


def _compile_enabled() -> bool:
    mode = os.environ.get("DDSSM_TORCH_COMPILE", "auto").strip().lower()
    return mode not in {"0", "false", "no"}


def _configure_dynamo() -> None:
    """Dynamo settings applied whenever we compile anything.

    * ``suppress_errors``: fall back to eager (never crash) on a graph that won't
      compile — the try/except only guards the compile *call*, not the first
      forward where most backend errors surface.
    * ``capture_scalar_outputs``: trace ``Tensor.item()`` / data-dependent scalars
      symbolically instead of graph-breaking. The encoder per-step body hits one
      (a torch-internal scalar read); without this, compiling it graph-breaks and
      most of the fusion win is lost.
    """
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.capture_scalar_outputs = True


def maybe_compile(module: nn.Module, *, dynamic: bool = False) -> nn.Module:
    """Compile ``module`` in place when enabled, preserving its ``state_dict``.

    Uses :meth:`torch.nn.Module.compile` rather than :func:`torch.compile` so the
    module keeps its identity and ``state_dict`` keys (no ``_orig_mod.`` prefix) —
    checkpoints stay portable across compiled / eager runs. The same object is
    returned for call-site convenience. **Callers must invoke the module via
    ``__call__`` (``module(...)``), not ``module.forward(...)``** — the in-place
    compile only hooks ``__call__``.

    Env var ``DDSSM_TORCH_COMPILE`` controls behavior:
      - ``0/false/no``: disabled
      - ``1/true/yes`` / unset / ``auto``: enabled (CUDA and CPU)
    """
    if not _compile_enabled():
        return module
    _configure_dynamo()
    try:
        module.compile(dynamic=dynamic)
    except RuntimeError as e:  # pragma: no cover - defensive fallback
        log.warning("torch.compile disabled after failure: %s", e, exc_info=True)
    return module


def maybe_compile_fn(fn, *, dynamic: bool = False):
    """Compile a callable / bound method (not an ``nn.Module``) when enabled.

    For per-step bodies like the encoder's ``_forward_with_stats`` that orchestrate
    several sub-modules and aren't themselves modules. Returns a ``torch.compile``
    wrapper (there's no ``state_dict`` to preserve, so the ``_orig_mod`` concern
    doesn't apply) or ``fn`` unchanged when disabled / on failure.
    """
    if not _compile_enabled():
        return fn
    _configure_dynamo()
    try:
        return torch.compile(fn, dynamic=dynamic)
    except RuntimeError as e:  # pragma: no cover - defensive fallback
        log.warning("torch.compile(fn) disabled after failure: %s", e, exc_info=True)
        return fn
