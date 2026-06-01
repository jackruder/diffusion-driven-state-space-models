"""Opt-in ``torch.compile`` wrapper gated by ``DDSSM_TORCH_COMPILE``."""

from __future__ import annotations

import os
import logging

from torch import nn
import torch._dynamo

log = logging.getLogger(__name__)


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

    ``torch._dynamo`` is set to fall back to eager on graph-compile failures so a
    long run never crashes for a backend it cannot compile.
    """
    mode = os.environ.get("DDSSM_TORCH_COMPILE", "auto").strip().lower()
    if mode in {"0", "false", "no"}:
        return module

    # Fall back to eager (rather than crash) if a graph fails to compile at run
    # time — the try/except below only guards the compile *call*, not the first
    # forward where most backend errors actually surface.
    torch._dynamo.config.suppress_errors = True

    try:
        module.compile(dynamic=dynamic)
    except RuntimeError as e:  # pragma: no cover - defensive fallback
        log.warning("torch.compile disabled after failure: %s", e, exc_info=True)
    return module
