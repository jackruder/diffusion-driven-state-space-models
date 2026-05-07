from __future__ import annotations

import logging
import os

import torch
from torch import nn

log = logging.getLogger(__name__)


def maybe_compile(module: nn.Module, *, dynamic: bool = False) -> nn.Module:
    """Compile module when enabled, with stable defaults for CUDA runs.

    Env var ``DDSSM_TORCH_COMPILE`` controls behavior:
      - ``0/false/no``: disabled
      - ``1/true/yes``: force-enabled
      - unset/``auto``: enabled on CPU, disabled on CUDA
    """

    mode = os.environ.get("DDSSM_TORCH_COMPILE", "auto").strip().lower()
    if mode in {"0", "false", "no"}:
        return module
    if mode == "auto" and torch.cuda.is_available():
        return module

    try:
        return torch.compile(module, dynamic=dynamic)
    except Exception as e:  # pragma: no cover - defensive fallback
        log.warning("torch.compile disabled after failure: %s", e)
        return module
