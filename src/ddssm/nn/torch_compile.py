"""Opt-in ``torch.compile`` wrapper gated by ``DDSSM_TORCH_COMPILE``.

Default is STRICT: any compile-call failure or dynamo tracing error raises,
so runs never silently drop to eager. Set ``DDSSM_TORCH_COMPILE=soft`` to
restore the previous silent-fallback behavior, or ``=0`` to disable compile.

On NixOS the triton backend calls ``/sbin/ldconfig`` to locate
``libcuda.so.1`` and ``ptxas``; that binary doesn't exist, so triton (and
therefore inductor codegen) fails. :func:`_autoset_nixos_triton_paths`
detects the standard NixOS locations (``/run/opengl-driver/lib`` and the
CUDA-merged store path exposed by ``CUDA_HOME``) and populates
``TRITON_LIBCUDA_PATH`` / ``TRITON_PTXAS_PATH`` when they are unset — so
compile works out of the box on this machine.
"""

from __future__ import annotations

import os
import shutil
import logging

import torch
from torch import nn
import torch._dynamo

log = logging.getLogger(__name__)


def _autoset_nixos_triton_paths() -> None:
    """Populate ``TRITON_LIBCUDA_PATH`` / ``TRITON_PTXAS_PATH`` and ensure
    ``openssl`` is on ``PATH`` for inductor's header cache.

    Triton's NVIDIA driver backend shells out to ``/sbin/ldconfig -p`` unless
    ``TRITON_LIBCUDA_PATH`` is set. On NixOS ldconfig doesn't exist at that
    path, so the subprocess raises ``FileNotFoundError`` and inductor codegen
    then bubbles it up as ``InductorError``. Inductor's precompiled-header
    cache also shells out to ``openssl sha512`` for content hashing, which
    fails identically when openssl isn't on ``PATH`` in the dev shell.
    Setting the two env vars ahead of time short-circuits triton's lookup;
    extending ``PATH`` short-circuits inductor's.

    Only writes an env var / extends PATH when (a) the value is not already
    set / a suitable binary isn't already on PATH, and (b) the target path
    exists — so a user-supplied override always wins and this is a no-op on
    non-NixOS hosts.
    """
    # libcuda: /run/opengl-driver/lib/libcuda.so.1 on NixOS with driver present
    if os.environ.get("TRITON_LIBCUDA_PATH") is None:
        p = "/run/opengl-driver/lib"
        if os.path.exists(os.path.join(p, "libcuda.so.1")):
            os.environ["TRITON_LIBCUDA_PATH"] = p
            log.debug("torch_compile: set TRITON_LIBCUDA_PATH=%s", p)
    # ptxas: prefer PATH, else fall back to $CUDA_HOME/bin
    if os.environ.get("TRITON_PTXAS_PATH") is None:
        ptxas = shutil.which("ptxas")
        if ptxas is None and (cuda_home := os.environ.get("CUDA_HOME")):
            candidate = os.path.join(cuda_home, "bin", "ptxas")
            if os.path.exists(candidate):
                ptxas = candidate
        if ptxas is not None:
            os.environ["TRITON_PTXAS_PATH"] = ptxas
            log.debug("torch_compile: set TRITON_PTXAS_PATH=%s", ptxas)
    # openssl (inductor cache):  add its containing dir to PATH if absent.
    if shutil.which("openssl") is None:
        import glob
        # Try a few candidate globs common on NixOS: nix profile + nix store.
        candidates = []
        for pattern in (
            "/nix/store/*-openssl-*-bin/bin/openssl",
            "/etc/profiles/*/bin/openssl",
            "/run/current-system/sw/bin/openssl",
        ):
            candidates.extend(glob.glob(pattern))
        for cand in candidates:
            if os.path.exists(cand):
                bin_dir = os.path.dirname(cand)
                os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
                log.debug("torch_compile: prepended %s to PATH for openssl", bin_dir)
                break


_OFF_VALUES = {"0", "false", "no", "off"}
_SOFT_VALUES = {"soft", "permissive"}


def _compile_mode() -> str:
    """Return one of ``"off" | "soft" | "strict"`` from ``DDSSM_TORCH_COMPILE``.

    Default (unset or any unknown value) is ``"strict"``: compile failures
    and dynamo tracing errors both raise — the run never silently drops to
    eager. Opt out with ``=0`` (disabled) or ``=soft`` (silent fallback).
    """
    mode = os.environ.get("DDSSM_TORCH_COMPILE", "strict").strip().lower()
    if mode in _OFF_VALUES:
        return "off"
    if mode in _SOFT_VALUES:
        return "soft"
    return "strict"


def _configure_dynamo(mode: str) -> None:
    """Dynamo settings applied whenever we compile anything.

    * Runs :func:`_autoset_nixos_triton_paths` first so the triton backend
      can find libcuda / ptxas without shelling out to ``/sbin/ldconfig``.
    * ``suppress_errors``: True under ``"soft"`` (dynamo tracing errors
      silently fall back to eager); False under ``"strict"`` (they raise).
    * ``capture_scalar_outputs``: trace ``Tensor.item()`` / data-dependent
      scalars symbolically instead of graph-breaking. The encoder per-step
      body hits one (a torch-internal scalar read); without this, compiling
      it graph-breaks and most of the fusion win is lost.
    """
    _autoset_nixos_triton_paths()
    torch._dynamo.config.suppress_errors = (mode == "soft")
    torch._dynamo.config.capture_scalar_outputs = True


def maybe_compile(module: nn.Module, *, dynamic: bool = False) -> nn.Module:
    """Compile ``module`` in place when enabled, preserving its ``state_dict``.

    Uses :meth:`torch.nn.Module.compile` rather than :func:`torch.compile` so the
    module keeps its identity and ``state_dict`` keys (no ``_orig_mod.`` prefix) —
    checkpoints stay portable across compiled / eager runs. The same object is
    returned for call-site convenience. **Callers must invoke the module via
    ``__call__`` (``module(...)``), not ``module.forward(...)``** — the in-place
    compile only hooks ``__call__``.

    Env var ``DDSSM_TORCH_COMPILE`` controls behavior (see module docstring):
      - ``0/false/no/off``: disabled (returns module unchanged)
      - ``soft/permissive``: try to compile, silently fall back to eager on failure
      - default / unknown: **strict** — any compile-call failure raises
    """
    mode = _compile_mode()
    if mode == "off":
        return module
    _configure_dynamo(mode)
    try:
        module.compile(dynamic=dynamic)
    except RuntimeError as e:
        if mode == "soft":
            log.warning("torch.compile disabled after failure: %s", e, exc_info=True)
            return module
        raise RuntimeError(
            f"torch.compile failed on {type(module).__name__} in strict mode; "
            f"set DDSSM_TORCH_COMPILE=soft to fall back to eager, "
            f"or =0 to disable compile."
        ) from e
    return module


def maybe_compile_fn(
    fn, *, dynamic: bool = False, compile_mode: str | None = None
):
    """Compile a callable / bound method (not an ``nn.Module``) when enabled.

    For per-step bodies like the encoder's ``_forward_with_stats`` that orchestrate
    several sub-modules and aren't themselves modules. Returns a ``torch.compile``
    wrapper (there's no ``state_dict`` to preserve, so the ``_orig_mod`` concern
    doesn't apply) or ``fn`` unchanged when disabled / soft-fallback / on failure.

    Args:
        dynamic: dynamic shape support (``dynamic=`` on ``torch.compile``).
        compile_mode: optional ``mode=`` for ``torch.compile``. ``None`` keeps
            the default (``"default"``); ``"reduce-overhead"`` activates CUDA
            graphs (static shapes only; captures inputs to fixed buffers).
    """
    mode = _compile_mode()
    if mode == "off":
        return fn
    _configure_dynamo(mode)
    kwargs: dict = {"dynamic": dynamic}
    if compile_mode is not None:
        kwargs["mode"] = compile_mode
    try:
        return torch.compile(fn, **kwargs)
    except RuntimeError as e:
        if mode == "soft":
            log.warning(
                "torch.compile(fn) disabled after failure: %s", e, exc_info=True
            )
            return fn
        raise RuntimeError(
            f"torch.compile failed on {getattr(fn, '__qualname__', repr(fn))} "
            f"in strict mode; set DDSSM_TORCH_COMPILE=soft to fall back to eager, "
            f"or =0 to disable compile."
        ) from e
