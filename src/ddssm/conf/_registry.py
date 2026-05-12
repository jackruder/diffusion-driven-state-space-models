"""Explicit Hydra ConfigStore registration for DDSSM configs."""

from __future__ import annotations

from threading import Lock

from ._infra import store

_REGISTERED = False
_REGISTER_LOCK = Lock()


def register_configs() -> None:
    """Import config modules and materialize their store entries once."""
    global _REGISTERED
    with _REGISTER_LOCK:
        if _REGISTERED:
            return

        from .experiments import kdd, component, synthetic, variance_probe  # noqa: F401

        store.add_to_hydra_store(overwrite_ok=True)
        _REGISTERED = True


__all__ = ["register_configs"]
