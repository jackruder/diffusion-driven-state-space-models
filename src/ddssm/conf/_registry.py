"""Explicit Hydra ConfigStore registration for DDSSM configs."""

from __future__ import annotations

from ._infra import store

_REGISTERED = False


def register_configs() -> None:
    """Import config modules and materialize their store entries once."""
    global _REGISTERED
    if _REGISTERED:
        return

    from .experiments import component, kdd, synthetic, variance_probe  # noqa: F401

    store.add_to_hydra_store(overwrite_ok=True)
    _REGISTERED = True


__all__ = ["register_configs"]
