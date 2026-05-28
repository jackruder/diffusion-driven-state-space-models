"""Tests for ``maybe_compile`` — the in-place torch.compile helper.

The contract that matters for checkpoint portability: compiling a submodule must
NOT change its ``state_dict`` keys (no ``_orig_mod.`` prefix), so a checkpoint
saved with compile active loads into an eager model under ``strict=True`` and
vice versa.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ddssm.torch_compile import maybe_compile


class _Producer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, combined):
        return self.lin(combined)


class _Model(nn.Module):
    """Mirrors the codebase pattern: compile a child in ``__init__``."""

    def __init__(self):
        super().__init__()
        self.producer = _Producer()
        self.producer = maybe_compile(self.producer, dynamic=True)

    def forward(self, x):
        return self.producer(x)  # __call__, so the compiled path fires


def _orig_mod_keys(module: nn.Module) -> list[str]:
    return [k for k in module.state_dict() if "_orig_mod" in k]


def test_compile_preserves_state_dict_keys(monkeypatch):
    monkeypatch.setenv("DDSSM_TORCH_COMPILE", "1")
    m = _Model()
    assert _orig_mod_keys(m) == []
    # In-place: identity and type are unchanged, compile was applied.
    assert isinstance(m.producer, _Producer)
    assert m.producer._compiled_call_impl is not None


def test_disabled_leaves_module_untouched(monkeypatch):
    monkeypatch.setenv("DDSSM_TORCH_COMPILE", "0")
    m = _Model()
    assert _orig_mod_keys(m) == []
    assert getattr(m.producer, "_compiled_call_impl", None) is None


def test_save_load_roundtrip_across_compile_toggle(tmp_path, monkeypatch):
    """A compiled-model checkpoint loads strictly into an eager model and back."""
    monkeypatch.setenv("DDSSM_TORCH_COMPILE", "1")
    compiled = _Model()
    path = str(tmp_path / "ckpt.pth")
    torch.save(compiled.state_dict(), path)

    monkeypatch.setenv("DDSSM_TORCH_COMPILE", "0")
    eager = _Model()
    # strict=True is the assertion: keys must match exactly across the toggle.
    eager.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.allclose(eager.producer.lin.weight, compiled.producer.lin.weight)

    # ...and the reverse direction (eager-saved → compiled-loaded).
    monkeypatch.setenv("DDSSM_TORCH_COMPILE", "1")
    recompiled = _Model()
    recompiled.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.allclose(recompiled.producer.lin.weight, eager.producer.lin.weight)


def test_forward_via_call_runs(monkeypatch):
    """Forward through ``__call__`` returns the right shape (eager fallback ok)."""
    monkeypatch.setenv("DDSSM_TORCH_COMPILE", "1")
    m = _Model()
    out = m(torch.randn(2, 4))
    assert out.shape == (2, 3)
