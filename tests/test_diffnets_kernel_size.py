"""Tests that the kernel_size knob in diffnets actually changes conv modules.

Verifies:
- ConvTimeLayer with kernel_size != 3 produces convolutions with that kernel_size.
- ConvFeatureLayer with kernel_size != 3 produces convolutions with that kernel_size.
- Default kernel_size=3 still produces the same architecture as before.
"""

from __future__ import annotations

import torch.nn as nn

from ddssm.nn.diffnets import (
    ConvFeatureLayer,
    ConvTimeLayer,
    SmallTimeConvStack,
    build_feature_layer,
    build_time_layer,
)


def _conv_kernel_sizes(module: nn.Module) -> list[int]:
    """Return sorted list of kernel_sizes from all Conv1d layers in module."""
    return sorted(
        m.kernel_size[0]
        for m in module.modules()
        if isinstance(m, nn.Conv1d) and m.kernel_size[0] > 1
    )


def test_small_time_conv_stack_default_kernel_size() -> None:
    """SmallTimeConvStack default k=3 produces 3x1 depthwise convs."""
    stack = SmallTimeConvStack(channels=8)
    sizes = _conv_kernel_sizes(stack)
    assert all(s == 3 for s in sizes), f"expected all 3, got {sizes}"


def test_small_time_conv_stack_custom_kernel_size() -> None:
    """SmallTimeConvStack k=5 propagates to both depthwise conv layers."""
    stack = SmallTimeConvStack(channels=8, k=5)
    sizes = _conv_kernel_sizes(stack)
    assert all(s == 5 for s in sizes), f"expected all 5, got {sizes}"


def test_conv_time_layer_default_kernel_size() -> None:
    """ConvTimeLayer default kernel_size=3 produces 3x1 depthwise convs."""
    layer = ConvTimeLayer(channels=8)
    sizes = _conv_kernel_sizes(layer)
    assert all(s == 3 for s in sizes), f"expected all 3, got {sizes}"


def test_conv_time_layer_custom_kernel_size() -> None:
    """ConvTimeLayer kernel_size=7 wires through to the underlying convolutions."""
    layer = ConvTimeLayer(channels=8, kernel_size=7)
    sizes = _conv_kernel_sizes(layer)
    assert all(s == 7 for s in sizes), f"expected all 7, got {sizes}"


def test_conv_feature_layer_default_kernel_size() -> None:
    """ConvFeatureLayer default kernel_size=3 produces 3x1 depthwise convs."""
    layer = ConvFeatureLayer(channels=8)
    sizes = _conv_kernel_sizes(layer)
    assert all(s == 3 for s in sizes), f"expected all 3, got {sizes}"


def test_conv_feature_layer_custom_kernel_size() -> None:
    """ConvFeatureLayer kernel_size=5 wires through to the underlying convolutions."""
    layer = ConvFeatureLayer(channels=8, kernel_size=5)
    sizes = _conv_kernel_sizes(layer)
    assert all(s == 5 for s in sizes), f"expected all 5, got {sizes}"


def test_build_time_layer_conv_custom_kernel_size() -> None:
    """build_time_layer('conv', ..., kernel_size=9) propagates to conv layers."""
    layer = build_time_layer("conv", channels=8, kernel_size=9)
    sizes = _conv_kernel_sizes(layer)
    assert all(s == 9 for s in sizes), f"expected all 9, got {sizes}"


def test_build_feature_layer_conv_custom_kernel_size() -> None:
    """build_feature_layer('conv', ..., kernel_size=5) propagates to conv layers."""
    layer = build_feature_layer("conv", channels=8, kernel_size=5)
    sizes = _conv_kernel_sizes(layer)
    assert all(s == 5 for s in sizes), f"expected all 5, got {sizes}"
