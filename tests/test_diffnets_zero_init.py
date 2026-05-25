"""Unit tests for the ``zero_init_output`` flag on the diffusion U-Nets.

Per ``init-experiment.org`` § Implementation precursors, the stage-2 score
network must start with its final layer zero-initialised so that
``D_ψ ≈ c_skip · z̃_t`` at the beginning of training.
"""

from __future__ import annotations

import torch

from ddssm.diffnets import CSDIUnet, MLPCSDIUnet

COMMON_KW = dict(
    output_len=1,
    diffusion_steps=50,
    latent_dim=4,
    latent_history_len=1,
    side_dim=3,
)


def test_csdi_unet_zero_init_default_true() -> None:
    """The default has been True historically (V2 already zero-inits)."""
    net = CSDIUnet(**COMMON_KW, channels=16, n_layers=2)
    assert torch.equal(
        net.output_projection2.weight, torch.zeros_like(net.output_projection2.weight)
    )
    if net.output_projection2.bias is not None:
        assert torch.equal(
            net.output_projection2.bias, torch.zeros_like(net.output_projection2.bias)
        )


def test_csdi_unet_zero_init_explicit_true() -> None:
    """Explicit ``zero_init_output=True`` zeros final layer weights and bias."""
    net = CSDIUnet(**COMMON_KW, channels=16, n_layers=2, zero_init_output=True)
    w = net.output_projection2.weight.detach()
    assert torch.equal(w, torch.zeros_like(w))
    if net.output_projection2.bias is not None:
        b = net.output_projection2.bias.detach()
        assert torch.equal(b, torch.zeros_like(b))


def test_csdi_unet_zero_init_false_keeps_kaiming() -> None:
    """With the flag off, final-layer weights retain Kaiming init."""
    net = CSDIUnet(**COMMON_KW, channels=16, n_layers=2, zero_init_output=False)
    assert bool(net.output_projection2.weight.detach().any())


def test_mlp_unet_zero_init_default_false() -> None:
    """MLPCSDIUnet default keeps standard PyTorch init."""
    net = MLPCSDIUnet(**COMMON_KW, channels=16, n_layers=2)
    final_layer = net.mlp[-1]
    assert bool(final_layer.weight.detach().any())


def test_mlp_unet_zero_init_true() -> None:
    """Explicit ``zero_init_output=True`` zeros the MLP's final layer."""
    net = MLPCSDIUnet(**COMMON_KW, channels=16, n_layers=2, zero_init_output=True)
    final_layer = net.mlp[-1]
    w = final_layer.weight.detach()
    assert torch.equal(w, torch.zeros_like(w))
    if final_layer.bias is not None:
        b = final_layer.bias.detach()
        assert torch.equal(b, torch.zeros_like(b))


def test_mlp_unet_zero_init_output_makes_initial_forward_zero() -> None:
    """With zero-init, the very first forward of an MLP yields a zero tensor."""
    net = MLPCSDIUnet(**COMMON_KW, channels=16, n_layers=2, zero_init_output=True)
    B = 2
    L = COMMON_KW["latent_history_len"] + COMMON_KW["output_len"]
    x = torch.randn(B, COMMON_KW["latent_dim"], L)
    side = torch.randn(B, COMMON_KW["side_dim"], COMMON_KW["latent_dim"], L)
    step = torch.zeros(B, dtype=torch.long)
    out = net(x, side, step)
    assert out.shape == (B, COMMON_KW["latent_dim"], COMMON_KW["output_len"])
    # MLP without skip connection ⇒ output is exactly zero on first pass.
    assert torch.equal(out, torch.zeros_like(out))
