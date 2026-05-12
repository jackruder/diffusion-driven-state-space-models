"""Score-net architectures for the synthetic-data diffusion transitions."""

from __future__ import annotations

from ddssm.builders import MLPUnet, Unet

from conf.registry import unet_store


# CSDI U-Net at the default channel/layer count — used by the
# small_diff / robot2d_diff transitions.
CSDI = Unet()

# MLP ablation — drop-in replacement of the same shape, used for
# sweeps via override(..., {"model.transition.unet": MLP}).
MLP = MLPUnet(channels=64, n_layers=3)

unet_store(CSDI, name="csdi")
unet_store(MLP, name="mlp")
