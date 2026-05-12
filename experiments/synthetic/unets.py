"""Score-net architectures for the synthetic-data diffusion transitions.

The 1D-synthetic Diffusion variant uses the *tiny* MLP score-net so
its total parameter count stays close to the Gaussian variant
(otherwise the CSDI U-Net would dominate the model at ~290k params,
making Gauss-vs-Diff comparisons unfair). The full CSDI U-Net is
kept registered for higher-D shapes (robot 2D, KDD) and for explicit
opt-in.
"""

from __future__ import annotations

from ddssm.builders import MLPUnet, Unet

from conf.registry import unet_store


# Full CSDI U-Net at the default channel/layer count — used by Robot2D
# and KDD diffusion transitions where the extra capacity helps.
CSDI = Unet()

# Default MLP ablation — drop-in replacement of the same interface.
MLP = MLPUnet(channels=64, n_layers=3)

# Tiny MLP score-net — used by the 1D synthetic Diff transition so
# its size matches Gauss (transition: ~10k vs 13k). Also used by the
# variance probes (where the diffusion net is just the thing being
# measured, not the thing that needs capacity).
MLPTiny = MLPUnet(channels=32, n_layers=2, embedding_dim=32)

unet_store(CSDI, name="csdi")
unet_store(MLP, name="mlp")
unet_store(MLPTiny, name="mlp_tiny")
