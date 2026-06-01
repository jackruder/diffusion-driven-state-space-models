"""Conditional diffusion network building blocks for time series.

Provides the CSDI-style U-Net denoiser (:class:`CSDIUnet`), an MLP ablation
denoiser, the pluggable time/feature mixing layers used inside the residual
stacks, EDM diffusion-step conditioning (:class:`DiffusionEmbedding`), and the
encoder-side :class:`ContextProducer`.
"""

import math
from typing import final
from dataclasses import field, dataclass

import torch
import torch.nn as nn

# from mamba_ssm import Mamba2
import torch.nn.functional as F

from ddssm.nn.net_utils import (
    Conv1d_with_init,
    get_torch_trans,
)


@final
class SmallTimeConv(nn.Module):
    """Depthwise-separable conv mixing over a short time axis.

    Input/output are ``(B*d, C, L)`` and the shape is preserved. Acts as a
    residual no-op when ``L == 1``.
    """

    def __init__(self, channels: int, k: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (k // 2)
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=k,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=True,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="relu")
        nn.init.zeros_(self.dw.bias)
        nn.init.kaiming_normal_(self.pw.weight, nonlinearity="relu")
        nn.init.zeros_(self.pw.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.size(-1) <= 1:  # L==1 guard
            return y
        h = F.silu(self.dw(y))
        h = self.pw(h)
        return y + h


@final
class SmallTimeConvStack(nn.Module):
    """Two stacked :class:`SmallTimeConv` blocks with dilation 1 then 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.m1 = SmallTimeConv(channels, k=3, dilation=1)
        self.m2 = SmallTimeConv(channels, k=3, dilation=2)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.m2(self.m1(y))


class TimeLayer(nn.Module):
    """Abstract base for time-mixing layers."""

    def forward(
        self, x_flat: torch.Tensor, base_shape: tuple[int, int, int, int]
    ) -> torch.Tensor:
        raise NotImplementedError


# class MambaTimeLayer(TimeLayer):
#     def __init__(self, channels: int, mamba_config: MambaTimeConfig):
#         super().__init__()
#         self.layer = Mamba2(
#             d_model=channels,
#             d_state=mamba_config.state_dim,
#             d_conv=4,
#             conv_init=None,
#             expand=2,
#             headdim=mamba_config.headdim,
#         )
#
#     def forward(
#         self, x_flat: torch.Tensor, base_shape: tuple[int, int, int, int]
#     ) -> torch.Tensor:
#         B, C, d, L = base_shape
#         if L == 1:
#             return x_flat
#
#         # (B, C, d*L) -> (B, C, d, L)
#         y = x_flat.view(B, C, d, L)
#         # treat each feature independently: (B*d, L, C)
#         y = y.permute(0, 2, 3, 1).reshape(B * d, L, C)
#         y = self.layer(y)  # (B*d, L, C)
#         # back to (B, C, d, L) -> (B, C, d*L)
#         y = y.reshape(B, d, L, C).permute(0, 3, 1, 2).reshape(B, C, d * L)
#         return y


class ConvTimeLayer(TimeLayer):
    """Time mixer using a :class:`SmallTimeConvStack` over the ``L`` axis."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.layer = SmallTimeConvStack(channels)

    def forward(
        self, x_flat: torch.Tensor, base_shape: tuple[int, int, int, int]
    ) -> torch.Tensor:
        B, C, d, L = base_shape
        if L == 1:
            return x_flat

        y = x_flat.view(B, C, d, L)
        # SmallTimeConvStack expects (B*d, C, L)
        y = y.permute(0, 2, 1, 3).reshape(B * d, C, L)
        y = self.layer(y)  # (B*d, C, L)
        y = y.reshape(B, d, C, L).permute(0, 2, 1, 3).reshape(B, C, d * L)
        return y


class GRUTimeLayer(TimeLayer):
    """Time mixer running a GRU over the ``L`` axis (per feature)."""

    def __init__(self, channels: int, gru_layers: int = 1):
        super().__init__()
        self.layer = nn.GRU(
            input_size=channels,
            hidden_size=channels,
            num_layers=gru_layers,
            batch_first=True,
        )

    def forward(
        self, x_flat: torch.Tensor, base_shape: tuple[int, int, int, int]
    ) -> torch.Tensor:
        B, C, d, L = base_shape
        if L == 1:
            return x_flat

        # (B, C, d, L) -> (B, d, L, C) -> (B*d, L, C)
        y = x_flat.view(B, C, d, L).permute(0, 2, 3, 1).reshape(B * d, L, C)

        # GRU returns (output, h_n)
        # output: (B*d, L, C)
        y, _ = self.layer(y)

        # (B*d, L, C) -> (B, d, L, C) -> (B, C, d, L) -> (B, C, d*L)
        y = y.reshape(B, d, L, C).permute(0, 3, 1, 2).reshape(B, C, d * L)
        return y


class FeatureLayer(nn.Module):
    """Abstract base for feature-mixing layers."""

    def forward(
        self, x_flat: torch.Tensor, base_shape: tuple[int, int, int, int]
    ) -> torch.Tensor:
        raise NotImplementedError


class TransformerFeatureLayer(FeatureLayer):
    """Feature mixer running a Transformer encoder over the ``d`` axis."""

    def __init__(self, channels: int, nheads: int = 8, layers: int = 1):
        super().__init__()
        self.layer = get_torch_trans(
            heads=nheads, layers=layers, channels=channels
        )

    def forward(
        self, x_flat: torch.Tensor, base_shape: tuple[int, int, int, int]
    ) -> torch.Tensor:
        B, C, d, L = base_shape
        if d == 1:
            return x_flat

        # (B, C, d, L) -> (B, L, d, C) -> (B*L, d, C)
        # Sequence length is 'd', feature dimension is 'C'
        y = x_flat.view(B, C, d, L).permute(0, 3, 2, 1).reshape(B * L, d, C)

        # transformer expects (batch=B*L, seq_len=d, C)
        y = self.layer(y)

        # back to (B, C, d*L)
        y = y.reshape(B, L, d, C).permute(0, 3, 2, 1).reshape(B, C, d * L)
        return y


class ConvFeatureLayer(FeatureLayer):
    """Fallback feature mixing using convs (treating feature dim as sequence)."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.layer = SmallTimeConvStack(channels)

    def forward(
        self, x_flat: torch.Tensor, base_shape: tuple[int, int, int, int]
    ) -> torch.Tensor:
        B, C, d, L = base_shape
        if d == 1:
            return x_flat

        # Treat d as sequence length: (B*L, C, d)
        y = x_flat.view(B, C, d, L).permute(0, 3, 1, 2).reshape(B * L, C, d)
        y = self.layer(y)
        y = y.reshape(B, L, C, d).permute(0, 2, 3, 1).reshape(B, C, d * L)
        return y


class IdentityLayer(FeatureLayer, TimeLayer):
    """No-op mixer; usable as either a time or feature layer."""

    def forward(
        self, x_flat: torch.Tensor, base_shape: tuple[int, int, int, int]
    ) -> torch.Tensor:
        return x_flat


def build_time_layer(time_type: str, channels: int, kernel_size: int = 3, gru_layers: int = 1) -> TimeLayer:
    """Factory: create a TimeLayer from a type string and shared ``channels``."""
    if time_type == "conv":
        return ConvTimeLayer(channels, kernel_size=kernel_size)
    if time_type == "gru":
        return GRUTimeLayer(channels, gru_layers=gru_layers)
    if time_type == "identity":
        return IdentityLayer()
    raise ValueError(f"Unknown time_type: {time_type!r}. Choose from 'conv', 'gru', 'identity'.")


def build_feature_layer(feature_type: str, channels: int, nheads: int = 8, n_layers: int = 1) -> FeatureLayer:
    """Factory: create a FeatureLayer from a type string and shared ``channels``."""
    if feature_type == "transformer":
        return TransformerFeatureLayer(channels, nheads=nheads, layers=n_layers)
    if feature_type == "conv":
        return ConvFeatureLayer(channels)
    if feature_type == "identity":
        return IdentityLayer()
    raise ValueError(f"Unknown feature_type: {feature_type!r}. Choose from 'transformer', 'conv', 'identity'.")


@dataclass
class TimeMixerConfig:
    """Config for time-mixing layers used inside residual blocks."""

    type: str = "conv"
    kernel_size: int = 3
    gru_layers: int = 1


@dataclass
class FeatureMixerConfig:
    """Config for feature-mixing layers used inside residual blocks."""

    type: str = "transformer"
    nheads: int = 8
    n_layers: int = 1


def _default_context_feature_mixer_config() -> FeatureMixerConfig:
    return FeatureMixerConfig(n_layers=2)


@dataclass
class ResidualBlockConfig:
    """Config for ``ResidualBlock`` internals."""

    time: TimeMixerConfig = field(default_factory=TimeMixerConfig)
    feature: FeatureMixerConfig = field(
        default_factory=_default_context_feature_mixer_config
    )


@dataclass
class DiffResidualBlockConfig:
    """Config for ``DiffResidualBlock`` internals."""

    time: TimeMixerConfig = field(default_factory=TimeMixerConfig)
    feature: FeatureMixerConfig = field(default_factory=FeatureMixerConfig)


class DiffusionEmbedding(nn.Module):
    """Continuous EDM conditioning: embeds ``c_noise`` scalars into vectors.

    A sinusoidal feature map followed by a two-layer SiLU projection. The
    input ``c_noise`` is the EDM scalar per sample, e.g. ``(1/4)·log(σ̃)``.

    Args:
        embedding_dim: Size of the sinusoidal feature vector before
            projection (must be even).
        projection_dim: Output dimension after the two-layer projection
            (defaults to ``embedding_dim``).
        max_freq_log10: Frequency range; uses frequencies
            ``10**(linspace(0, 1, embedding_dim/2) * max_freq_log10)``.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        projection_dim: int | None = None,
        max_freq_log10: float = 4.0,
    ) -> None:
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even"
        if projection_dim is None:
            projection_dim = embedding_dim

        half = embedding_dim // 2
        freqs = 10.0 ** (torch.linspace(0.0, 1.0, half) * max_freq_log10)  # (half,)
        self.register_buffer("frequencies", freqs, persistent=False)

        self.projection1 = nn.Linear(2 * half, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        """Embed ``c_noise`` ``(B,)`` (or ``(B, 1)``) into ``(B, projection_dim)``."""
        if c_noise.dim() == 2 and c_noise.size(1) == 1:
            c_noise = c_noise.squeeze(1)
        assert c_noise.dim() == 1, "c_noise must be shape (B,) or (B,1)"

        args = c_noise.unsqueeze(1) * self.frequencies.unsqueeze(0)  # (B, half)
        feat = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half)

        x = F.silu(self.projection1(feat))
        x = F.silu(self.projection2(x))
        return x


@final
class DiffResidualBlock(nn.Module):
    """CSDI residual block with diffusion-step and side-info conditioning.

    Composes a time mixer over ``L`` (conv / gru / identity), a feature mixer
    over ``d`` (transformer / conv / identity), and a gated conv-style update
    returning a residual + skip pair.

    Shapes:
        x: ``(B, C, d, L)``
        side_info: ``(B, side_dim, d, L)``
        diffusion_emb: ``(B, diffusion_embedding_dim)``
    """

    def __init__(
        self,
        side_dim: int,
        channels: int,  # C (kept small; used as d_model)
        diffusion_embedding_dim: int,
        time_layer: TimeLayer,
        feature_layer: FeatureLayer,
    ) -> None:
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.feature_layer = feature_layer
        self.time_layer = time_layer

    def forward(
        self, x: torch.Tensor, side_info: torch.Tensor, diffusion_emb: torch.Tensor
    ):
        """Apply the residual block.

        Args:
            x: ``(B, C, d, L)`` block input.
            side_info: ``(B, side_dim, d, L)`` conditioning tensor.
            diffusion_emb: ``(B, diffusion_embedding_dim)`` step embedding.

        Returns:
            Tuple ``(out, skip)``, each ``(B, C, d, L)``.
        """
        B, C, d, L = x.shape
        base_shape = x.shape
        x_flat = x.view(B, C, d * L)

        # diffusion-step conditioning
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,C,1)
        y = x_flat + diffusion_emb

        # time then feature mixing
        y = self.time_layer(y, base_shape)
        y = self.feature_layer(y, base_shape)

        # mid projection
        y = self.mid_projection(y)  # (B, 2C, d*L)

        # side-info conditioning
        if side_info is not None and side_info.size(1) > 0:
            _, cond_dim, _, _ = side_info.shape
            side_info_flat = side_info.reshape(B, cond_dim, d * L)
            side_info_flat = self.cond_projection(side_info_flat)  # (B, 2C, d*L)
            y = y + side_info_flat

        # gated activation
        gate, filt = torch.chunk(y, 2, dim=1)  # each (B, C, d*L)
        y = torch.sigmoid(gate) * torch.tanh(filt)  # (B, C, d*L)
        y = self.output_projection(y)  # (B, 2C, d*L)

        # residual + skip
        residual, skip = torch.chunk(y, 2, dim=1)  # each (B, C, d*L)
        x = x_flat.view(base_shape)
        residual = residual.view(base_shape)
        skip = skip.view(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


@final
class CSDIUnet(nn.Module):
    """U-Net style denoising network for conditional diffusion on time series."""

    def __init__(
        self,
        output_len: int,
        diffusion_steps: int,
        latent_dim: int,
        latent_history_len: int,  # h
        side_dim: int,
        channels: int = 64,
        n_layers: int = 4,
        embedding_dim: int = 128,
        projection_dim: int | None = None,
        residual_block: DiffResidualBlockConfig | None = None,
        zero_init_output: bool = True,
        cond_mask_channel: int | None = None,
    ) -> None:
        super().__init__()
        if residual_block is None:
            residual_block = DiffResidualBlockConfig()
        self.output_len = output_len
        self.latent_dim = latent_dim
        self.latent_history_len = latent_history_len

        self.channels = channels
        self.n_layers = n_layers

        self.side_dim = side_dim

        # Absolute channel index of the *conditioning* mask within ``side_info``
        # (1 on clean-history slots, 0 on the noised target slot). The
        # ``get_side_info`` layout is ``[time(+cov), feat, cond_mask,
        # padding_mask]``, so cond_mask sits at ``side_dim - 2`` and the
        # padding mask at ``side_dim - 1``. We index it *explicitly* rather than
        # relatively: a prior bug read ``side_info[:, -1]`` (the padding mask),
        # which silently routed the whole window into the noisy stream. The
        # owning transition passes the exact index; the ``-2`` fallback matches
        # the both-masks-present layout. See model-v2.org § "Padding mask in the
        # diffusion side-info tensor".
        if cond_mask_channel is None:
            cond_mask_channel = side_dim - 2
        if not (0 <= cond_mask_channel < side_dim):
            raise ValueError(
                f"cond_mask_channel={cond_mask_channel} out of range "
                f"[0, side_dim={side_dim})"
            )
        self.cond_mask_channel = int(cond_mask_channel)

        self.diffusion_embedding_dim = embedding_dim
        self.diffusion_projection_dim = projection_dim or embedding_dim

        self.diffusion_embedding = DiffusionEmbedding(
            embedding_dim=self.diffusion_embedding_dim,
            projection_dim=self.diffusion_projection_dim,
        )

        self.input_projection = Conv1d_with_init(2, self.channels, 1)

        self.residual_layers = nn.ModuleList([
                DiffResidualBlock(
                    side_dim=self.side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=self.diffusion_projection_dim,
                    time_layer=build_time_layer(
                        residual_block.time.type,
                        channels,
                        kernel_size=residual_block.time.kernel_size,
                        gru_layers=residual_block.time.gru_layers,
                    ),
                    feature_layer=build_feature_layer(
                        residual_block.feature.type,
                        channels,
                        nheads=residual_block.feature.nheads,
                        n_layers=residual_block.feature.n_layers,
                    ),
                )
                for _ in range(self.n_layers)
        ])
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        # Conv1d_with_init already zero-initialises bias; the standard
        # EDM convention also zeros the final-layer weights so D_ψ ≈
        # c_skip · z̃_t at the start of training.  Toggle via flag for
        # ablations / explicit control under model-v2 diffusion.
        if zero_init_output:
            nn.init.zeros_(self.output_projection2.weight)
            if self.output_projection2.bias is not None:
                nn.init.zeros_(self.output_projection2.bias)

    def _split_history_noise(
        self, x: torch.Tensor, side_info: torch.Tensor
    ) -> torch.Tensor:
        """Split ``x`` into the clean-history and noised streams.

        Uses the *conditioning* mask (``cond_mask == 1`` on the clean-history
        slots, ``0`` on the noised target slot), read by its explicit absolute
        channel index ``self.cond_mask_channel``. A prior bug read the last
        channel — the *padding* mask — which on the t≥2 training path is all
        zeros, silently routing the whole window into the noisy stream and
        zeroing the clean-history stream.

        Args:
            x: ``(B, d, L)`` window (history slots + noised target slot).
            side_info: ``(B, side_dim, d, L)`` side-info tensor.

        Returns:
            ``(B, 2, d, L)`` stacked ``[x_noisy, x_hist_clean]``.
        """
        c = self.cond_mask_channel
        mask = side_info[:, c : c + 1, :, :]  # (B, 1, d, L)
        x = x.unsqueeze(1)  # (B, 1, d, L)
        x_noisy = x * (1.0 - mask)  # zero out clean history
        x_hist_clean = x * mask  # zero out the noised target
        return torch.cat([x_noisy, x_hist_clean], dim=1)  # (B, 2, d, L)

    def forward(
        self, x: torch.Tensor, side_info: torch.Tensor, diffusion_step: torch.Tensor
    ):
        """Forward pass of the U-Net denoiser.

        Args:
            x: ``(B, d, L)`` input. The last ``output_len`` slots hold the
                noised target; the first ``latent_history_len`` slots are the
                clean history (``L = latent_history_len + output_len``).
            side_info: ``(B, side_dim, d, L)`` side-information tensor.
            diffusion_step: ``(B,)`` EDM ``c_noise`` conditioning scalar.

        Returns:
            ``(B, d, output_len)`` EDM denoiser output ``F_ψ`` for the target
            slots.
        """
        # L = J + 1
        B, d, L = x.shape
        P = self.output_len

        x_cat = self._split_history_noise(x, side_info)  # (B, 2, d, L)

        x_cat = x_cat.view(B, 2, d * L)
        x = self.input_projection(x_cat)
        x = torch.relu(x)
        x = x.view(B, self.channels, d, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.view(B, self.channels, d * L)
        x = self.output_projection1(x)
        x = torch.relu(x)
        x = self.output_projection2(x)
        x = x.view(B, d, L)  # (B, d, L)
        x = x[:, :, -P:]  # (B, d, P)
        return x


@final
class MLPCSDIUnet(nn.Module):
    """Simple MLP denoiser used for architectural ablations/testing."""

    def __init__(
        self,
        output_len: int,
        diffusion_steps: int,
        latent_dim: int,
        latent_history_len: int,
        side_dim: int,
        channels: int = 64,
        n_layers: int = 2,
        embedding_dim: int = 128,
        projection_dim: int | None = None,
        residual_block: DiffResidualBlockConfig | None = None,
        zero_init_output: bool = False,
        cond_mask_channel: int | None = None,
    ) -> None:
        super().__init__()
        # Kept for constructor compatibility with ``CSDIUnet``. The MLP flattens
        # every side-info channel into its input, so it has no separate
        # history/noise stream and ignores ``cond_mask_channel``.
        _ = (
            diffusion_steps, embedding_dim, projection_dim, residual_block,
            cond_mask_channel,
        )

        self.output_len = int(output_len)
        self.latent_dim = int(latent_dim)
        self.latent_history_len = int(latent_history_len)
        self.side_dim = int(side_dim)

        L = self.latent_history_len + self.output_len
        in_dim = self.latent_dim * L + self.side_dim * self.latent_dim * L + 1
        out_dim = self.latent_dim * self.output_len
        hidden = max(int(channels), 16)
        depth = max(int(n_layers), 1)

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        final = nn.Linear(hidden, out_dim)
        if zero_init_output:
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, side_info: torch.Tensor, diffusion_step: torch.Tensor
    ) -> torch.Tensor:
        B, d, L = x.shape
        assert d == self.latent_dim
        assert self.latent_history_len + self.output_len == L

        x_flat = x.reshape(B, -1)
        side_flat = side_info.reshape(B, -1)
        step = diffusion_step.reshape(B, 1).to(dtype=x.dtype)
        inp = torch.cat([x_flat, side_flat, step], dim=1)
        out = self.mlp(inp)
        return out.reshape(B, self.latent_dim, self.output_len)


@final
class ResidualBlock(nn.Module):
    """Residual block variant for the encoder, matching CSDI side-info integration.

    Shapes:
      x: (B, C, d, L)
      side_info: (B, side_dim, d, L)
    """

    def __init__(
        self,
        side_dim: int,
        channels: int,  # C (kept small; used as d_model)
        time_layer: TimeLayer,
        feature_layer: FeatureLayer,
    ) -> None:
        super().__init__()
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.feature_layer = feature_layer
        self.time_layer = time_layer

    def forward(
        self,
        x: torch.Tensor,  # (B, C, d, L)
        side_info: torch.Tensor,  # (B, side_dim, d, L)
    ):
        B, C, d, L = x.shape
        base_shape = x.shape
        x_flat = x.view(B, C, d * L)

        # time then feature mixing
        y = self.time_layer(x_flat, base_shape)
        y = self.feature_layer(y, base_shape)

        # mid projection
        y = self.mid_projection(y)  # (B, 2C, d*L)

        # side-info conditioning
        if side_info is not None and side_info.size(1) > 0:
            _, cond_dim, _, _ = side_info.shape
            side_info_flat = side_info.reshape(B, cond_dim, d * L)
            side_info_flat = self.cond_projection(side_info_flat)  # (B, 2C, d*L)
            y = y + side_info_flat

        # gated activation
        gate, filt = torch.chunk(y, 2, dim=1)  # each (B, C, d*L)
        y = torch.sigmoid(gate) * torch.tanh(filt)  # (B, C, d*L)
        y = self.output_projection(y)  # (B, 2C, d*L)

        # residual + skip
        residual, skip = torch.chunk(y, 2, dim=1)  # each (B, C, d*L)
        x = x_flat.view(base_shape)
        residual = residual.view(base_shape)
        skip = skip.view(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class ContextProducer(nn.Module):
    """Produce a context token feeding a residual block stack.

    Builds a tensor combining the per-step side info with the latent history
    ``z_{t-j:t-1}`` so the residual-block time/feature mixers can summarise the
    history into the parameters for ``z_t``. The token is returned flattened to
    ``(B, C * H_seq)`` for a downstream head to split.
    """

    def __init__(
        self,
        channels: int,  # C
        num_layers: int,
        combined_dim: int,  # H_seq
        mask_tot_dim: int,  # H_mask
        emb_time_dim: int,  # H_time
        combined_len: int,  # L
        residual_block: ResidualBlockConfig | None = None,
        skip_mask: bool = False,
        static_emb_dim: int = 0,
    ) -> None:
        super().__init__()
        if residual_block is None:
            residual_block = ResidualBlockConfig()
        self.channels = channels  # C
        self.combined_dim = combined_dim
        self.mask_tot_dim = mask_tot_dim
        self.emb_time_dim = emb_time_dim
        self.combined_len = combined_len
        self.static_emb_dim = static_emb_dim
        self.side_dim = mask_tot_dim + emb_time_dim + static_emb_dim

        self.tot_dim = combined_dim + mask_tot_dim + emb_time_dim

        self.num_layers = num_layers
        self.emb_time_dim = emb_time_dim
        self.skip_mask = skip_mask
        self.eps = 1e-8

        if skip_mask:
            assert mask_tot_dim == 0, (
                "If skip_mask is True, mask_tot_dim must be 0 (no mask input)."
            )

        # mirror CSDIUnet: project a 1-channel input over (d, L_enc)
        # to C channels
        self.input_projection = nn.Conv1d(
            in_channels=1,
            out_channels=self.channels,
            kernel_size=1,
        )
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="relu")
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)

        blocks = []
        for _ in range(self.num_layers):
            blocks.append(
                ResidualBlock(
                    side_dim=self.side_dim,
                    channels=self.channels,
                    time_layer=build_time_layer(
                        residual_block.time.type,
                        channels,
                        kernel_size=residual_block.time.kernel_size,
                        gru_layers=residual_block.time.gru_layers,
                    ),
                    feature_layer=build_feature_layer(
                        residual_block.feature.type,
                        channels,
                        nheads=residual_block.feature.nheads,
                        n_layers=residual_block.feature.n_layers,
                    ),
                )
            )
        self.context_blocks = nn.ModuleList(blocks)

        # 1D conv over context time dimension (j+1) with groups
        self.context_conv = nn.Conv1d(
            in_channels=self.channels * self.combined_dim,
            out_channels=self.channels * self.combined_dim,
            kernel_size=self.combined_len,
            groups=self.channels * self.combined_dim,
            bias=True,
        )

    # ---- main calls ----
    def forward(
        self,
        *,
        combined: torch.Tensor,  # (B, H_seq, L)
        mask_embedded: torch.Tensor | None,  # (B, H_mask, L)
        hist_time_emb: torch.Tensor,  # (B, H_time, L)
        static_embedded: torch.Tensor | None = None,  # (B, E_static, H_seq)
    ) -> torch.Tensor:
        """Return the ``(B, C * H_seq)`` context token for the window."""
        device = combined.device

        # z_prev: (B, d, j)
        B, H_seq, L = combined.shape

        Bt, H_time, Lt = hist_time_emb.shape

        if self.skip_mask:
            if mask_embedded is None:
                mask_embedded = torch.zeros(
                    (B, 0, L), device=device, dtype=combined.dtype
                )
            Bm, H_mask, Lm = mask_embedded.shape
            assert H_mask == 0
        else:
            assert mask_embedded is not None
            Bm, H_mask, Lm = mask_embedded.shape
        assert B == Bm == Bt
        assert L == Lm == Lt
        assert self.combined_len == L

        assert H_seq == self.combined_dim
        assert H_mask == self.mask_tot_dim
        assert H_time == self.emb_time_dim

        #  Base side info (Time and Mask) - varies over L, shared over H_seq
        side_components_L = [hist_time_emb, mask_embedded]
        side_info_L = torch.cat(side_components_L, dim=1)  # (B, H_time + H_mask, L)

        # Expand across the spatial/feature dimension (d = H_seq)
        side_info = side_info_L.unsqueeze(2).expand(
            -1, -1, H_seq, -1
        )  # (B, H_time + H_mask, H_seq, L)

        # 2. Static side info - varies over H_seq, shared over L
        if self.static_emb_dim > 0 and static_embedded is not None:
            # static_embedded is (B, E_static, H_seq)
            # Expand across the time dimension (L)
            static_expanded = static_embedded.unsqueeze(-1).expand(
                -1, -1, -1, L
            )  # (B, E_static, H_seq, L)

            side_info = torch.cat(
                [side_info, static_expanded], dim=1
            )  # (B, side_dim, H_seq, L)

        # ---- project main sequence to C channels ----
        combined_flat = combined.reshape(B, 1, H_seq * L)  # (B, 1, d*L)
        x = self.input_projection(combined_flat)  # (B, C, d*L)
        x = torch.relu(x)

        # reshape back to (B, C, d, L) for residual blocks
        x = x.reshape(B, self.channels, H_seq, L)

        # ---- run through residual block stack ----
        skips = []
        for blk in self.context_blocks:
            x, skip = blk(x, side_info)
            skips.append(skip)

        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.context_blocks))
        # x: (B, C, d, L)

        # ---- context conv over time dim ----
        x = x.view(B, self.channels * H_seq, L)  # (B, C*H_seq, L)
        x = self.context_conv(x)  # (B, C*H_seq, 1)
        x = x.squeeze(-1)  # (B, C*H_seq)

        return x


@final
class MLPContextProducer(nn.Module):
    """Simple MLP context producer used for architectural ablations/testing."""

    def __init__(
        self,
        channels: int,
        num_layers: int,
        combined_dim: int,
        mask_tot_dim: int,
        emb_time_dim: int,
        combined_len: int,
        residual_block: ResidualBlockConfig | None = None,
        skip_mask: bool = False,
        static_emb_dim: int = 0,
    ) -> None:
        super().__init__()
        # Kept for constructor compatibility with ``ContextProducer``.
        _ = residual_block

        self.channels = int(channels)
        self.num_layers = int(num_layers)
        self.combined_dim = int(combined_dim)
        self.mask_tot_dim = int(mask_tot_dim)
        self.emb_time_dim = int(emb_time_dim)
        self.combined_len = int(combined_len)
        self.skip_mask = bool(skip_mask)
        self.static_emb_dim = int(static_emb_dim)

        in_dim = (
            self.combined_dim * self.combined_len
            + self.mask_tot_dim * self.combined_len
            + self.emb_time_dim * self.combined_len
            + self.static_emb_dim * self.combined_dim
        )
        out_dim = self.channels * self.combined_dim
        hidden = max(out_dim, 16)
        depth = max(self.num_layers, 1)

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        *,
        combined: torch.Tensor,
        mask_embedded: torch.Tensor | None,
        hist_time_emb: torch.Tensor,
        static_embedded: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H_seq, L = combined.shape
        assert H_seq == self.combined_dim
        assert self.combined_len == L

        if self.skip_mask:
            if mask_embedded is None:
                mask_embedded = torch.zeros(
                    B, 0, L, device=combined.device, dtype=combined.dtype
                )
            assert mask_embedded.shape == (B, 0, L)
        else:
            assert mask_embedded is not None
            assert mask_embedded.shape == (B, self.mask_tot_dim, L)

        assert hist_time_emb.shape == (B, self.emb_time_dim, L)

        if self.static_emb_dim > 0:
            if static_embedded is None:
                static_embedded = torch.zeros(
                    B,
                    self.static_emb_dim,
                    self.combined_dim,
                    device=combined.device,
                    dtype=combined.dtype,
                )
            assert static_embedded.shape == (B, self.static_emb_dim, self.combined_dim)
            static_flat = static_embedded.reshape(B, -1)
        else:
            static_flat = torch.zeros(B, 0, device=combined.device, dtype=combined.dtype)

        inp = torch.cat(
            [
                combined.reshape(B, -1),
                mask_embedded.reshape(B, -1),
                hist_time_emb.reshape(B, -1),
                static_flat,
            ],
            dim=1,
        )
        return self.mlp(inp)
