"""Shared neural-network utilities: time embeddings, side-info construction, and common layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(pos: torch.Tensor, d_model: int = 128, device: torch.device = "cpu"):
    """Sinusoidal (Vaswani) embeddings for integer or real timestamps.

    Reserved for the future irregular-timestep / relative-time regime
    (``use_time_embedding=True``). When ``d_model == 0`` the call short-circuits
    to an empty ``(B, T, 0)`` tensor so the regular-timestep path can call
    through unconditionally.

    Args:
        pos: ``(B, T)`` timestamps (cast to float).
        d_model: Embedding dimension. ``0`` disables the embedding.
        device: Device to build the embedding on.

    Returns:
        ``(B, T, d_model)`` sinusoidal time embeddings.
    """
    pos = pos.to(device).float()  # ensure float for sin/cos
    B, T = pos.shape
    if d_model == 0:
        return torch.empty(B, T, 0, device=device, dtype=pos.dtype)
    pe = torch.zeros(B, T, d_model, device=device, dtype=pos.dtype)
    # inv_freq[k] = 1 / (10000^{2k/d_model})
    k = torch.arange(0, d_model, 2, device=device, dtype=pos.dtype)
    inv_freq = 10000.0 ** (-k / d_model)
    ang = pos.unsqueeze(-1) * inv_freq  # (B,T,d_model/2)
    pe[..., 0::2] = torch.sin(ang)
    pe[..., 1::2] = torch.cos(ang)
    return pe.contiguous()


def get_side_info(
    data_dim: int,
    time_embed: torch.Tensor,  # (B, T, E_t)
    embed_layer: nn.Embedding,  # nn.Embedding(D, E_f)
    cond_mask: torch.Tensor | None = None,  # (B, D, T) optional
    device: str = "cpu",
    padding_mask: torch.Tensor | None = None,  # (B, T) optional
):
    """Build covariate information tensors.

    For a batch of size B and sequence length T, for data of dimension D,
    The side information combines:
        - given time embeddings per timestep (of dimension E_t)
        - learned feature embeddings per data dimension (of dimension E_f)
        - optional conditioning mask (1 channel)
        - optional padding mask (1 channel)

    ``padding_mask`` is per-slot (shape ``(B, T)``) and flags slot identity
    (e.g.\\ "padded auxiliary z_0" vs.\\ "real previous latent" in the
    model-v2 VHP-via-diffusion construction).  It is broadcast across the
    data dimension when appended as a side-info channel.  Per
    ``init-experiment.org`` § Implementation precursors and
    ``model-v2.org`` § Padding mask in the diffusion side-info tensor.

    TODO : support additional covariates,
      - static feature covariates (per D)
      - dynamic time-varying covariates (per T)
      -

    Args:
        data_dim: int, D
        time_embed: (B, T, E_t) time embeddings per timestep
        embed_layer: nn.Embedding(D, E_f) feature embedding layer
        cond_mask: optional (B, D, T) conditioning mask, for missing data
        device: str, device to put tensors on
        padding_mask: optional (B, T) per-slot binary mask flagging
            padded auxiliary slots.  Broadcast across the D axis.


    Returns:
        side_info: (B, C_side, D, T), where
            C_side = E_t + E_f (+1 if cond_mask) (+1 if padding_mask)
    """
    B, T, E_t = time_embed.shape
    D = data_dim
    E_f = int(embed_layer.embedding_dim)

    time_embed = time_embed.to(device)

    # Skip time and feature channels independently when their dim is 0. Branch
    # on the Python ints (compile-time constants) rather than letting
    # ``(…, 0)`` tensors cascade — cuDNN's conv backward errors when a
    # ``(D, 0)`` ``embed_layer.weight`` parameter is wired into the autograd
    # graph through the downstream ``cond_projection`` Conv1d, even though the
    # forward pass succeeds.
    channels: list[torch.Tensor] = []
    if E_t > 0:
        time_b = time_embed.unsqueeze(2).expand(B, T, D, E_t)
        channels.append(time_b)
    if E_f > 0:
        feats = torch.arange(D, device=device)
        feat_embed = embed_layer(feats)  # (D, E_f)
        feat_b = feat_embed.unsqueeze(0).unsqueeze(0).expand(B, T, D, -1)
        channels.append(feat_b)

    if channels:
        side = torch.cat(channels, dim=-1).permute(0, 3, 2, 1).contiguous()
    else:
        # No time, no feature embeddings — start from an empty (B, 0, D, T)
        # tensor with no grad-tracked degenerate parameters in its history.
        side = torch.zeros(B, 0, D, T, device=device, dtype=time_embed.dtype)

    if cond_mask is not None:
        cond_mask = cond_mask.to(device).unsqueeze(1)  # (B, 1, D, T)
        side = torch.cat([side, cond_mask], dim=1)

    if padding_mask is not None:
        if padding_mask.dim() != 2 or padding_mask.shape != (B, T):
            raise ValueError(
                "padding_mask must have shape (B, T) matching time_embed; "
                f"got {tuple(padding_mask.shape)} vs (B={B}, T={T})"
            )
        # (B, T) -> (B, 1, 1, T) -> (B, 1, D, T)
        pm = padding_mask.to(device).to(side.dtype)
        pm = pm.unsqueeze(1).unsqueeze(2).expand(B, 1, D, T)
        side = torch.cat([side, pm], dim=1)

    return side  # (B, C_side, D, T)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """Create a Conv1d with Kaiming-normal weights and zero bias."""
    conv = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def _precompute_rope_freqs(
    head_dim: int, max_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute RoPE complex exponentials for up to ``max_len`` positions.

    Args:
        head_dim: Per-head dimension; must be even.
        max_len: Maximum sequence length supported at inference time.
        theta: RoPE base frequency.

    Raises:
        ValueError: If ``head_dim`` is odd.
    """
    if head_dim % 2 != 0:
        raise ValueError(
            f"head_dim={head_dim} must be even for RoPE (got an odd value). "
            "Ensure d_model is divisible by nheads and that "
            "head_dim = d_model // nheads is even."
        )
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings.

    Args:
        x: ``(B, T, nheads, head_dim)`` query or key tensor.
        freqs_cis: ``(max_len, head_dim // 2)`` complex rotation factors.

    Raises:
        ValueError: If the sequence length ``T`` exceeds ``max_len``.
    """
    B, T, H, D = x.shape
    max_len = freqs_cis.shape[0]
    if T > max_len:
        raise ValueError(
            f"Sequence length T={T} exceeds RoPE max_len={max_len}. "
            "Increase max_len when constructing TransformerEncoder (rope=True)."
        )
    xf = x.float().reshape(B, T, H, D // 2, 2).contiguous()
    x_c = torch.view_as_complex(xf)
    fc = freqs_cis[:T].unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(x_c * fc).reshape(B, T, H, D).to(x.dtype)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm, SwiGLU FFN, direct SDPA, optional RoPE."""

    def __init__(
        self,
        d_model: int,
        nheads: int,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % nheads == 0
        self.nheads = nheads
        self.head_dim = d_model // nheads
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.w_gate = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w_up = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w_down = nn.Linear(dim_feedforward, d_model, bias=False)
        self._ff_drop = nn.Dropout(dropout)
        self._attn_drop_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv_proj(h).reshape(B, T, 3, self.nheads, self.head_dim)
        q, k, v = qkv.unbind(2)
        if freqs_cis is not None:
            q = _apply_rope(q, freqs_cis)
            k = _apply_rope(k, freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            dropout_p=self._attn_drop_p if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        x = x + self.out_proj(y)
        h2 = self.norm2(x)
        x = x + self._ff_drop(self.w_down(F.silu(self.w_gate(h2)) * self.w_up(h2)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of :class:`TransformerBlock` with optional RoPE and causal masking.

    Replaces ``nn.TransformerEncoder`` with RMSNorm, SwiGLU, and direct SDPA
    (bypasses the MHA slow-path during training).
    """

    def __init__(
        self,
        d_model: int,
        nheads: int,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        max_len: int = 512,
        causal: bool = False,
        rope: bool = False,
    ):
        super().__init__()
        head_dim = d_model // nheads
        if head_dim % 8 != 0:
            raise ValueError(
                f"head_dim={head_dim} (d_model={d_model} / nheads={nheads}) "
                f"must be a multiple of 8 for SDPA backend compatibility"
            )
        self.causal = causal
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, nheads, dim_feedforward, dropout)
            for _ in range(num_layers)
        )
        if rope:
            self.register_buffer(
                "freqs_cis",
                _precompute_rope_freqs(head_dim, max_len),
                persistent=False,
            )
        else:
            self.freqs_cis = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc = self.freqs_cis
        for layer in self.layers:
            x = layer(x, freqs_cis=fc, is_causal=self.causal)
        return x


def get_torch_trans(heads=8, layers=1, channels=64, dropout: float = 0.1):
    """Build a :class:`TransformerEncoder` (RMSNorm, SwiGLU, direct SDPA).

    Args:
        heads: Number of attention heads.
        layers: Number of encoder layers.
        channels: Model dimension (``d_model``). ``channels // heads`` must be
            a multiple of 8 for SDPA backend compatibility.
        dropout: Attention/FFN dropout. Callers that gradient-checkpoint this
            module must pass ``0.0`` — a stochastic forward breaks the
            deterministic recompute.

    Returns:
        The configured :class:`TransformerEncoder`.
    """
    return TransformerEncoder(
        d_model=channels,
        nheads=heads,
        num_layers=layers,
        dim_feedforward=64,
        dropout=dropout,
    )


def softplus_inv(x: torch.Tensor | float) -> torch.Tensor:
    """Inverse softplus ``log(exp(x) - 1)``, numerically stable for ``x > 0``."""
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.log(torch.expm1(x))


def hist_abs_time_tokens(
    time_embed: torch.Tensor,  # (B, T, E_t)
    t_idx: torch.Tensor,  # (B,)
    j: int,  # number of elements
    prepend_fut: bool = False,  # do we add t at the beginning?
    plus_one: bool = False,  # if true, [t-j + 1 ... t]
) -> torch.Tensor:
    """Gather absolute time embeddings for a history window ending at ``t_idx``.

    With both flags ``False`` (the default, used by the encoder), returns the
    ``j`` slots strictly BEFORE ``t``: indices ``[t-j, ..., t-1]``, shape
    ``(B, j, E_t)`` — the current step ``t`` is excluded (no leakage).

    Flags (mutually exclusive) optionally include ``t``:
      - ``prepend_fut=True``: prepend ``t`` → ``[t, t-j, ..., t-1]``, ``(B, j+1, E_t)``.
      - ``plus_one=True``:    shift the window forward by one →
        ``[t-j+1, ..., t]``, ``(B, j, E_t)``.

    All indices are clamped to ``[0, T-1]``.
    """
    assert not (prepend_fut and plus_one)  # mutually exclusive
    assert j > 0  # not implemented otherwise
    B, T, E = time_embed.shape
    device = time_embed.device

    # offsets: [0, -j, -(j-1), ..., -1]
    offs = -torch.arange(j, 0, -1, device=device)  # [-j, ..., -1]
    if prepend_fut:
        offs = torch.cat([torch.zeros(1, device=device, dtype=torch.long), offs])
        # offs: (j+1,) = [0, -j, ..., -1]
    if plus_one:
        offs = offs.add(1)

    idx = t_idx.unsqueeze(1) + offs  # (B, j) or (B, j+1) if prepend_fut
    idx = idx.clamp(min=0, max=T - 1)

    b = torch.arange(B, device=device).unsqueeze(1).expand(B, idx.shape[1])
    return time_embed[b, idx, :]  # (B, j[+1], E_t)
