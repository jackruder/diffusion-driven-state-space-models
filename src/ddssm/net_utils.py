import torch
import torch.nn as nn


def time_embedding(pos: torch.Tensor, d_model: int = 128, device: torch.device = "cpu"):
    """Sinusoidal (Vaswani) embeddings for integer or real timestamps.
    pos: (B, T) -> returns (B, T, d_model)
    """
    pos = pos.to(device).float()  # ensure float for sin/cos
    B, T = pos.shape
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
):
    """Build covariate information tensors.

    For a batch of size B and sequence length T, for data of dimension D,
    The side information combines:
        - given time embeddings per timestep (of dimension E_t)
        - learned feature embeddings per data dimension (of dimension E_f)
        - optional conditioning mask (1 channel)

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


    Returns:
        side_info: (B, C_side, D, T), where
            C_side = E_t + E_f (+1 if cond_mask provided)
    """
    B, T, E_t = time_embed.shape
    D = data_dim

    time_embed = time_embed.to(device)
    feats = torch.arange(D, device=device)
    feat_embed = embed_layer(feats)  # (D, E_f)

    # Broadcast to (B, T, D, *)
    time_b = time_embed.unsqueeze(2).expand(B, T, D, E_t)
    feat_b = feat_embed.unsqueeze(0).unsqueeze(0).expand(B, T, D, -1)

    # Concatenate along channels then permute to (B, C_side, D, T)
    side = torch.cat([time_b, feat_b], dim=-1).permute(0, 3, 2, 1).contiguous()

    if cond_mask is not None:
        cond_mask = cond_mask.to(device).unsqueeze(1)  # (B, 1, D, T)
        side = torch.cat([side, cond_mask], dim=1)

    return side  # (B, C_side, D, T)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """Utility to create a 1×1 Conv1d with daiming init and zero bias."""
    conv = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def softplus_inv(x: torch.Tensor | float) -> torch.Tensor:
    # inverse of softplus: log(exp(x) - 1), numerically stable for x>0
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.log(torch.expm1(x))


def hist_abs_time_tokens(
    time_embed: torch.Tensor,  # (B, T, E_t)
    t_idx: torch.Tensor,  # (B,)
    j: int,  # number of elements
    prepend_fut: bool = False,  # do we add t at the beginning?
    plus_one: bool = False,  # if true, [t-j + 1 ... t]
) -> torch.Tensor:
    """Return absolute time embeddings.
    Default behavior is to return [t-j, ..., t], where t is given
    by t_idx.

    Args:
    for the j+1 history slots
    [t, t-j, t-(j-1), ..., t-1], clamping indices to [0, T-1].

    Shape: (B, j+1, E_t) or (B, j+1, E_t)
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

    idx = t_idx.unsqueeze(1) + offs  # (B, j+1)
    idx = idx.clamp(min=0, max=T - 1)

    b = torch.arange(B, device=device).unsqueeze(1).expand(B, idx.shape[1])  # (B, j+1)
    return time_embed[b, idx, :]  # (B, j+1, E_t)
