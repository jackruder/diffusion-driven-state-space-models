"""Gaussian distribution helpers: log-probability, entropy, and parameterisation heads."""

import math
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GaussianHeadConfig
from .net_utils import softplus_inv


def logvar_from_raw(raw: torch.Tensor, var_min) -> torch.Tensor:
    # raw -> var -> logvar (positive & stable)
    var = F.softplus(raw) + var_min
    return var.log()


def clamp_logvar(
    logvar: torch.Tensor, clamp_logvar_min, clamp_logvar_max
) -> torch.Tensor:
    return logvar.clamp(clamp_logvar_min, clamp_logvar_max)


def gaussian_log_prob(
    z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Log N(z; mu, diag(exp(logvar))) summed over last dim."""
    var = logvar.exp()
    diff = z - mu
    logp_per_dim = -0.5 * (diff * diff / var + logvar + math.log(2.0 * math.pi))
    return logp_per_dim.sum(dim=-1)  # (...,)


def gaussian_entropy(logvars: torch.Tensor) -> torch.Tensor:
    """Entropy of a diagonal Gaussian with given log-variances.

    Keeps the leading two dims (e.g., batch and sample) and sums over the rest.
    Returns a tensor shaped like logvars[:, :1, ...] squeezed to (B, S).
    """
    if logvars.ndim < 3:
        raise ValueError("logvars must have at least 3 dims (B, S, d[,...])")
    device, dtype = logvars.device, logvars.dtype
    dims_to_sum = tuple(range(2, logvars.ndim))
    num_dims = math.prod(logvars.shape[2:])
    if num_dims == 0:
        return torch.zeros(logvars.shape[:2], device=device, dtype=dtype)
    const = num_dims * math.log(2.0 * math.pi * math.e)
    return 0.5 * (logvars.sum(dim=dims_to_sum) + const)


def gaussian_kl_divergence(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) for two diagonal Gaussians.

    Returns:
        kl: (...) tensor, summed over the last dimension (latent dim d).
    """
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (
        var_q / var_p + (mu_p - mu_q).pow(2) / var_p - 1.0 + logvar_p - logvar_q
    )
    return kl.sum(dim=-1)


class GaussianStats(TypedDict, total=False):
    # Present for Gaussian encoders, optional for others
    mus: torch.Tensor  # (B, S, d, T)
    logvars: torch.Tensor  # (B,S, d, T)
    # Optional additional fields (flows, etc.)
    # e.g., 'base_logqs': torch.Tensor  # if you track flow base log density


class GaussianHead(nn.Module):
    """Produces Gaussian parameters (mu, logvar) from input features.

    Takes a feature vector and outputs mean and log-variance for a
    Gaussian distribution, with numerical stability guarantees.
    """

    def __init__(
        self,
        config: GaussianHeadConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.var_min = float(config.var_min)
        self.clamp_logvar_min = float(config.clamp_logvar_min)
        self.clamp_logvar_max = float(config.clamp_logvar_max)
        self.init_logvar = float(config.init_logvar)

        # Mean head
        self.mu_head = nn.Linear(in_features, out_features)

        # Variance head (outputs raw value before softplus)
        self.var_head_raw = nn.Linear(in_features, out_features)

        # Learnable bias for variance, initialized to target init_logvar
        init_var = float(math.exp(self.init_logvar))
        v_soft = float(max(init_var - self.var_min, 1e-6))
        raw0 = float(softplus_inv(v_soft).item())
        self.var_bias_raw = nn.Parameter(
            torch.full((out_features,), raw0, dtype=torch.float32)
        )

        # Weight initialization
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.5)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.var_head_raw.weight)
        nn.init.zeros_(self.var_head_raw.bias)

    def _global_logvar_unclamped(self) -> torch.Tensor:
        """Return the global (input-independent) log-variance before clamping.

        Useful for variance prior regularization.
        """
        var = F.softplus(self.var_bias_raw) + self.var_min
        return var.log()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Gaussian parameters from input features.

        Args:
            x: Input features of shape (..., in_features)

        Returns:
            mu: Mean of shape (..., out_features)
            logvar: Log-variance of shape (..., out_features), clamped for stability
        """
        mu = self.mu_head(x)

        raw = self.var_head_raw(x) + self.var_bias_raw
        logvar = logvar_from_raw(raw, self.var_min)
        logvar = logvar.clamp(self.clamp_logvar_min, self.clamp_logvar_max)

        return mu, logvar
