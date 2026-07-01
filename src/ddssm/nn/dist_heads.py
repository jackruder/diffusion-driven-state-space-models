"""Pluggable distribution heads for the encoder.

A :class:`BaseDistHead` consumes the combiner's feature vector and produces
``(z, logq, step_params)`` — a reparameterized sample, its log-density under
the encoder ``q``, and the per-step distribution parameters that downstream
losses can stack along the time axis (via :meth:`stack_stats`).

:class:`GaussianDistHead` wraps :class:`~ddssm.nn.gaussians.GaussianHead` for
reparameterized Gaussian sampling and supports closed-form entropy.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn

from ddssm.nn.gaussians import (
    GaussianHead,
    gaussian_entropy,
    gaussian_log_prob,
)


class BaseDistHead(nn.Module, metaclass=abc.ABCMeta):
    """Distribution-head interface: features → ``(z, logq, step_params)``."""

    def __init__(
        self,
        *,
        in_features: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.latent_dim = int(latent_dim)

    @property
    def is_gaussian_family(self) -> bool:
        """True if downstream code can rely on closed-form Gaussian KL / entropy."""
        return False

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,  # (B, in_features)
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Map features to a reparameterized sample and its log-density.

        Returns:
            z: ``(B, latent_dim)`` reparameterized sample.
            logq: ``(B,)`` log q(z | x).
            step_params: dict of per-step parameters for this head's family.
        """
        ...

    @abc.abstractmethod
    def stack_stats(self, step_params_list: list[dict]) -> dict:
        """Stack a list of per-step parameter dicts along a trailing time axis.

        Returns a dict whose values are tensors with shape ``(B, ..., T)``.
        """
        ...

    # ---- Optional closed-form helpers; defaults raise NotImplementedError ----

    def entropy_init(self, stats: dict, steps: int) -> torch.Tensor:
        raise NotImplementedError(
            "Closed-form init entropy not available for this dist head."
        )

    def entropy_transition(self, stats: dict, j: int) -> torch.Tensor:
        raise NotImplementedError(
            "Closed-form transition entropy not available for this dist head."
        )


class GaussianDistHead(BaseDistHead):
    """Reparameterized diagonal-Gaussian head.

    Wraps :class:`GaussianHead` and adds sampling + log-prob + closed-form entropy.
    """

    def __init__(
        self,
        *,
        in_features: int,
        latent_dim: int,
        init_logvar: float = 0.0,
        var_min: float = 1e-6,
        clamp_logvar_min: float = -13.0,
        clamp_logvar_max: float = 13.0,
    ) -> None:
        super().__init__(in_features=in_features, latent_dim=latent_dim)
        self.gauss_head = GaussianHead(
            in_features=in_features,
            out_features=latent_dim,
            init_logvar=init_logvar,
            var_min=var_min,
            clamp_logvar_min=clamp_logvar_min,
            clamp_logvar_max=clamp_logvar_max,
        )

    @property
    def is_gaussian_family(self) -> bool:
        return True

    def forward(
        self,
        x: torch.Tensor,
        mean_offset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        mu, logvar = self.gauss_head(x)  # (B, d), (B, d)
        if mean_offset is not None:  # persistence frame: μ = z_{t-1} + free μ
            mu = mu + mean_offset
        sigma = (0.5 * logvar).exp()
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        logq = gaussian_log_prob(z, mu, logvar)  # (B,)
        return z, logq, {"mu": mu, "logvar": logvar}

    def stack_stats(self, step_params_list: list[dict]) -> dict:
        if not step_params_list:
            return {}
        mus = torch.stack([p["mu"] for p in step_params_list], dim=-1)  # (B, d, T)
        logvars = torch.stack([p["logvar"] for p in step_params_list], dim=-1)
        return {"mus": mus, "logvars": logvars}

    def entropy_init(self, stats: dict, steps: int) -> torch.Tensor:
        assert "logvars" in stats
        logvars = stats["logvars"]  # (B, S, d, T)
        T = logvars.shape[-1]
        steps = min(steps, T)
        if steps == 0:
            return torch.zeros((), device=logvars.device, dtype=logvars.dtype)
        lv = logvars[..., :steps]
        return gaussian_entropy(lv).mean()

    def entropy_transition(self, stats: dict, j: int) -> torch.Tensor:
        assert "logvars" in stats
        logvars = stats["logvars"]  # (B, S, d, T)
        T = logvars.shape[-1]
        if j >= T:
            return torch.zeros((), device=logvars.device, dtype=logvars.dtype)
        lv = logvars[..., j:]
        return gaussian_entropy(lv).mean()
