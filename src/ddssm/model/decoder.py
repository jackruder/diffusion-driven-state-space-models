"""Decoder p_θ(x_t | z_{t-j+1:t}, time_window) over the latent history.

Runs a ContextProducer over the length-``j`` latent history to parameterise a
diagonal-Gaussian observation model for ``x_t``.
"""

import abc
import math
from functools import partial
from collections.abc import Callable

import torch
import torch.nn as nn

from ddssm.nn.diffnets import ContextProducer
from ddssm.nn.gaussians import GaussianHead
from ddssm.nn.net_utils import hist_abs_time_tokens
from ddssm.nn.torch_compile import maybe_compile


class BaseDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Common interface for decoders p_θ(x_t | z_{t-j+1:t}, ...).

    Implementations must provide ``forward`` returning ``(mu, logvar)``
    Gaussian observation parameters of shape ``(B, D)`` and
    ``log_likelihood`` returning ``(logp_t, mu_x, logvar_x, obs_count_t)``
    suitable for masked Gaussian observation models.

    Concrete decoders are composed in Python when building the model (see
    ``experiments/init_centering/model.py`` and ``src/ddssm/builders.py``),
    not selected via a Hydra config group.
    """

    @abc.abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        time_embed: torch.Tensor,
        time_idx: torch.Tensor,
        covariates: torch.Tensor | None = None,
        static_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, logvar)`` Gaussian observation params, each ``(B, D)``."""
        ...

    @abc.abstractmethod
    def log_likelihood(
        self,
        x_t: torch.Tensor,
        z_hist: torch.Tensor,
        time_embed: torch.Tensor,
        time_idx: torch.Tensor,
        observation_mask_t: torch.Tensor | None = None,
        covariates: torch.Tensor | None = None,
        static_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logp_t, mu_x, logvar_x, obs_count_t)`` for x_t."""
        ...


class GaussianDecoder(BaseDecoder):
    """Gaussian decoder p_θ(x_t | z_{t-j+1:t}, time_window).

    Treats z_{t-j+1:t} as a short sequence of length j, runs a
    ContextProducer along this history axis, and outputs diagonal Gaussian
    parameters (mu, logvar) for x_t.
    """

    def __init__(
        self,
        latent_dim: int,  # d
        data_dim: int,  # D
        j: int = 1,
        emb_time_dim: int = 64,
        covariate_dim: int = 0,
        static_covariate_dim: int = 0,
        hidden_dim: int = 64,
        mask_emb_dim: int = 8,
        context: Callable[..., ContextProducer] | None = None,
        gaussian_head: Callable[..., GaussianHead] | None = None,
    ) -> None:
        super().__init__()
        if context is None:
            context = partial(ContextProducer, channels=8, num_layers=2)
        if gaussian_head is None:
            gaussian_head = GaussianHead

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.j = j
        self.emb_time_dim = emb_time_dim
        self.covariate_dim = covariate_dim
        self.mask_emb_dim = mask_emb_dim
        self.hidden_dim = hidden_dim

        self.total_static_dim = static_covariate_dim

        # -- projection layers --
        self.z_hist_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        # -- mask embedding (for valid/pad positions) --
        self.mask_embed = nn.Linear(1, self.mask_emb_dim)

        # -- context producer --
        self.context_producer = context(
            combined_dim=self.hidden_dim,
            mask_tot_dim=self.mask_emb_dim,
            emb_time_dim=self.emb_time_dim + self.covariate_dim,
            combined_len=self.j,
            static_emb_dim=self.total_static_dim,
        )

        head_in_dim = self.context_producer.channels * self.hidden_dim

        if self.total_static_dim > 0:
            # Project spatial dimension from D (data) -> hidden_dim (latent seq spatial dim)
            self.static_proj_context = nn.Linear(data_dim, self.hidden_dim)

            # Project to Output Head's input dimension (residual/skip connection) flatly
            self.static_proj_out = nn.Linear(
                data_dim * self.total_static_dim, head_in_dim
            )
        else:
            self.static_proj_context = None
            self.static_proj_out = None

        # -- Gaussian output head --
        self.gaussian_head = gaussian_head(
            in_features=head_in_dim,
            out_features=self.data_dim,
        )

        self.context_producer = maybe_compile(self.context_producer, dynamic=True)

        # Variance prior parameters
        self.logvar_prior_mean = self.gaussian_head.init_logvar
        self.logvar_prior_std = 1.0

    def forward_unpadded(
        self,
        z: torch.Tensor,  # (B, d, j)
        time_embed: torch.Tensor,  # (B, T, E_t)
        time_idx: torch.Tensor,  # (B,) current time index t
        covariates: torch.Tensor | None = None,  # (B, V, T)
        static_embed: torch.Tensor | None = None,  # (B, D, V_s) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent history to observation parameters.

        Args:
            z: (B, d, j) latent history z_{t-j+1:t}
            time_embed: (B, T, E_t) full time embeddings
            time_idx: (B,) current time index t

        Returns:
            mu: (B, D) mean
            logvar: (B, D) log-variance
        """
        device = z.device
        B, d, j = z.shape

        assert d == self.latent_dim, f"z latent dim {d} != {self.latent_dim}"
        assert j == self.j, f"z history len {j} != configured j={self.j}"
        assert time_idx.shape == (B,)
        assert time_idx.dtype == torch.long

        # get time embeddings for history: [t-j+1, ..., t]
        hist_time_emb = hist_abs_time_tokens(
            time_embed=time_embed,
            t_idx=time_idx,
            j=self.j,
            prepend_fut=False,
            plus_one=True,
        )  # (B, j, E_t)

        if covariates is not None:
            covs = covariates.permute(0, 2, 1)  # (B, T, V)
            hist_covs = hist_abs_time_tokens(
                time_embed=covs,
                t_idx=time_idx,
                j=self.j,
                prepend_fut=False,
                plus_one=True,
            )  # (B, j, V)
            hist_time_emb = torch.cat([hist_time_emb, hist_covs], dim=-1)

        # project latent history
        z_hist = z.permute(0, 2, 1)  # (B, j, d)
        z_proj = self.z_hist_proj(z_hist)  # (B, j, H)
        combined = z_proj.permute(0, 2, 1)  # (B, H, j)

        # Process static embedding once if available
        se_flat = None
        static_context = None

        if (
            self.total_static_dim > 0
            and static_embed is not None
            and self.static_proj_context is not None
        ):
            # 1. Flatten for the late output residual
            se_flat = static_embed.reshape(B, -1)

            # 2. Transpose & Map D -> hidden_dim for the 2D ContextProducer
            # (B, D, E_static) -> (B, E_static, D)
            se_perms = static_embed.permute(0, 2, 1)
            # Map D -> hidden_dim (which is H_seq) -> Result: (B, E_static, hidden_dim)
            static_context = self.static_proj_context(se_perms)

        # time embeddings to (B, E_t, j)
        hist_time_emb = hist_time_emb.permute(0, 2, 1)  # (B, E_t, j)

        # build pad mask: for decoder, pad positions are zeros (no imputation)
        # mask is 1 for valid positions, 0 for padded
        pad_mask = self._build_pad_mask(time_idx, device, combined.dtype)  # (B, j)

        # embed mask
        mask_emb = self.mask_embed(pad_mask.unsqueeze(-1))  # (B, j, E_mask)
        mask_emb = mask_emb.permute(0, 2, 1)  # (B, E_mask, j)

        # run context producer (via __call__ so the in-place torch.compile fires)
        x = self.context_producer(
            combined=combined,
            mask_embedded=mask_emb,
            hist_time_emb=hist_time_emb,
            static_embedded=static_context,
        )  # (B, C*tot_dim)

        # residual/skip connection from static covariates if available
        if self.static_proj_out is not None and se_flat is not None:
            static_out = self.static_proj_out(se_flat)  # (B, head_in_dim)
            x = x + static_out

        # Gaussian head
        return self.gaussian_head(x)  # (mu, logvar), both (B, D)

    def _build_pad_mask(
        self,
        time_idx: torch.Tensor,  # (B,)
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build pad mask for decoder history.

        For early timestamps where t < j, we have missing history.
        Mask is 1 for valid positions, 0 for padded.
        Decoder uses zero padding with no imputation.

        Returns:
            pad_mask: (B, j) where 1=valid, 0=padded
        """
        B = time_idx.shape[0]

        # number of valid history positions: min(t+1, j)
        # at t=0: 1 valid, at t=j-1: j valid, at t>=j: j valid
        valid_len = (time_idx + 1).clamp(max=self.j)  # (B,)

        # build mask: position i is valid if i >= (j - valid_len)
        pos = torch.arange(self.j, device=device)  # (j,)
        threshold = (self.j - valid_len).unsqueeze(1)  # (B, 1)
        pad_mask = (pos >= threshold).to(dtype)  # (B, j)

        return pad_mask

    def forward(
        self,
        z: torch.Tensor,  # (B, d, k) where k <= j
        time_embed: torch.Tensor,  # (B, T, E_t)
        time_idx: torch.Tensor,  # (B,) current time index t
        covariates: torch.Tensor | None = None,
        static_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with automatic zero-padding for missing history.

        For early timestamps where we have fewer than j latents,
        this pads with zeros on the left.

        Args:
            z: (B, d, k) latent history where k <= j
            time_embed: (B, T, E_t) full time embeddings
            time_idx: (B,) current time index t

        Returns:
            mu: (B, D) mean
            logvar: (B, D) log-variance
        """
        device = z.device
        B, d, k = z.shape

        assert d == self.latent_dim
        assert k <= self.j

        if k < self.j:
            # left-pad with zeros
            num_pad = self.j - k
            pad_z = torch.zeros(B, d, num_pad, device=device, dtype=z.dtype)
            z_full = torch.cat([pad_z, z], dim=-1)  # (B, d, j)
        else:
            z_full = z

        return self.forward_unpadded(
            z_full,
            time_embed,
            time_idx,
            covariates=covariates,
            static_embed=static_embed,
        )

    def variance_prior_loss(self) -> torch.Tensor:
        """L2 penalty keeping global log-variance near N(init_logvar, prior_std^2)."""
        # Access the variance bias from the gaussian head
        logvar = self.gaussian_head._global_logvar_unclamped()
        diff = (logvar - self.logvar_prior_mean) / self.logvar_prior_std
        return 0.5 * (diff * diff).mean()

    def log_likelihood(
        self,
        x_t: torch.Tensor,  # (B, D)
        z_hist: torch.Tensor,  # (B, d, k<=j)
        time_embed: torch.Tensor,  # (B, T, E_t)
        time_idx: torch.Tensor,  # (B,)
        observation_mask_t: torch.Tensor | None = None,  # (B, D) or None
        covariates: torch.Tensor | None = None,
        static_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log p_θ(x_t | z_hist, t) under the current decoder.

        For the current Gaussian parameterization:
            p_θ(x_t | ·) = N(mu, diag(exp(logvar))).

        Returns:
            logp_t      : (B,) log-likelihood per sequence (sum over D)
            mu_x        : (B, D)
            logvar_x    : (B, D)
            obs_count_t : (B,) number of observed dims contributing
        """
        device = x_t.device
        B, D = x_t.shape
        assert self.data_dim == D

        # Get Gaussian params; this handles k<j via zero-padding
        mu_x, logvar_x = self.forward(
            z=z_hist,
            time_embed=time_embed,
            time_idx=time_idx,
            covariates=covariates,
            static_embed=static_embed,
        )  # (B, D)
        # TODO expose via config
        logvar_x = logvar_x.clamp(-20.0, 20.0)

        if observation_mask_t is None:
            m_t = torch.ones_like(x_t, device=device)
        else:
            m_t = observation_mask_t.to(device)

        resid = torch.where(m_t > 0, x_t - mu_x, torch.zeros_like(mu_x))  # (B, D)
        inv_var = torch.exp(-logvar_x)
        const = math.log(2.0 * math.pi)

        nll = 0.5 * m_t * (resid * resid * inv_var + logvar_x + const)  # (B, D)
        logp_t = -nll.sum(dim=1)  # (B,)

        obs_count_t = m_t.sum(dim=1).clamp_min(1.0)  # (B,)

        return logp_t, mu_x, logvar_x, obs_count_t


class IdentityDecoder(BaseDecoder):
    """Pinned identity emission ``p(x_t | z_t) = N(z_t, σ_x²)`` (fixed σ_x).

    Requires ``latent_dim == data_dim``. Passes the current latent ``z_t`` (the
    last history slot) straight through to ``x_t`` with a FIXED log-variance, so
    the predictive spread comes from the diffusion TRANSITION, not a learnable
    decoder. Pairs with :class:`~ddssm.model.encoder.IdentityEncoder` to make the
    pipeline an observation-space (CSDI-style) model. Param-free (no optimizer
    group). ``fixed_logvar`` defaults to ``log(0.1²)`` to match the nlblmv obs
    noise.
    """

    def __init__(
        self,
        latent_dim: int,
        data_dim: int,
        j: int = 1,
        emb_time_dim: int = 0,
        fixed_logvar: float = -4.605,  # log(0.1²): true obs-noise σ_x
        **_unused,
    ) -> None:
        super().__init__()
        if latent_dim != data_dim:
            raise ValueError(
                "IdentityDecoder requires latent_dim == data_dim; got "
                f"latent_dim={latent_dim}, data_dim={data_dim}"
            )
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.j = j
        self.emb_time_dim = emb_time_dim
        self.register_buffer(
            "fixed_logvar", torch.tensor(float(fixed_logvar)), persistent=False
        )

    def forward(
        self,
        z: torch.Tensor,  # (B, d, k) with k <= j; last slot = current z_t
        time_embed: torch.Tensor,
        time_idx: torch.Tensor,
        covariates: torch.Tensor | None = None,
        static_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, d, _k = z.shape
        assert d == self.latent_dim, f"z latent dim {d} != {self.latent_dim}"
        mu = z[:, :, -1]  # (B, D) = z_t
        logvar = self.fixed_logvar.to(device=z.device, dtype=z.dtype).expand(
            B, self.data_dim
        )
        return mu, logvar

    def variance_prior_loss(self) -> torch.Tensor:
        return torch.zeros(
            (), device=self.fixed_logvar.device, dtype=self.fixed_logvar.dtype
        )

    def log_likelihood(
        self,
        x_t: torch.Tensor,  # (B, D)
        z_hist: torch.Tensor,  # (B, d, k<=j)
        time_embed: torch.Tensor,
        time_idx: torch.Tensor,
        observation_mask_t: torch.Tensor | None = None,
        covariates: torch.Tensor | None = None,
        static_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x_t.device
        B, D = x_t.shape
        assert self.data_dim == D
        mu_x, logvar_x = self.forward(z_hist, time_embed, time_idx)

        if observation_mask_t is None:
            m_t = torch.ones_like(x_t, device=device)
        else:
            m_t = observation_mask_t.to(device)

        resid = torch.where(m_t > 0, x_t - mu_x, torch.zeros_like(mu_x))
        inv_var = torch.exp(-logvar_x)
        const = math.log(2.0 * math.pi)
        nll = 0.5 * m_t * (resid * resid * inv_var + logvar_x + const)
        logp_t = -nll.sum(dim=1)
        obs_count_t = m_t.sum(dim=1).clamp_min(1.0)
        return logp_t, mu_x, logvar_x, obs_count_t
