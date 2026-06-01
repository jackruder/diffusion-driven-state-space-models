"""Variational posterior over the auxiliary state z_{-j+1:0}.

Implements ``q_Φ(z_{-j+1:0} | z_{1:j})`` for the VHP-via-diffusion
construction described in ``model-v2.org`` § Reframing VHP to use
diffusion.

This module sits next to :mod:`ddssm.model.encoder` because it is part of
the *variational inference* machinery: the auxiliary latents
``z_{-j+1:0}`` are unobserved random variables and ``q_Φ`` is the
diagonal-Gaussian amortised posterior the ELBO uses to evaluate the
initial-state term.  It is NOT part of the baseline-centering
machinery in :mod:`ddssm.model.centering` — only the *transition*
machinery is centered; the aux posterior is centering-agnostic.

The implementation is the parameter-free pieces of the now-removed
legacy InitPrior (its ``context_producer_aux``, ``aux_proj``,
``aux_posterior_head``, ``aux_posterior_params``,
``sample_aux_posterior``) reduced to a small MLP.  The InitPrior's
own Gaussian head + ``latent_init`` padding module + hierarchical-KL
bound are NOT ported — they are replaced by the t=1..j VHP path that
the transitions own (see :doc:`ADR-0006 </adr/0006-polymorphic-transition-interface>`).

Per the doc, ``q_Φ`` is described as a "small diagonal-Gaussian MLP
head" conditioned on z_1 (or, in the general-j case, z_{1:j}).  We
do not condition on time embeddings or covariates: the doc's
construction is intentionally minimal.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

# Clamp bounds for the raw ``logvar`` output of the q_Φ head. Matches the
# encoder ``GaussianHead`` and ``centering.baselines`` convention; without
# the guard a single Linear layer can emit logvar≈±20 and NaN the KL.
_LOGVAR_MIN: float = -9.0
_LOGVAR_MAX: float = 6.0


class AuxPosterior(nn.Module):
    """Diagonal-Gaussian amortised posterior ``q_Φ(z_{-j+1:0} | z_{1:j})``.

    Args:
        latent_dim: Latent dimension ``d`` of each auxiliary z.
        j: Latent history length / number of aux latents.
        hidden_dim: Hidden layer width of the small MLP body.
        n_layers: Number of hidden layers in the MLP body.
    """

    def __init__(
        self,
        latent_dim: int,
        j: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.j = int(j)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)

        in_dim = self.latent_dim * self.j
        out_dim = 2 * self.latent_dim * self.j  # mu + logvar

        layers: list[nn.Module] = [nn.Linear(in_dim, self.hidden_dim), nn.SiLU()]
        for _ in range(max(0, self.n_layers - 1)):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(self.hidden_dim, out_dim))
        self.body = nn.Sequential(*layers)

    def forward(self, z_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ``q_Φ(z_{-j+1:0} | z_{1:j})`` parameters.

        Args:
            z_init: ``(B, d, j)`` tensor of the first j real latents.

        Returns:
            ``(aux_mu, aux_logvar)``, each ``(B, d, j)``.
        """
        if z_init.dim() != 3:
            raise ValueError(
                f"z_init must be (B, d, j); got shape {tuple(z_init.shape)}"
            )
        B, d, j = z_init.shape
        if d != self.latent_dim or j != self.j:
            raise ValueError(
                f"z_init shape mismatch: expected (B, {self.latent_dim}, "
                f"{self.j}); got (B, {d}, {j})"
            )

        h = self.body(z_init.reshape(B, d * j))  # (B, 2*d*j)
        aux_mu, aux_logvar = h.chunk(2, dim=-1)
        aux_mu = aux_mu.view(B, d, j)
        aux_logvar = aux_logvar.view(B, d, j).clamp(min=_LOGVAR_MIN, max=_LOGVAR_MAX)
        return aux_mu, aux_logvar

    def sample(
        self, z_init: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterised draw from ``q_Φ(· | z_{1:j})``.

        Returns:
            ``(z_aux, aux_mu, aux_logvar)``, each ``(B, d, j)``.  Gradients
            flow through ``aux_mu`` and ``aux_logvar`` via the
            reparameterisation trick.
        """
        aux_mu, aux_logvar = self.forward(z_init)
        aux_sigma = (0.5 * aux_logvar).exp()
        eps = torch.randn_like(aux_mu)
        z_aux = aux_mu + aux_sigma * eps
        return z_aux, aux_mu, aux_logvar

    @staticmethod
    def kl_against_standard_normal(
        aux_mu: torch.Tensor,
        aux_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Analytic ``KL[q_Φ(z_aux | z_init) || N(0, I_{j·d})]``.

        Computed per-element then summed over the ``j·d`` dims, then
        averaged over the batch — matches the convention used by the
        now-removed legacy InitPrior's hierarchical-KL bound.

        Args:
            aux_mu: ``(B, d, j)`` posterior means.
            aux_logvar: ``(B, d, j)`` posterior log-variances.

        Returns:
            Scalar tensor.
        """
        # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * (mu^2 + sigma^2 - 1 - log sigma^2)
        per_elem = 0.5 * (aux_mu.pow(2) + aux_logvar.exp() - 1.0 - aux_logvar)
        per_batch = per_elem.sum(dim=(1, 2))  # (B,)
        return per_batch.mean()
