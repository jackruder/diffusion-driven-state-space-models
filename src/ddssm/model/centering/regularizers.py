"""Free-function regularizers for the model-v2 baseline-centering scheme.

* :func:`r_sigma_p_loss` — the stage-1 global log-variance anchor
  ``(λ_σp / 2) · (mean log σ_p²)²`` (``model-v2.org`` § State-conditional
  prior variance).  Pulls the average of ``log σ_p²`` over ``(t, d,
  batch)`` toward zero so ``σ_p`` settles near 1 without pinning
  per-(t, d) means.

* :func:`r_mu_p_loss` — the stage-2 Learnable-mode soft anchor on μ_p
  toward its handoff snapshot μ_p^(0) (``model-v2.org`` §
  Baseline-mode variants).  Evaluated under the encoder on trajectory
  samples; the input is detached by the caller so the regularizer
  affects only ``μ_p``, not the encoder.

Both are pure functions, not module methods on :class:`BaseBaseline`:
the regularizers depend on training-time bookkeeping (anchor module,
λ) rather than the baseline itself.
"""

from __future__ import annotations

import torch

from ddssm.model.centering.baselines import BaseBaseline


def r_sigma_p_loss(
    baseline: BaseBaseline,
    z_hist_samples: torch.Tensor,
    lambda_sigma_p: float,
) -> torch.Tensor:
    """``(λ_σp / 2) · (mean log σ_p²)²`` over the batch's z_hist samples.

    Per ``model-v2.org`` § State-conditional prior variance, the
    regularizer is global: the mean of ``log σ_p²`` over
    ``(t, d, batch)`` is squared and scaled by λ_σp / 2.  Caller
    supplies ``z_hist_samples`` already detached / shaped ``(N, d, j)``;
    we pass them through ``baseline.mean_and_logvar`` and average over
    all dimensions of the resulting ``log σ_p²``.

    Args:
        baseline: Live baseline module (μ_p / σ_p heads).
        z_hist_samples: ``(N, d, j)`` history samples (already detached
            from the encoder graph if the caller does not want the
            regularizer to flow into the encoder).
        lambda_sigma_p: λ_σp scalar.

    Returns:
        Scalar regularizer value.
    """
    if lambda_sigma_p <= 0.0:
        return torch.zeros((), device=z_hist_samples.device, dtype=z_hist_samples.dtype)
    _, log_sigma_p2 = baseline.mean_and_logvar(z_hist_samples)
    return 0.5 * lambda_sigma_p * log_sigma_p2.mean().pow(2)


def r_mu_p_loss(
    baseline: BaseBaseline,
    baseline_anchor: BaseBaseline,
    z_hist_samples: torch.Tensor,
    lambda_mu_p: float,
) -> torch.Tensor:
    """``(λ_μp / 2) · E ‖μ_p(z_{t-1}) − μ_p^(0)(z_{t-1})‖²``.

    Per ``model-v2.org`` § Baseline-mode variants, evaluated under the
    encoder on trajectory samples ``z_{t-1}^(s)`` already drawn for
    the ELBO; gradient on ``z_hist_samples`` is stopped by the caller
    so the regularizer contributes only to μ_p (and not back through
    the encoder).  The anchor is the frozen snapshot taken at the
    stage-1 → stage-2 handoff.

    Args:
        baseline: Live baseline module.
        baseline_anchor: Frozen snapshot returned by
            :meth:`BaseBaseline.snapshot` at the handoff.
        z_hist_samples: ``(N, d, j)`` already detached.
        lambda_mu_p: λ_μp scalar.

    Returns:
        Scalar regularizer value.
    """
    if lambda_mu_p <= 0.0:
        return torch.zeros((), device=z_hist_samples.device, dtype=z_hist_samples.dtype)
    mu_live = baseline.mean(z_hist_samples)
    with torch.no_grad():
        mu_anchor = baseline_anchor.mean(z_hist_samples)
    diff = mu_live - mu_anchor
    # Per-sample squared L2, averaged over the batch.
    per_sample = diff.pow(2).sum(dim=1)
    return 0.5 * lambda_mu_p * per_sample.mean()
