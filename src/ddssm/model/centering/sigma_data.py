"""EMA buffer tracking the per-step centered-residual variance ``σ_data²(t)``.

Per ``model-v2.org`` § Data-variance tracking and § σ_data buffer
extension, the diffusion transition needs ``σ_data²(t)`` to scale the EDM
preconditioning constants ``(c_skip, c_out, c_in)``.  The buffer:

* Covers ``t = 1 … T`` (extended to include the VHP-covered initial
  ``j`` slots).
* Accumulates *passively* throughout stage 1 (the stage-1 Gaussian
  closed-form KL does not consume the buffer, but the stage-1
  transition still calls ``update(...)`` with the centered moments so
  the buffer is populated by the close of pretraining).  This stage-1
  pre-warm is *not required* under ``"per_t"``/``"global_ema"``: the
  EMA warmup (see :meth:`_update_unchecked`) makes the buffer
  self-calibrate within ``~1/(1-γ)`` stage-2 updates from any init, so
  a stage-2-only run still gets a correctly-scaled buffer.  Only
  ``"fixed"`` (frozen at handoff) genuinely needs a representative
  encoder before the freeze.
* Carries its value through the stage-1 → stage-2 handoff
  (§ Stage-1 → stage-2 handoff step 5); only the EMA *schedule* (step
  counter, ``frozen`` flag for "fixed" tracking) resets at handoff.

Three tracking modes per § Tracking-mode variants:

* ``"fixed"``     — the buffer is held at its handoff value for all of
  stage 2; ``update`` is a no-op once ``frozen`` is set by
  :meth:`reset_schedule`.
* ``"global_ema"`` — every per-t lookup reads the same scalar; updates
  pool across the timesteps visited in the batch.
* ``"per_t"``    — independent per-t buffers; each ``update`` touches
  only the slots whose t is supplied.

External callers index the buffer 1-based (``t = 1 … T_max`` matches
the doc's notation).  Internally the array is 0-based.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

_TRACKING_MODES = ("fixed", "global_ema", "per_t")


def _coerce_t_idx(t_idx: int | torch.Tensor) -> torch.Tensor:
    if isinstance(t_idx, int):
        return torch.tensor([t_idx], dtype=torch.long)
    if t_idx.dim() == 0:
        return t_idx.long().reshape(1)
    return t_idx.long()


class SigmaDataBuffer(nn.Module):
    """EMA buffer of σ_data²(t) for the diffusion transition.

    Args:
        T_max: Max latent timestep covered by the buffer (1-based,
            inclusive).
        tracking_mode: One of "fixed", "global_ema", "per_t". Default
            ``"per_t"`` — each timestep gets its own running EMA, kept
            tracking through stage 2 so the centered-ESM target
            preconditioning stays calibrated as the encoder's residual
            distribution drifts. ``"fixed"`` freezes the buffer at the
            centering handoff (init_smoke_simple still uses this for
            its numerical V2 anchor); ``"global_ema"`` collapses all t
            into one shared scalar.
        ema_decay: Steady-state EMA decay γ in [0, 1).  Larger → slower
            tracking once warmed up.  A per-slot warmup (running-mean
            blend for the first ``~1/(1-γ)`` updates) makes the *initial*
            convergence fast regardless of γ, so γ can stay high for
            stable drift tracking without paying a slow cold start.
        init_value: Value used to fill every slot at construction.  Only
            matters before the first update — the warmup's first step
            (α=1) fully replaces it — so it is just a safe default for
            reads that happen before any training step (e.g. the very
            first stage-2 forward, which reads then updates).  Default
            1.0 ("approximately unit variance").
    """

    sigma_data2: torch.Tensor
    ema_step: torch.Tensor
    n_updates: torch.Tensor

    def __init__(
        self,
        T_max: int,
        tracking_mode: str = "per_t",
        ema_decay: float = 0.999,
        init_value: float = 1.0,
    ) -> None:
        super().__init__()
        if T_max <= 0:
            raise ValueError(f"T_max must be > 0; got {T_max}")
        if tracking_mode not in _TRACKING_MODES:
            raise ValueError(
                f"tracking_mode must be one of {_TRACKING_MODES}; "
                f"got {tracking_mode!r}"
            )
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1); got {ema_decay}")

        self.T_max = int(T_max)
        self.tracking_mode = tracking_mode
        self.ema_decay = float(ema_decay)
        self.init_value = float(init_value)
        self.frozen: bool = False

        self.register_buffer(
            "sigma_data2",
            torch.full((self.T_max,), self.init_value, dtype=torch.float32),
        )
        self.register_buffer(
            "ema_step",
            torch.zeros(self.T_max, dtype=torch.long),
        )
        # Lifetime per-slot update count driving the EMA warmup. Distinct from
        # ``ema_step``: it is NOT reset at the stage-1 → stage-2 handoff, so a
        # slot already primed in stage 1 keeps tracking with the steady-state
        # EMA instead of re-warming and discarding its persisted value. Persisted
        # in ``state_dict`` so a preemption-resume does not re-fire the warmup.
        self.register_buffer(
            "n_updates",
            torch.zeros(self.T_max, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read(self, t_idx: int | torch.Tensor) -> torch.Tensor:
        """Return current σ_data²(t).

        Under ``"global_ema"`` every t maps to the same scalar buffer
        slot (we keep all entries synchronised, so any read returns the
        right value).
        """
        idx = _coerce_t_idx(t_idx)
        self._check_in_range(idx)
        # External 1-based → internal 0-based.
        return self.sigma_data2[idx - 1]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self,
        t_idx: int | torch.Tensor,
        mu_hat_batch: torch.Tensor,
        sigma_t2_batch: torch.Tensor,
    ) -> None:
        """Apply one EMA update for the timesteps in ``t_idx``.

        σ_data is a TRAINING-only running statistic, so the update is a no-op
        when ``frozen`` OR when autograd is disabled. Eval / inference passes run
        under ``torch.no_grad``; mutating σ_data there corrupts the buffer and
        inflates the eval ELBO's transition-KL term (the eval drifts σ_data
        toward the eval-data residual over its forward passes). This gate must
        live OUTSIDE any ``@torch.no_grad`` wrapper so it observes the *caller's*
        grad state — hence the split into :meth:`_update_unchecked`. Regression:
        ``tests/test_centering/test_sigma_data.py::test_update_is_noop_under_no_grad``.

        Args:
            t_idx: scalar or (n,) tensor of 1-based timestep indices.
            mu_hat_batch: ``(N, d)`` centered residual means.  If
                ``t_idx`` is a vector of length ``n``, ``N`` may be a
                multiple of ``n`` and the rows are assumed to be
                blocked by t (rows ``[k·B : (k+1)·B]`` correspond to
                ``t_idx[k]``).  See :func:`_estimator_per_t`.
            sigma_t2_batch: ``(N, d)`` matching per-sample encoder
                posterior variance.
        """
        if self.frozen or not torch.is_grad_enabled():
            return
        self._update_unchecked(t_idx, mu_hat_batch, sigma_t2_batch)

    @torch.no_grad()
    def _update_unchecked(
        self,
        t_idx: int | torch.Tensor,
        mu_hat_batch: torch.Tensor,
        sigma_t2_batch: torch.Tensor,
    ) -> None:
        """The EMA update body; assumes the caller already gated on frozen/grad."""
        idx = _coerce_t_idx(t_idx).to(self.sigma_data2.device)
        self._check_in_range(idx)

        bar = self._estimator_per_t(idx, mu_hat_batch, sigma_t2_batch)
        # EMA warmup: blend at the running-mean rate ``1/(n+1)`` until it falls
        # to the steady-state rate ``1 - γ``, where ``n`` is the slot's lifetime
        # update count. The first real update (n=0 → α=1) fully replaces the
        # uninformative ``init_value``, and the early estimate is the exact
        # batch-average — so a freshly-built buffer is calibrated within a few
        # steps instead of taking ``~1/(1-γ)`` steps to decay the init away.
        # After the crossover (``n ≳ 1/(1-γ)``) ``α`` pins to ``1 - γ`` and the
        # buffer reverts to a plain EMA that tracks the encoder's drift.
        min_alpha = 1.0 - self.ema_decay

        if self.tracking_mode == "global_ema":
            scalar = bar.mean()
            alpha = max(min_alpha, 1.0 / (float(self.n_updates[0]) + 1.0))
            new_value = (1.0 - alpha) * self.sigma_data2[0] + alpha * scalar
            self.sigma_data2.fill_(new_value)
            self.ema_step += 1
            self.n_updates += 1
            return

        # "per_t" or "fixed-but-still-accumulating" (pre-handoff stage 1).
        ext = idx - 1  # to internal 0-based
        step = self.n_updates[ext].to(torch.float32)
        alpha = torch.clamp(1.0 / (step + 1.0), min=min_alpha)  # (n,)
        new_value = (1.0 - alpha) * self.sigma_data2[ext] + alpha * bar
        self.sigma_data2[ext] = new_value
        self.ema_step[ext] += 1
        self.n_updates[ext] += 1

    @staticmethod
    def _estimator_per_t(
        idx: torch.Tensor,
        mu_hat_batch: torch.Tensor,
        sigma_t2_batch: torch.Tensor,
    ) -> torch.Tensor:
        """``bar_σ_data²(t) = (1/D) (E[‖σ_t‖²] + tr Var[μ̂_t])`` per t.

        Per ``model-v2.org`` § Data-variance tracking, the per-batch
        estimator decomposes into average posterior variance plus the
        spread of residual means.  We compute it per t when
        ``mu_hat_batch`` / ``sigma_t2_batch`` are pre-grouped by t.

        Args:
            idx: ``(n,)`` 1-based t indices.
            mu_hat_batch: ``(N, d)`` centered means; ``N`` is split
                into ``n`` equal-sized blocks corresponding to ``idx``.
            sigma_t2_batch: ``(N, d)`` per-sample posterior variances.

        Returns:
            ``(n,)`` per-t estimator values.
        """
        if mu_hat_batch.shape != sigma_t2_batch.shape:
            raise ValueError(
                "mu_hat_batch and sigma_t2_batch shapes differ: "
                f"{tuple(mu_hat_batch.shape)} vs {tuple(sigma_t2_batch.shape)}"
            )
        N, d = mu_hat_batch.shape
        n = idx.shape[0]
        if N % n != 0:
            raise ValueError(
                f"mu_hat_batch.shape[0]={N} not divisible by len(t_idx)={n}"
            )
        per_t = N // n

        mu_blocks = mu_hat_batch.view(n, per_t, d)
        s2_blocks = sigma_t2_batch.view(n, per_t, d)

        avg_post_var = s2_blocks.mean(dim=1).sum(dim=1)  # (n,) = E[‖σ_t‖²] = E[Σ_d σ²_d]
        # tr Var[μ̂_t] = sum_d Var_b[μ̂_{t,b,d}]. Use Bessel-corrected
        # (``unbiased=True``) so the EMA's steady-state target is the true
        # marginal variance regardless of ``per_t``. The biased (1/per_t)
        # estimator shifts the target by a factor ``(per_t − 1)/per_t``
        # (~6% at per_t=16), making σ_data² depend on batch size. Fall
        # back to zero when ``per_t == 1`` — a single sample carries no
        # cross-sample dispersion information.
        if per_t > 1:
            mu_var = mu_blocks.var(dim=1, unbiased=True).sum(dim=1)  # (n,)
        else:
            mu_var = torch.zeros(n, device=mu_blocks.device, dtype=mu_blocks.dtype)

        return (avg_post_var + mu_var) / float(d)

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset_schedule(self) -> None:
        """Reset the EMA schedule for the stage-1 → stage-2 handoff.

        Zeros the per-t step counter and (under "fixed" tracking)
        freezes the buffer.  Does NOT touch ``sigma_data2`` — the
        values persist across the handoff per
        ``model-v2.org`` § Stage-1 → stage-2 handoff step 5 — nor
        ``n_updates``: a slot primed in stage 1 must keep tracking with
        the steady-state EMA in stage 2, not re-warm and discard its
        persisted value.
        """
        self.ema_step.zero_()
        if self.tracking_mode == "fixed":
            self.frozen = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_in_range(self, idx: torch.Tensor) -> None:
        if (idx < 1).any() or (idx > self.T_max).any():
            raise IndexError(
                f"t_idx out of range [1, {self.T_max}]: "
                f"{idx.tolist() if idx.numel() < 16 else idx.shape}"
            )

    def extra_repr(self) -> str:
        return (
            f"T_max={self.T_max}, tracking_mode={self.tracking_mode!r}, "
            f"ema_decay={self.ema_decay}, init_value={self.init_value}, "
            f"frozen={self.frozen}"
        )


def visited_timesteps(t_min: int, t_max: int) -> Iterable[int]:
    """Convenience helper for ``range(t_min, t_max + 1)`` (1-based, inclusive)."""
    return range(t_min, t_max + 1)
