"""EMA buffer tracking the per-step centered-residual variance ``σ_data²(t)``.

Per ``model-v2.org`` § Data-variance tracking and § σ_data buffer
extension, the diffusion transition needs ``σ_data²(t)`` to scale the EDM
preconditioning constants ``(c_skip, c_out, c_in)``.  The buffer covers
``t = 1 … T`` (extended to include the VHP-covered initial ``j`` slots).
Under the tracking modes the EMA warmup (see :meth:`_update_unchecked`)
makes the buffer self-calibrate within ``~1/(1-γ)`` updates from any
init.

Three tracking modes per § Tracking-mode variants:

* ``"fixed"``     — a true constant ``σ_data² ≡ init_value`` (=1),
  frozen from construction; ``update`` is a permanent no-op.
* ``"global_ema"`` — every per-t lookup reads the same scalar; updates
  pool across the timesteps visited in the batch.
* ``"per_t"``    — independent per-t buffers; each ``update`` touches
  only the slots whose t is supplied.

External callers index the buffer 1-based (``t = 1 … T_max`` matches
the doc's notation).  Internally the array is 0-based.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

try:
    import torch.distributed as dist

    _dist_available = True
except ImportError:
    dist = None  # type: ignore[assignment]
    _dist_available = False

_TRACKING_MODES = ("fixed", "global_ema", "per_t")


def _coerce_t_idx(
    t_idx: int | torch.Tensor, *, device: torch.device | None = None
) -> torch.Tensor:
    if isinstance(t_idx, int):
        # ``device=`` matters under CUDA graph capture — creating a CPU
        # tensor and letting the caller move it is not permitted (unpinned
        # CPU→GPU copies fail capture). Caller passes the buffer's device.
        return torch.tensor([t_idx], dtype=torch.long, device=device)
    if t_idx.dim() == 0:
        return t_idx.long().reshape(1)
    return t_idx.long()


class SigmaDataBuffer(nn.Module):
    """EMA buffer of σ_data²(t) for the diffusion transition.

    Args:
        T_max: Max latent timestep covered by the buffer (1-based,
            inclusive).
        tracking_mode: One of "fixed", "global_ema", "per_t". Default
            ``"per_t"`` — each timestep gets its own running EMA so the
            centered-ESM target preconditioning stays calibrated as the
            encoder's residual distribution drifts. ``"fixed"`` holds the
            buffer at ``init_value`` for the whole run (frozen from
            construction, ``update`` a no-op); ``"global_ema"`` collapses
            all t into one shared scalar.
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
                f"tracking_mode must be one of {_TRACKING_MODES}; got {tracking_mode!r}"
            )
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1); got {ema_decay}")

        self.T_max = int(T_max)
        self.tracking_mode = tracking_mode
        self.ema_decay = float(ema_decay)
        self.init_value = float(init_value)
        # "fixed" is a true constant σ_data² ≡ init_value: freeze from
        # construction so `update` is a permanent no-op from step 0.
        self.frozen: bool = tracking_mode == "fixed"

        self.register_buffer(
            "sigma_data2",
            torch.full((self.T_max,), self.init_value, dtype=torch.float32),
        )
        self.register_buffer(
            "ema_step",
            torch.zeros(self.T_max, dtype=torch.long),
        )
        # Lifetime per-slot update count driving the EMA warmup. Persisted in
        # ``state_dict`` so a preemption-resume does not re-fire the warmup.
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
        idx = _coerce_t_idx(t_idx, device=self.sigma_data2.device)
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
        idx = _coerce_t_idx(t_idx, device=self.sigma_data2.device)
        self._check_in_range(idx)

        suff = self._suff_stats_per_t(idx, mu_hat_batch, sigma_t2_batch)

        # DDP all-reduce: accumulate sufficient statistics across ranks before
        # estimating.  This is a preventive fix — single-rank runs are a no-op
        # because neither branch is entered.  Under multi-rank DDP, each rank
        # computes its partial suff stats; the all-reduce SUM yields the global
        # suff stats, and the pure estimator then produces an unbiased estimate
        # over the full global batch (Bessel correction uses the combined count).
        if _dist_available and dist.is_initialized():
            # Pack all four suff-stat tensors into a single flat 1-D vector so
            # we incur only one all-reduce call per update.  Layout:
            #   [ sum_mu.flatten()  |  sum_mu2_total  |  sum_s2_total  |  count ]
            # n and d are known from idx and mu_hat_batch so unpacking is exact.
            n, d = suff["sum_mu"].shape
            payload = torch.cat(
                [
                    suff["sum_mu"].reshape(-1),        # n*d
                    suff["sum_mu2_total"],              # n
                    suff["sum_s2_total"],               # n
                    suff["count"].to(suff["sum_mu"].dtype),  # n (cast for cat)
                ]
            )
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
            # Unpack
            offset = 0
            suff = {
                "sum_mu": payload[offset : offset + n * d].reshape(n, d),
                "sum_mu2_total": payload[offset + n * d : offset + n * d + n],
                "sum_s2_total": payload[offset + n * d + n : offset + n * d + 2 * n],
                "count": payload[offset + n * d + 2 * n : offset + n * d + 3 * n].long(),
            }

        bar = self._estimator_from_suff_stats(suff, mu_hat_batch.shape[1])

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

        # "per_t" (the "fixed" mode is frozen from construction, so `update`
        # never reaches here).
        ext = idx - 1  # to internal 0-based
        step = self.n_updates[ext].to(torch.float32)
        alpha = torch.clamp(1.0 / (step + 1.0), min=min_alpha)  # (n,)
        new_value = (1.0 - alpha) * self.sigma_data2[ext] + alpha * bar
        self.sigma_data2[ext] = new_value
        self.ema_step[ext] += 1
        self.n_updates[ext] += 1

    @staticmethod
    def _suff_stats_per_t(
        idx: torch.Tensor,
        mu_hat_batch: torch.Tensor,
        sigma_t2_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute per-t sufficient statistics for the σ_data² estimator.

        Returns additive (linear) statistics that can be summed across DDP
        ranks before passing to :meth:`_estimator_from_suff_stats`.

        Args:
            idx: ``(n,)`` 1-based t indices.
            mu_hat_batch: ``(N, d)`` centered means; ``N`` is split
                into ``n`` equal-sized blocks corresponding to ``idx``.
            sigma_t2_batch: ``(N, d)`` per-sample posterior variances.

        Returns:
            Dict with:
            - ``sum_mu``: ``(n, d)`` — sum of per-sample means over K samples.
            - ``sum_mu2_total``: ``(n,)`` — sum of (μ²) over K and d.
            - ``sum_s2_total``: ``(n,)`` — sum of σ² over K and d.
            - ``count``: ``(n,)`` — number of samples K per t.
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

        mu_blocks = mu_hat_batch.view(n, per_t, d)   # (n, K, d)
        s2_blocks = sigma_t2_batch.view(n, per_t, d)  # (n, K, d)

        sum_mu = mu_blocks.sum(dim=1)                        # (n, d)
        sum_mu2_total = mu_blocks.pow(2).sum(dim=1).sum(dim=1)  # (n,)
        sum_s2_total = s2_blocks.sum(dim=1).sum(dim=1)       # (n,)
        count = torch.full(
            (n,), per_t, dtype=torch.long, device=mu_hat_batch.device
        )

        return {
            "sum_mu": sum_mu,
            "sum_mu2_total": sum_mu2_total,
            "sum_s2_total": sum_s2_total,
            "count": count,
        }

    @staticmethod
    def _estimator_from_suff_stats(
        suff_stats: dict[str, torch.Tensor],
        d: int,
    ) -> torch.Tensor:
        """Compute the per-t σ_data² estimator from sufficient statistics.

        ``bar_σ_data²(t) = (1/D) (avg_post_var + tr Var[μ̂_t])``

        The Bessel-corrected variance uses the *combined* count (sum over
        ranks after all-reduce), so two ranks with 1 sample each yield
        count=2 and real cross-rank dispersion — unlike the single-rank
        per_t==1 fallback which correctly zeros mu_var (no dispersion).

        Args:
            suff_stats: Dict from :meth:`_suff_stats_per_t` (possibly
                all-reduced across DDP ranks).
            d: Feature dimension.

        Returns:
            ``(n,)`` per-t estimator values.
        """
        sum_mu = suff_stats["sum_mu"]             # (n, d)
        sum_mu2_total = suff_stats["sum_mu2_total"]  # (n,)
        sum_s2_total = suff_stats["sum_s2_total"]    # (n,)
        count = suff_stats["count"].to(sum_mu.dtype)  # (n,) float for division

        avg_post_var = sum_s2_total / count  # (n,) mean σ² over K and d
        # Bessel-corrected variance of μ̂ across K samples, summed over d:
        #   Var[μ̂] = (Σ_k μ̂_k² − (Σ_k μ̂_k)²/K) / (K − 1)
        # The fallback (count == 1) zeros mu_var: a single sample carries no
        # dispersion information.  After DDP all-reduce, count reflects the
        # combined global sample count, so two ranks × 1 sample = count 2
        # and mu_var is computed from the real cross-rank dispersion.
        mu_var_numer = sum_mu2_total - sum_mu.pow(2).sum(dim=1) / count  # (n,)
        mu_var = mu_var_numer / (count - 1)
        mu_var = torch.where(count > 1, mu_var, torch.zeros_like(mu_var))

        return (avg_post_var + mu_var) / float(d)

    @staticmethod
    def _estimator_per_t(
        idx: torch.Tensor,
        mu_hat_batch: torch.Tensor,
        sigma_t2_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Thin wrapper: compute suff stats then the pure estimator.

        Preserved for any external callers that reference this method directly.
        Internal code now routes through :meth:`_suff_stats_per_t` +
        :meth:`_estimator_from_suff_stats` to support DDP all-reduce.
        """
        suff = SigmaDataBuffer._suff_stats_per_t(idx, mu_hat_batch, sigma_t2_batch)
        return SigmaDataBuffer._estimator_from_suff_stats(suff, mu_hat_batch.shape[1])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_in_range(self, idx: torch.Tensor) -> None:
        # Defensive bounds check — off under compile because ``.any()`` +
        # Python ``if`` is a data-dependent branch dynamo can't trace.
        # If ``idx`` really is out of range the downstream index below
        # will raise ``IndexError`` on its own; we don't need this to fire
        # first.
        if torch.compiler.is_compiling():
            return
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
