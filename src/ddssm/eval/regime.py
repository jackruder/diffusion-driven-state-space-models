"""System-agnostic regime-switching metrics for metastable dynamics.

A *regime* is a discrete label imposed on a continuous trajectory by
thresholding one observed channel — the two Lorenz lobes via ``sign(x)``,
the two wells of a double-well potential via ``sign(z)``, etc. The label
is a diagnostic coarse-graining, not part of any model: forecast samples
and the true continuation pass through the same labelling function, so
the comparison is well-defined for any system with metastable switching.
Dataset-specific choices (which channel, how wide a deadband) enter only
through ``EvalSpec.kwargs``; nothing here is Lorenz-specific.

Per-sequence switch timing is chaos-limited — beyond ~1-2 Lyapunov times
even a perfect model cannot pin the switch time — so the point-error key
(``regime_first_switch_mae``) measures sharpness on short horizons, while
the calibration (``regime_first_switch_coverage_80``) and climatology
(``regime_residence_jsd``) keys are the ones a perfect model maxes out.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ddssm.eval.metrics import (
    EvalContext,
    _hist_mass,
    _jsd_discrete,
    register_metric,
)


def regime_labels(
    x: torch.Tensor,
    *,
    threshold: float = 0.0,
    deadband: float = 0.0,
    initial: torch.Tensor | None = None,
) -> torch.Tensor:
    """Debounced ±1 regime labels along the last (time) dimension.

    A point is firmly ``+1`` when ``x > threshold + deadband`` and ``-1``
    when ``x < threshold - deadband``. Points inside the band inherit the
    last firm label (a debounce, not a third regime); leading in-band
    points fall back to ``initial`` when given, else to the first firm
    label that follows (backfill). Rows that never leave the band come
    back all-zero and are excluded by the metric.

    Args:
        x: ``(..., T)`` channel values.
        threshold: Regime boundary in the same units as ``x``.
        deadband: Half-width of the no-commit band around ``threshold``.
        initial: ``(...)`` reference labels for leading in-band points
            (e.g. the last firm label of the conditioning context).

    Returns:
        ``(..., T)`` int8 labels in ``{-1, 0, +1}``.
    """
    firm = torch.zeros_like(x, dtype=torch.int8)
    firm[x > threshold + deadband] = 1
    firm[x < threshold - deadband] = -1
    out = firm.clone()
    T = out.shape[-1]
    prev = (
        initial.to(torch.int8)
        if initial is not None
        else torch.zeros_like(out[..., 0])
    )
    for t in range(T):
        cur = torch.where(out[..., t] == 0, prev, out[..., t])
        out[..., t] = cur
        prev = cur
    if initial is None:
        # Zeros now remain only before the first firm label; backfill them.
        nxt = torch.zeros_like(out[..., 0])
        for t in range(T - 1, -1, -1):
            cur = torch.where(out[..., t] == 0, nxt, out[..., t])
            out[..., t] = cur
            nxt = cur
    return out


def first_switch_times(
    labels: torch.Tensor, ref: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """First index whose firm label differs from ``ref``, censored at T.

    Args:
        labels: ``(..., T)`` debounced labels.
        ref: ``(...)`` reference label each row is measured against.

    Returns:
        ``(times, switched)``: int64 first-switch indices (``T`` where the
        row never switches) and the bool switched mask.
    """
    diff = (labels != ref.unsqueeze(-1)) & (labels != 0)
    switched = diff.any(dim=-1)
    T = labels.shape[-1]
    first = torch.argmax(diff.to(torch.int8), dim=-1)
    times = torch.where(switched, first, torch.full_like(first, T))
    return times, switched


def run_lengths(labels: np.ndarray, *, drop_censored: bool = True) -> np.ndarray:
    """Constant-label run lengths pooled from a ``(rows, T)`` label array.

    All-zero rows (never left the deadband) contribute nothing. With
    ``drop_censored`` the first and last run of each row — which touch a
    window edge so their true length is unknown — are dropped; a row that
    is one single run therefore contributes nothing.

    Returns:
        1-D int64 array of run lengths.
    """
    rows, T = labels.shape
    out: list[int] = []
    for r in range(rows):
        lab = labels[r]
        if not (lab != 0).any():
            continue
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        bounds = np.concatenate(([0], change, [T]))
        lens = np.diff(bounds)
        if drop_censored:
            lens = lens[1:-1]
        out.extend(int(v) for v in lens)
    return np.asarray(out, dtype=np.int64)


@register_metric("regime")
def eval_regime(
    ctx: EvalContext,
    *,
    channel: int = 0,
    threshold: float = 0.0,
    deadband: float = 0.0,
    k_steps: tuple[int, ...] = (4, 8, 16),
    n_duration_bins: int = 16,
    max_batches: int | None = None,
) -> Dict[str, Any]:
    """Regime-switching forecast quality on one thresholded channel.

    Walks the loader once, calls ``model.forecast`` per batch, labels the
    forecast samples and the true continuation with the same debounced
    thresholding, and scores three things:

    * **Residence accuracy** — ``regime_acc_per_t`` (per-step label
      agreement averaged over sequences × samples) and
      ``regime_acc_at_{k}`` (fraction of rollouts whose first ``k`` steps
      all carry the correct label), with the persistence baseline
      ``regime_persistence_acc_at_{k}`` (always predict the context's
      last regime) alongside.
    * **First-switch timing** — vs the context's last firm label:
      ``regime_first_switch_mae`` (truth vs per-sequence median sample
      switch time; censored rollouts count at the horizon),
      ``regime_switch_recall``, ``regime_first_switch_coverage_80``
      (truth inside the samples' [q10, q90]), ``regime_false_switch_rate``
      (samples switching when the truth never does), and the support size
      ``regime_n_truth_switches``.
    * **Residence climatology** — ``regime_residence_jsd``: JSD between
      run-length histograms pooled from forecast samples and from the
      full ground-truth sequences (edge-censored runs dropped on both
      sides; lengths clipped at ``n_duration_bins``).

    Args:
        channel: Observed-data channel carrying the regime signal.
        threshold: Regime boundary value on that channel.
        deadband: No-commit half-width absorbing noise chatter near the
            boundary (e.g. the observation-noise scale).
        k_steps: Horizons for the all-correct residence accuracies.
        n_duration_bins: Unit-width histogram bins for residence times.
        max_batches: Optional cap on evaluated batches.
    """
    if ctx.model is None or ctx.loader is None or ctx.T_split is None:
        raise ValueError("regime requires model, loader, and T_split.")
    model, device = ctx.model, ctx.device
    L1 = int(ctx.T_split)
    transform = ctx.batch_transform
    ks = [int(k) for k in k_steps]

    n_seq = 0
    n_ambiguous = 0
    num_samples = int(ctx.num_samples)
    agree_sum: torch.Tensor | None = None  # (L2,) float64
    agree_rows = 0
    acc_at = dict.fromkeys(ks, 0)
    persist_at = dict.fromkeys(ks, 0)
    persist_rows = 0
    t_true_all: list[np.ndarray] = []
    t_pred_all: list[np.ndarray] = []
    switched_true_all: list[np.ndarray] = []
    switched_pred_all: list[np.ndarray] = []
    pred_run_buf: list[np.ndarray] = []
    true_run_buf: list[np.ndarray] = []

    with torch.no_grad():
        for i, batch in enumerate(ctx.loader):
            if max_batches is not None and i >= int(max_batches):
                break
            if transform is not None:
                batch = transform(batch, device)
            else:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            obs = batch["observed_data"]
            mask = batch["observation_mask"]
            past_time = batch["timepoints"][:, :L1]
            future_time = batch["timepoints"][:, L1:]

            out = model.forecast(
                x_hist=obs[..., :L1],
                x_mask=mask[..., :L1],
                past_time=past_time,
                future_time=future_time,
                num_samples=num_samples,
            )
            pred = out["pred_samples"][:, :, channel, :].cpu()  # (B, S, L2)
            x_full = obs[:, channel, :].cpu()  # (B, T)
            B, S, L2 = pred.shape

            # Reference regime: last debounced context label. Rows still in
            # the deadband at the boundary are ambiguous and excluded.
            ctx_labels = regime_labels(
                x_full[:, :L1], threshold=threshold, deadband=deadband
            )
            ref = ctx_labels[:, -1]
            valid = ref != 0
            n_seq += B
            n_ambiguous += int((~valid).sum())
            if not valid.any():
                continue

            ref_v = ref[valid]
            fut_labels = regime_labels(
                x_full[valid, L1:], threshold=threshold, deadband=deadband,
                initial=ref_v,
            )
            pred_labels = regime_labels(
                pred[valid], threshold=threshold, deadband=deadband,
                initial=ref_v.unsqueeze(1).expand(-1, S),
            )

            agree = (pred_labels == fut_labels.unsqueeze(1)).to(torch.float64)
            if agree_sum is None:
                agree_sum = torch.zeros(L2, dtype=torch.float64)
            agree_sum += agree.sum(dim=(0, 1))
            agree_rows += agree.shape[0] * S
            for k in ks:
                kk = min(k, L2)
                acc_at[k] += int(agree[:, :, :kk].all(dim=-1).sum())
                persist_at[k] += int(
                    (fut_labels[:, :kk] == ref_v.unsqueeze(-1)).all(dim=-1).sum()
                )
            persist_rows += int(valid.sum())

            t_true, sw_true = first_switch_times(fut_labels, ref_v)
            t_pred, sw_pred = first_switch_times(
                pred_labels, ref_v.unsqueeze(1).expand(-1, S)
            )
            t_true_all.append(t_true.numpy())
            t_pred_all.append(t_pred.numpy())
            switched_true_all.append(sw_true.numpy())
            switched_pred_all.append(sw_pred.numpy())

            pred_run_buf.append(pred_labels.reshape(-1, L2).numpy())
            full_labels = regime_labels(
                x_full, threshold=threshold, deadband=deadband
            )
            true_run_buf.append(full_labels.numpy())

    if agree_sum is None or agree_rows == 0:
        return {"regime_available": False, "regime_n_sequences": n_seq}

    L2 = int(agree_sum.shape[0])
    result: Dict[str, Any] = {
        "regime_n_sequences": n_seq,
        "regime_n_ambiguous_ref": n_ambiguous,
        "regime_num_samples": num_samples,
        "regime_acc_per_t": (agree_sum / agree_rows).tolist(),
    }
    for k in ks:
        result[f"regime_acc_at_{k}"] = acc_at[k] / agree_rows
        result[f"regime_persistence_acc_at_{k}"] = persist_at[k] / persist_rows

    t_true = np.concatenate(t_true_all)  # (N,)
    t_pred = np.concatenate(t_pred_all)  # (N, S)
    sw_true = np.concatenate(switched_true_all)
    sw_pred = np.concatenate(switched_pred_all)
    n_switch = int(sw_true.sum())
    result["regime_n_truth_switches"] = n_switch
    if n_switch > 0:
        tt = t_true[sw_true].astype(np.float64)
        tp = t_pred[sw_true].astype(np.float64)  # censored samples count at L2
        med = np.median(tp, axis=1)
        q10 = np.quantile(tp, 0.1, axis=1)
        q90 = np.quantile(tp, 0.9, axis=1)
        result["regime_first_switch_mae"] = float(np.abs(med - tt).mean())
        result["regime_switch_recall"] = float(
            sw_pred[sw_true].any(axis=1).mean()
        )
        result["regime_first_switch_coverage_80"] = float(
            ((tt >= q10) & (tt <= q90)).mean()
        )
    if n_switch < sw_true.size:
        result["regime_false_switch_rate"] = float(
            sw_pred[~sw_true].mean()
        )

    pred_runs = run_lengths(np.concatenate(pred_run_buf, axis=0))
    true_runs = run_lengths(np.concatenate(true_run_buf, axis=0))
    result["regime_n_pred_runs"] = int(pred_runs.size)
    result["regime_n_truth_runs"] = int(true_runs.size)
    if pred_runs.size > 0 and true_runs.size > 0:
        edges = np.arange(0.5, n_duration_bins + 1.5, 1.0)
        p = _hist_mass(np.clip(pred_runs, 1, n_duration_bins), edges)
        q = _hist_mass(np.clip(true_runs, 1, n_duration_bins), edges)
        result["regime_residence_jsd"] = _jsd_discrete(p, q)
        result["regime_residence_pred_mean"] = float(pred_runs.mean())
        result["regime_residence_truth_mean"] = float(true_runs.mean())

    return result
