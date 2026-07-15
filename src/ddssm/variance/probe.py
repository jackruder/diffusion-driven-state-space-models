"""Core variance-probe loop used by :mod:`ddssm.variance.runner`."""

from __future__ import annotations

import time
import random
from typing import Any
import logging
from collections import defaultdict

import numpy as np
import torch

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch (incl. CUDA) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _p_k_for_mode(
    transition: torch.nn.Module, mode: str, *, sigma_d2: float = 1.0,
) -> torch.Tensor:
    if not hasattr(transition, "p_k"):
        raise TypeError("Variance probe currently supports transitions with a p_k buffer.")
    # The adaptive modes are computed per-row at loss-time, so their owning
    # transition has ``self.p_k = None``. Fall back to ``sigma_tilde`` to size
    # / dtype the static-mode tensors when that's the case.
    if mode == "uniform":
        if transition.p_k is not None:
            p_k = torch.full_like(transition.p_k, 1.0 / float(transition.p_k.numel()))
        else:
            K = int(transition.sigma_tilde.numel())
            p_k = torch.full(
                (K,), 1.0 / K,
                dtype=transition.sigma_tilde.dtype,
                device=transition.sigma_tilde.device,
            )
    elif mode == "lsgm_is":
        eps = torch.finfo(transition.beta.dtype).eps
        proposal = transition.beta / (1.0 - transition.alpha.pow(2)).clamp_min(eps)
        proposal = proposal.clamp_min(float(getattr(transition, "gfloor", 1e-12)))
        gamma = float(getattr(transition, "gamma", 1.0))
        if gamma != 1.0:
            proposal = proposal.pow(gamma)
        p_k = proposal / proposal.sum().clamp_min(eps)
    elif mode == "adaptive_is":
        from ddssm.model.transitions.diffusion import _adaptive_is_density_meandom

        sd2 = torch.tensor(
            [sigma_d2], dtype=transition.sigma_tilde.dtype,
            device=transition.sigma_tilde.device,
        )
        p_k = _adaptive_is_density_meandom(
            transition.sigma_tilde, sd2, floor=transition.gfloor,
            # Match training: the loss clips the normalized density at
            # p_k_clip; the probe must measure the same estimator.
            p_k_clip=getattr(transition, "p_k_clip", 0.0),
        ).squeeze(0)
    elif mode == "adaptive_is_full":
        from ddssm.model.transitions.diffusion import _adaptive_is_density_full

        # TODO: pull σ² and μ̂² from real batch stats in a future iteration —
        # for now the probe is diagnostic-only and uses unit defaults.
        dtype = transition.sigma_tilde.dtype
        device = transition.sigma_tilde.device
        sd2 = torch.tensor([sigma_d2], dtype=dtype, device=device)
        sg2 = torch.tensor([1.0], dtype=dtype, device=device)
        mh2 = torch.tensor([1.0], dtype=dtype, device=device)
        p_k = _adaptive_is_density_full(
            transition.sigma_tilde, sd2, sg2, mh2, floor=transition.gfloor,
            # Match training: same p_k_clip as the loss-time density.
            p_k_clip=getattr(transition, "p_k_clip", 0.0),
        ).squeeze(0)
    else:
        raise ValueError(f"Unsupported k_sampling_mode {mode!r}.")
    return p_k


def _freeze_model(model: torch.nn.Module, freeze: list[str]) -> None:
    frozen = set(freeze)
    for name, module in model.named_children():
        req = name not in frozen
        for param in module.parameters():
            param.requires_grad = req
    if hasattr(model, "transition") and hasattr(model.transition, "diffmodel"):
        for p in model.transition.diffmodel.parameters():
            p.requires_grad = True


def _grad_vector(module: torch.nn.Module) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for p in module.parameters():
        if p.grad is not None:
            chunks.append(p.grad.detach().reshape(-1))
    if not chunks:
        return torch.zeros(1)
    return torch.cat(chunks)


def _encode_seeded_batches(model, loader, transform, seeds, n_batches, device):
    """Encode all (seed, batch_idx) probe_batches and concatenate along B.

    Returns:
        combined: a :class:`ProbeBatch` with tensors concatenated along dim 0
            covering every (seed, batch) pair.
        slot_metadata: length ``n_seeds*n_batches`` list of
            ``(seed_idx, seed_value, batch_idx)`` tuples, one per source
            batch (= one "sample slot" for gradient / row emission).
        slot_masks: list of boolean tensors on ``device`` selecting each
            slot's rows in the flat ``B*S`` row layout used by
            :meth:`~ddssm.model.transitions.diffusion.DiffusionTransition._esm_chunk_loss`
            (row layout: ``i -> (b=i//S, s=i%S)``).
    """
    from ddssm.model.dssd import ProbeBatch

    zs_list, mus_list, lv_list = [], [], []
    lq_list, te_list, cov_list = [], [], []
    slot_metadata: list[tuple[int, int, int]] = []
    B_per_slot: list[int] = []
    for seed_idx, seed in enumerate(seeds):
        seed_everything(int(seed))
        batch_iter = iter(loader)
        for batch_idx in range(int(n_batches)):
            batch = next(batch_iter)
            if transform is not None:
                batch = transform(batch, device)
            probe_batch = model.encode_for_probe(batch)
            zs_list.append(probe_batch.zs)
            mus_list.append(probe_batch.enc_stats["mus"])
            lv_list.append(probe_batch.enc_stats["logvars"])
            lq_list.append(probe_batch.logq_paths)
            te_list.append(probe_batch.time_embed)
            cov_list.append(probe_batch.covariates)
            B_per_slot.append(int(probe_batch.zs.shape[0]))
            slot_metadata.append((seed_idx, int(seed), batch_idx))

    if not zs_list:
        raise ValueError("No probe batches produced — check spec.seeds and spec.n_batches.")

    S = zs_list[0].shape[1]
    combined = ProbeBatch(
        zs=torch.cat(zs_list, dim=0),
        logq_paths=torch.cat(lq_list, dim=0),
        enc_stats={
            "mus": torch.cat(mus_list, dim=0),
            "logvars": torch.cat(lv_list, dim=0),
        },
        time_embed=torch.cat(te_list, dim=0),
        covariates=(
            torch.cat(cov_list, dim=0) if cov_list[0] is not None else None
        ),
    )

    # Build one boolean mask per slot over the combined B*S row layout.
    # A slot owns a contiguous range of B indices; its rows in the flat
    # (b, s) layout are b*S + s for b in that range, s in [0, S).
    slot_masks: list[torch.Tensor] = []
    b_start = 0
    for B_here in B_per_slot:
        row_indices = np.arange(b_start * S, (b_start + B_here) * S, dtype=np.int64)
        mask = torch.zeros(combined.zs.shape[0] * S, dtype=torch.bool, device=device)
        mask[torch.from_numpy(row_indices).to(device)] = True
        slot_masks.append(mask)
        b_start += B_here
    return combined, slot_metadata, slot_masks


def _replay_backward_per_slot(
    *,
    trans: torch.nn.Module,
    per_sample: torch.Tensor,
    slot_masks: list[torch.Tensor],
    slot_metadata: list[tuple[int, int, int]],
    rows: list[dict[str, Any]],
    grad_bucket: list[np.ndarray] | None,
    per_k_bucket: dict[tuple[str, str, int], list[np.ndarray]] | None,
    k_idx: int,
    objective: str,
    k_sampling_mode: str,
    replica: int,
    kind: str,
) -> None:
    """Backward-per-slot: one gradient per (seed, batch) slot from one forward.

    ``per_sample`` traces back through the (already-run) fwd graph; we
    ``.backward(retain_graph=True)`` on each slot's summed loss to extract
    an independent gradient sample for that slot, matching the pre-fold
    "one gradient per (seed, batch)" contract expected by ``metric_grad_var``
    / ``per_k_grad_var``. Retaining the graph is essential — otherwise
    subsequent slots have nothing to backprop through.
    """
    n_slots = len(slot_masks)
    for j in range(n_slots):
        trans.zero_grad(set_to_none=True)
        slot_loss = per_sample[slot_masks[j]].sum()
        slot_loss.backward(retain_graph=(j < n_slots - 1))
        g = _grad_vector(trans.diffmodel)
        g_norm = float(g.norm().item())
        g_np = g.cpu().numpy()
        if grad_bucket is not None:
            grad_bucket.append(g_np)
        if per_k_bucket is not None:
            per_k_bucket[objective, k_sampling_mode, int(k_idx)].append(g_np)

        _seed_idx, seed_value, batch_idx = slot_metadata[j]
        slot_ps = per_sample[slot_masks[j]].detach().cpu().numpy()
        l_p_mean = float(np.mean(slot_ps))
        if kind == "forced_k":
            rows.append({
                "seed": int(seed_value),
                "batch_idx": int(batch_idx),
                "replica": int(replica),
                "objective": objective,
                "k_sampling_mode": k_sampling_mode,
                "kind": kind,
                "k_idx": int(k_idx),
                "sample_idx": -1,
                "L_p": l_p_mean,
                "L_p_scalar": l_p_mean,
                "grad_norm": g_norm,
            })
        else:  # "replica"
            for i, v in enumerate(slot_ps):
                rows.append({
                    "seed": int(seed_value),
                    "batch_idx": int(batch_idx),
                    "replica": int(replica),
                    "objective": objective,
                    "k_sampling_mode": k_sampling_mode,
                    "kind": kind,
                    "k_idx": int(k_idx),
                    "sample_idx": int(i),
                    "L_p": float(v),
                    "L_p_scalar": l_p_mean,
                    "grad_norm": g_norm,
                })


def run_probe(
    experiment,
    spec,
    *,
    device: torch.device,
    checkpoint_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, torch.nn.Module]]:
    """Run the variance-probe loop and collect per-sample rows and summaries.

    Loads a checkpoint, freezes the encoder/decoder/embed submodules, then
    for each (seed, batch) draws ``R`` replicas — measuring the transition
    score-net loss and gradient under each probe cell with shared noise and
    step indices so cells are paired. When ``spec.force_per_k`` is set it
    additionally sweeps every diffusion step ``k`` to build the per-τ curves.

    Args:
        experiment: The built :class:`~ddssm.experiment.Experiment`.
        spec: The driving :class:`~ddssm.variance.runner.ProbeSpec`.
        device: Torch device for the forward/backward passes.
        checkpoint_path: Checkpoint to load (EMA shadows by default).

    Returns:
        A ``(rows, summary, transitions)`` triple: per-sample row dicts,
        the per-cell / per-k summary, and the transition modules keyed by
        k-sampling mode.

    Raises:
        TypeError: If the model lacks a ``transition`` module or the
            transition has no ``p_k`` buffer.
        ValueError: If the probed split has no loader.
    """
    from ddssm.training.checkpoint import prepare_model

    # ``prepare_model`` defaults to ``load_ema=True`` — the probe measures
    # the sampling-path EMA shadows, matching training-time sampling
    # (ADR-0005).
    model = prepare_model(
        experiment, checkpoint_path=checkpoint_path, device=device,
    )
    _freeze_model(model, list(spec.freeze))

    # The probe measures score-net gradient/loss variance at fixed model
    # state — freeze σ_data so transition_kl's EMA update doesn't mutate
    # the buffer across replicas.
    if getattr(model, "sigma_data", None) is not None:
        model.sigma_data.frozen = True

    if not hasattr(model, "transition"):
        raise TypeError("Model is missing transition module for variance probing.")

    # The probe generates one shared (eps, k_idx) pair sized ``bs = B*S`` per
    # replica and hands it to ``transition_kl`` as a single ``mc_override`` —
    # that only lines up with the model's internal per-chunk row count when
    # each window chunk covers exactly one timestep. Models trained with
    # ``time_chunk_size > 1`` (e.g. the CSDI-style batching in
    # ``experiments/gluonts_forecast``) batch several timesteps into one
    # score-net call, giving chunks of ``N = B*S*chunk_len`` rows and a shape
    # mismatch in ``_vp_precondition``. Force single-timestep chunking for the
    # probe only — this doesn't touch the score net's weights, just how many
    # timesteps are batched into each call, so loss/gradient values are
    # unaffected.
    schedule = getattr(model.transition, "schedule", None)
    if schedule is not None and getattr(schedule, "time_chunk_size", None) not in (None, 1):
        log.info(
            "Overriding transition.schedule.time_chunk_size %r -> 1 for probing "
            "(keeps per-chunk row count at B*S to match the probe's shared "
            "mc_override).", schedule.time_chunk_size,
        )
        schedule.time_chunk_size = 1

    modes = sorted({cell.k_sampling_mode for cell in spec.cells})
    transitions: dict[str, torch.nn.Module] = {mode: model.transition for mode in modes}
    # ``transition.p_k`` is ``None`` when the transition's own training-time
    # mode is adaptive (the buffer is computed per-row at loss-time); fall
    # back to ``sigma_tilde`` for the dtype in that case.
    _pk_buf = model.transition.p_k
    _pk_dtype = _pk_buf.dtype if _pk_buf is not None else model.transition.sigma_tilde.dtype
    p_k_by_mode = {
        mode: _p_k_for_mode(model.transition, mode).to(device=device, dtype=_pk_dtype)
        for mode in modes
    }

    # The cell loops below mutate mode state on the (single, shared)
    # ``model.transition``; snapshot it so the probe can restore the
    # training-time configuration afterwards instead of leaking whichever
    # cell happened to run last.
    _orig_p_k = model.transition.p_k
    _orig_mode = model.transition.k_sampling_mode
    _orig_sched_mode = model.transition.schedule.k_sampling_mode

    loader = experiment.data.loader(spec.split)
    if loader is None:
        raise ValueError("Variance probe requires a non-empty loader.")
    transform = experiment.data.batch_transform

    rows: list[dict[str, Any]] = []
    cell_grads: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    # per-(cell, k) grad vectors collected during the force_per_k loop —
    # variance is taken across (seed, batch) per (cell, k).
    per_k_cell_grads: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)

    seeds = list(spec.seeds)
    n_seeds = len(seeds)
    n_batches = int(spec.n_batches)
    R = int(spec.R)
    n_cells = len(spec.cells)

    # Fold seeds × batches into a single combined ``probe_batch`` — each seed
    # still owns a disjoint slice of the batch dim (so its rows are
    # statistically independent from the other seeds'), but a single fwd
    # through the score-net now covers ALL (seed, batch) pairs. Backward is
    # replayed per (seed, batch) slot with ``retain_graph=True`` to recover
    # per-slot gradients (one gradient sample per slot, same as before) — this
    # trades one shared fwd for n_slots bwds instead of n_slots × (fwd + bwd).
    combined, slot_metadata, slot_masks = _encode_seeded_batches(
        model, loader, transform, seeds, n_batches, device,
    )
    n_slots = len(slot_metadata)  # == n_seeds * n_batches
    combined_bs = combined.zs.shape[0] * combined.zs.shape[1]
    d = combined.zs.shape[2]
    sk = int(model.transition.S_k)
    log.info(
        "Probe run: %d seeds × %d batches × %d replicas × %d cells "
        "(force_per_k=%s, K_bins=%d, split=%r, combined_bs=%d, "
        "n_slots=%d [seed-folded])",
        n_seeds, n_batches, R, n_cells,
        spec.force_per_k, int(getattr(model.transition, "num_steps", 0)),
        spec.split, combined_bs, n_slots,
    )
    log_every = max(1, R // 8)   # ~8 heartbeats per replica loop
    t_run_start = time.perf_counter()
    t_batch_start = time.perf_counter()

    for replica in range(R):
        if replica % log_every == 0 and replica > 0:
            elapsed = time.perf_counter() - t_batch_start
            eta = elapsed * (R - replica) / replica
            log.info(
                "  replica %d/%d  elapsed %.1fs  eta %.1fs",
                replica, R, elapsed, eta,
            )
        shared_eps = torch.randn(combined_bs, d, sk, device=device)
        shared_k_idx: dict[str, torch.Tensor] = {}
        for mode in transitions:
            shared_k_idx[mode] = torch.multinomial(
                p_k_by_mode[mode], combined_bs * sk, replacement=True
            ).view(combined_bs, sk).to(device=device, dtype=torch.long)

        for cell in spec.cells:
            trans = transitions[cell.k_sampling_mode]
            trans.p_k = p_k_by_mode[cell.k_sampling_mode]
            trans.k_sampling_mode = cell.k_sampling_mode
            trans.schedule.k_sampling_mode = cell.k_sampling_mode
            trans.zero_grad(set_to_none=True)
            out = trans.transition_kl(
                **combined.as_kwargs(),
                sigma_data=model.sigma_data,
                mc_override={
                    "eps": shared_eps,
                    "k_idx": shared_k_idx[cell.k_sampling_mode],
                    "objective": cell.objective,
                },
                return_per_sample=True,
            )
            per_sample = out["L_p_per_sample"]  # (combined_bs,)
            cell_key = (cell.objective, cell.k_sampling_mode)
            _replay_backward_per_slot(
                trans=trans, per_sample=per_sample, slot_masks=slot_masks,
                slot_metadata=slot_metadata, rows=rows,
                grad_bucket=cell_grads[cell_key],
                per_k_bucket=None, k_idx=-1,
                objective=cell.objective, k_sampling_mode=cell.k_sampling_mode,
                replica=replica, kind="replica",
            )

    if spec.force_per_k:
        k_max = int(getattr(model.transition, "num_steps", 0))
        log.info("  force_per_k: %d steps × %d cells", k_max, n_cells)
        t_pk_start = time.perf_counter()
        for k in range(k_max):
            if k > 0 and k % max(1, k_max // 4) == 0:
                log.info(
                    "    force_per_k step %d/%d  elapsed %.1fs",
                    k, k_max, time.perf_counter() - t_pk_start,
                )
            forced_idx = torch.full((combined_bs, sk), k, device=device, dtype=torch.long)
            forced_eps = torch.randn(combined_bs, d, sk, device=device)
            for cell in spec.cells:
                trans = transitions[cell.k_sampling_mode]
                trans.zero_grad(set_to_none=True)
                out = trans.transition_kl(
                    **combined.as_kwargs(),
                    sigma_data=model.sigma_data,
                    mc_override={
                        "eps": forced_eps,
                        "k_idx": forced_idx,
                        "objective": cell.objective,
                    },
                    return_per_sample=True,
                )
                per_sample = out["L_p_per_sample"]  # (combined_bs,)
                _replay_backward_per_slot(
                    trans=trans, per_sample=per_sample, slot_masks=slot_masks,
                    slot_metadata=slot_metadata, rows=rows,
                    grad_bucket=None,
                    per_k_bucket=per_k_cell_grads,
                    k_idx=int(k),
                    objective=cell.objective, k_sampling_mode=cell.k_sampling_mode,
                    replica=-1, kind="forced_k",
                )

    log.info(
        "Probe loop done in %.1fs — %d rows across %d cells",
        time.perf_counter() - t_run_start, len(rows), len(cell_grads),
    )

    summary_cells: dict[str, dict[str, float]] = {}
    for (objective, mode), grad_list in cell_grads.items():
        key = f"{objective}:{mode}"
        grad_arr = np.stack(grad_list, axis=0)
        summary_cells[key] = {
            "grad_norm_mean": float(np.linalg.norm(grad_arr, axis=1).mean()),
            "grad_variance": float(np.var(grad_arr, axis=0).mean()),
        }

    # Per-(cell, k) gradient variance from the force_per_k loop. Variance
    # is taken across the (seed, batch) samples per (cell, k); with only
    # n_seeds × n_batches samples the estimate is noisy — bump
    # ``seeds`` for a tighter read.
    per_k_grad_var: dict[str, dict[int, float]] = {}
    for (objective, mode, k), grad_list in per_k_cell_grads.items():
        cell_key = f"{objective}:{mode}"
        if len(grad_list) < 2:
            # Variance is undefined with a single sample; record NaN so
            # plots / metrics surface the underspecification instead of
            # silently showing 0.
            per_k_grad_var.setdefault(cell_key, {})[k] = float("nan")
            continue
        grad_arr = np.stack(grad_list, axis=0)
        per_k_grad_var.setdefault(cell_key, {})[k] = float(
            np.var(grad_arr, axis=0).mean()
        )

    # Restore the transition's training-time mode config so the probe
    # doesn't leak whichever cell happened to run last onto the shared
    # ``model.transition``.
    model.transition.p_k = _orig_p_k
    model.transition.k_sampling_mode = _orig_mode
    model.transition.schedule.k_sampling_mode = _orig_sched_mode

    summary = {"cells": summary_cells, "per_k_grad_var": per_k_grad_var}
    return rows, summary, transitions
