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
    transition: torch.nn.Module,
    mode: str,
    *,
    sigma_d2: float = 1.0,
) -> torch.Tensor:
    if not hasattr(transition, "p_k"):
        raise TypeError(
            "Variance probe currently supports transitions with a p_k buffer."
        )
    # The adaptive modes are computed per-row at loss-time, so their owning
    # transition has ``self.p_k = None``. Fall back to ``sigma_tilde`` to size
    # / dtype the static-mode tensors when that's the case.
    if mode == "uniform":
        if transition.p_k is not None:
            p_k = torch.full_like(transition.p_k, 1.0 / float(transition.p_k.numel()))
        else:
            K = int(transition.sigma_tilde.numel())
            p_k = torch.full(
                (K,),
                1.0 / K,
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
            [sigma_d2],
            dtype=transition.sigma_tilde.dtype,
            device=transition.sigma_tilde.device,
        )
        p_k = _adaptive_is_density_meandom(
            transition.sigma_tilde,
            sd2,
            floor=transition.gfloor,
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
            transition.sigma_tilde,
            sd2,
            sg2,
            mh2,
            floor=transition.gfloor,
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
        experiment,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    _freeze_model(model, list(spec.freeze))

    # The probe measures score-net gradient/loss variance at fixed model
    # state — freeze σ_data so transition_kl's EMA update doesn't mutate
    # the buffer across replicas.
    if getattr(model, "sigma_data", None) is not None:
        model.sigma_data.frozen = True

    if not hasattr(model, "transition"):
        raise TypeError("Model is missing transition module for variance probing.")

    modes = sorted({cell.k_sampling_mode for cell in spec.cells})
    transitions: dict[str, torch.nn.Module] = {mode: model.transition for mode in modes}
    # ``transition.p_k`` is ``None`` when the transition's own training-time
    # mode is adaptive (the buffer is computed per-row at loss-time); fall
    # back to ``sigma_tilde`` for the dtype in that case.
    _pk_buf = model.transition.p_k
    _pk_dtype = (
        _pk_buf.dtype if _pk_buf is not None else model.transition.sigma_tilde.dtype
    )
    p_k_by_mode = {
        mode: _p_k_for_mode(model.transition, mode).to(device=device, dtype=_pk_dtype)
        for mode in modes
    }

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
    log.info(
        "Probe run: %d seeds × %d batches × %d replicas × %d cells "
        "(force_per_k=%s, K_bins=%d, split=%r)",
        n_seeds,
        n_batches,
        R,
        n_cells,
        spec.force_per_k,
        int(getattr(model.transition, "num_steps", 0)),
        spec.split,
    )
    log_every = max(1, R // 8)  # ~8 heartbeats per replica loop
    t_run_start = time.perf_counter()

    for seed_i, seed in enumerate(seeds, start=1):
        seed_everything(int(seed))
        batch_iter = iter(loader)
        for batch_idx in range(n_batches):
            batch = next(batch_iter)
            if transform is not None:
                batch = transform(batch, device)
            probe_batch = model.encode_for_probe(batch)
            bs = probe_batch.zs.shape[0] * probe_batch.zs.shape[1]
            d = probe_batch.zs.shape[2]
            sk = int(model.transition.S_k)
            log.info(
                "  seed %d/%d batch %d/%d: %d replicas × %d cells "
                "(bs=%d, d=%d, S_k=%d)",
                seed_i,
                n_seeds,
                batch_idx + 1,
                n_batches,
                R,
                n_cells,
                bs,
                d,
                sk,
            )
            t_batch_start = time.perf_counter()

            for replica in range(R):
                if replica % log_every == 0 and replica > 0:
                    elapsed = time.perf_counter() - t_batch_start
                    eta = elapsed * (R - replica) / replica
                    log.info(
                        "    replica %d/%d  elapsed %.1fs  eta %.1fs",
                        replica,
                        R,
                        elapsed,
                        eta,
                    )
                shared_eps = torch.randn(bs, d, sk, device=device)
                shared_k_idx: dict[str, torch.Tensor] = {}
                for mode, trans in transitions.items():
                    shared_k_idx[mode] = (
                        torch
                        .multinomial(p_k_by_mode[mode], bs * sk, replacement=True)
                        .view(bs, sk)
                        .to(device=device, dtype=torch.long)
                    )

                for cell in spec.cells:
                    trans = transitions[cell.k_sampling_mode]
                    trans.p_k = p_k_by_mode[cell.k_sampling_mode]
                    trans.k_sampling_mode = cell.k_sampling_mode
                    trans.schedule.k_sampling_mode = cell.k_sampling_mode
                    trans.zero_grad(set_to_none=True)
                    out = trans.transition_kl(
                        **probe_batch.as_kwargs(),
                        sigma_data=model.sigma_data,
                        mc_override={
                            "eps": shared_eps,
                            "k_idx": shared_k_idx[cell.k_sampling_mode],
                            "objective": cell.objective,
                        },
                        return_per_sample=True,
                    )
                    lp = out["L_p"]
                    lp.backward()
                    g = _grad_vector(trans.diffmodel)
                    g_norm = float(g.norm().item())
                    cell_key = (cell.objective, cell.k_sampling_mode)
                    cell_grads[cell_key].append(g.cpu().numpy())
                    per_sample = out["L_p_per_sample"].detach().cpu().numpy()
                    # Mean-per-batch loss: out["L_p"] (== lp) is summed over the
                    # B·S batch samples, so divide by the sample count to get a
                    # per-sample-mean scale for the loss_var estimator.
                    l_p_scalar = float(np.mean(per_sample))
                    for i, v in enumerate(per_sample):
                        rows.append({
                            "seed": int(seed),
                            "batch_idx": int(batch_idx),
                            "replica": int(replica),
                            "objective": cell.objective,
                            "k_sampling_mode": cell.k_sampling_mode,
                            "kind": "replica",
                            "k_idx": -1,
                            "sample_idx": int(i),
                            "L_p": float(v),
                            "L_p_scalar": l_p_scalar,
                            "grad_norm": g_norm,
                        })

            if spec.force_per_k:
                k_max = int(getattr(model.transition, "num_steps", 0))
                log.info(
                    "    force_per_k: %d steps × %d cells",
                    k_max,
                    n_cells,
                )
                t_pk_start = time.perf_counter()
                for k in range(k_max):
                    if k > 0 and k % max(1, k_max // 4) == 0:
                        log.info(
                            "      force_per_k step %d/%d  elapsed %.1fs",
                            k,
                            k_max,
                            time.perf_counter() - t_pk_start,
                        )
                    forced_idx = torch.full(
                        (bs, sk), k, device=device, dtype=torch.long
                    )
                    forced_eps = torch.randn(bs, d, sk, device=device)
                    for cell in spec.cells:
                        trans = transitions[cell.k_sampling_mode]
                        trans.zero_grad(set_to_none=True)
                        out = trans.transition_kl(
                            **probe_batch.as_kwargs(),
                            sigma_data=model.sigma_data,
                            mc_override={
                                "eps": forced_eps,
                                "k_idx": forced_idx,
                                "objective": cell.objective,
                            },
                            return_per_sample=True,
                        )
                        out["L_p"].backward()
                        g = _grad_vector(trans.diffmodel)
                        g_norm = float(g.norm().item())
                        per_k_cell_grads[
                            cell.objective, cell.k_sampling_mode, int(k)
                        ].append(g.cpu().numpy())
                        per_sample = out["L_p_per_sample"].detach().cpu().numpy()
                        # Mean-per-batch loss (see the replica branch above):
                        # out["L_p"] is summed over the B·S batch samples.
                        l_p_mean = float(np.mean(per_sample))
                        rows.append({
                            "seed": int(seed),
                            "batch_idx": int(batch_idx),
                            "replica": -1,
                            "objective": cell.objective,
                            "k_sampling_mode": cell.k_sampling_mode,
                            "kind": "forced_k",
                            "k_idx": int(k),
                            "sample_idx": -1,
                            "L_p": l_p_mean,
                            "L_p_scalar": l_p_mean,
                            "grad_norm": g_norm,
                        })

    log.info(
        "Probe loop done in %.1fs — %d rows across %d cells",
        time.perf_counter() - t_run_start,
        len(rows),
        len(cell_grads),
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

    summary = {"cells": summary_cells, "per_k_grad_var": per_k_grad_var}
    return rows, summary, transitions
