"""Core variance-probe loop used by :mod:`ddssm.variance.runner`."""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_loader(experiment, split: str):
    if split == "train":
        return experiment.data.train_loader()
    if split == "val":
        return experiment.data.val_loader()
    if split == "test":
        return experiment.data.test_loader()
    raise ValueError(f"Unknown variance split: {split!r}")


def _retarget_sampling_mode(transition: torch.nn.Module, mode: str) -> torch.nn.Module:
    clone = copy.deepcopy(transition)
    if not hasattr(clone, "p_k"):
        raise TypeError("Variance probe currently supports transitions with a p_k buffer.")
    if mode == "uniform":
        p_k = torch.full_like(clone.p_k, 1.0 / float(clone.p_k.numel()))
    elif mode == "lsgm_is":
        eps = torch.finfo(clone.beta.dtype).eps
        proposal = clone.beta / (1.0 - clone.alpha.pow(2)).clamp_min(eps)
        proposal = proposal.clamp_min(float(getattr(clone, "gfloor", 1e-12)))
        gamma = float(getattr(clone, "gamma", 1.0))
        if gamma != 1.0:
            proposal = proposal.pow(gamma)
        p_k = proposal / proposal.sum().clamp_min(eps)
    else:
        raise ValueError(f"Unsupported k_sampling_mode {mode!r}.")
    clone.k_sampling_mode = mode
    clone.p_k = p_k.to(clone.p_k.device, clone.p_k.dtype)
    clone.schedule.k_sampling_mode = mode
    return clone


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
    model = experiment.model.to(device)
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        model.load_state_dict(state, strict=True)
    model.eval()
    _freeze_model(model, list(spec.freeze))

    if not hasattr(model, "transition"):
        raise TypeError("Model is missing transition module for variance probing.")

    modes = sorted({cell.k_sampling_mode for cell in spec.cells})
    transitions: dict[str, torch.nn.Module] = {}
    for mode in modes:
        if mode == getattr(model.transition, "k_sampling_mode", None):
            transitions[mode] = model.transition
        else:
            twin = _retarget_sampling_mode(model.transition, mode)
            twin.load_state_dict(model.transition.state_dict(), strict=False)
            twin.diffmodel = model.transition.diffmodel
            twin.embed_layer = model.transition.embed_layer
            transitions[mode] = twin.to(device)

    loader = _select_loader(experiment, spec.split)
    if loader is None:
        raise ValueError("Variance probe requires a non-empty loader.")
    transform = experiment.data.batch_transform

    rows: list[dict[str, Any]] = []
    cell_grads: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)

    for seed in spec.seeds:
        seed_everything(int(seed))
        batch_iter = iter(loader)
        for batch_idx in range(int(spec.n_batches)):
            batch = next(batch_iter)
            if transform is not None:
                batch = transform(batch, device)
            probe_batch = model.encode_for_probe(batch)
            bs = probe_batch.zs.shape[0] * probe_batch.zs.shape[1]
            d = probe_batch.zs.shape[2]
            sk = int(model.transition.S_k)

            for replica in range(int(spec.R)):
                shared_eps = torch.randn(bs, d, sk, device=device)
                shared_k_idx: dict[str, torch.Tensor] = {}
                for mode, trans in transitions.items():
                    shared_k_idx[mode] = torch.multinomial(
                        trans.p_k, bs * sk, replacement=True
                    ).view(bs, sk).to(device=device, dtype=torch.long)

                for cell in spec.cells:
                    trans = transitions[cell.k_sampling_mode]
                    trans.zero_grad(set_to_none=True)
                    out = trans.transition_kl(
                        **probe_batch.as_kwargs(),
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
                            "L_p_scalar": float(lp.item()),
                            "grad_norm": g_norm,
                        })

            if spec.force_per_k:
                k_max = int(getattr(model.transition, "num_steps", 0))
                for k in range(k_max):
                    forced_idx = torch.full((bs, sk), int(k), device=device, dtype=torch.long)
                    forced_eps = torch.randn(bs, d, sk, device=device)
                    for cell in spec.cells:
                        trans = transitions[cell.k_sampling_mode]
                        out = trans.transition_kl(
                            **probe_batch.as_kwargs(),
                            mc_override={
                                "eps": forced_eps,
                                "k_idx": forced_idx,
                                "objective": cell.objective,
                            },
                            return_per_sample=True,
                        )
                        per_sample = out["L_p_per_sample"].detach().cpu().numpy()
                        rows.append({
                            "seed": int(seed),
                            "batch_idx": int(batch_idx),
                            "replica": -1,
                            "objective": cell.objective,
                            "k_sampling_mode": cell.k_sampling_mode,
                            "kind": "forced_k",
                            "k_idx": int(k),
                            "sample_idx": -1,
                            "L_p": float(np.mean(per_sample)),
                            "L_p_scalar": float(out["L_p"].item()),
                            "grad_norm": math.nan,
                        })

    summary_cells: dict[str, dict[str, float]] = {}
    for (objective, mode), grad_list in cell_grads.items():
        key = f"{objective}:{mode}"
        grad_arr = np.stack(grad_list, axis=0)
        summary_cells[key] = {
            "grad_norm_mean": float(np.linalg.norm(grad_arr, axis=1).mean()),
            "grad_variance": float(np.var(grad_arr, axis=0).mean()),
        }

    summary = {"cells": summary_cells}
    return rows, summary, transitions
