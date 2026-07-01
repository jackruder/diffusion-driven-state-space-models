"""Optimizer construction utilities: per-component AdamW param groups and LR schedules."""

import math

import torch
import torch.nn as nn


def param_groups_for_adamw(
    model,
    enc_lr: float,
    dec_lr: float,
    trans_lr: float,
    weight_decay: float = 0.01,
    baseline_lr: float | None = None,
):
    """Build per-component AdamW parameter groups with selective weight decay.

    Norm layers, bias parameters, embeddings, and log-variance raw parameters
    are placed in a zero-weight-decay group; all other parameters receive the
    full ``weight_decay``. Frozen (``requires_grad=False``) parameters are
    included too — AdamW leaves them untouched while their grads are None —
    so group membership does not depend on the current stage trainable mask.

    Args:
        model: The ``DDSSM_base`` model.
        enc_lr: Learning rate for encoder parameters.
        dec_lr: Learning rate for decoder parameters.
        trans_lr: Learning rate for transition model parameters.
        weight_decay: L2 regularisation coefficient for decay groups.
        baseline_lr: Learning rate for the baseline μ_p head; defaults to
            ``trans_lr`` when ``None``.

    Returns:
        List of parameter-group dicts ready to pass to ``torch.optim.AdamW``.
    """
    groups = []
    # Some submodules (notably ``baseline``) are reachable from multiple
    # top-level attributes (e.g. ``model.baseline`` and
    # ``model.transition.baseline``) because the model-v2 architecture
    # shares the μ_p instance across both transitions and exposes it at
    # the top level for the declarative trainable mask.  Without
    # deduplication AdamW raises "some parameters appear in more than
    # one parameter group".  The first add_module call that walks a
    # given Parameter wins (the LR/wd group it ends up in is
    # deterministic via the call order below).
    _claimed_ids: set[int] = set()

    def add_module(module, lr: float, tag: str):
        if module is None or lr is None:
            return

        # Collect params that should never get weight decay by module type
        no_wd_ids = set()
        for m in module.modules():
            if isinstance(
                m,
                (
                    nn.LayerNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                    nn.Embedding,
                ),
            ):
                for p in m.parameters(recurse=False):
                    no_wd_ids.add(id(p))

        decay_params, nodecay_params = [], []
        # Params are included regardless of ``requires_grad``: the per-stage
        # trainable mask is applied AFTER optimizer (re)builds, so filtering
        # here silently dropped modules that a stage was about to unfreeze
        # (they would never train), and made optimizer state_dicts
        # load-incompatible across stages on resume. AdamW skips params whose
        # ``.grad`` is None — no update and no weight decay — so
        # ``_set_trainable`` remains the single gradient-suppression
        # mechanism and group membership stays stable across stage masks.
        for n, p in module.named_parameters(recurse=True):
            if id(p) in _claimed_ids:
                continue
            _claimed_ids.add(id(p))

            n_lower = n.lower()
            is_bias = n_lower.endswith("bias")
            is_norm = ("norm" in n_lower) or n_lower.endswith(
                "weight_g"
            )  # weightnorm g
            is_embed = "embedding" in n_lower
            is_decoder_logvar = "logvar_raw" in n_lower

            if (
                id(p) in no_wd_ids
                or is_bias
                or is_norm
                or is_embed
                or is_decoder_logvar
            ):
                nodecay_params.append(p)
            else:
                decay_params.append(p)

        if decay_params:
            groups.append({
                "params": decay_params,
                "lr": lr,
                "weight_decay": weight_decay,
            })
        if nodecay_params:
            groups.append({"params": nodecay_params, "lr": lr, "weight_decay": 0.0})

    add_module(getattr(model, "encoder", None), enc_lr, "encoder")
    add_module(getattr(model, "decoder", None), dec_lr, "decoder")
    add_module(getattr(model, "transition", None), trans_lr, "transition")

    # Capture top-level static embeddings (using encoder's LR)
    add_module(getattr(model, "static_embeddings", None), enc_lr, "static_embeddings")

    # model-v2 slots: aux_posterior shares the encoder LR (it's part of the
    # variational-inference family); baseline gets its own LR, defaulting to
    # trans_lr if not provided.
    add_module(getattr(model, "aux_posterior", None), enc_lr, "aux_posterior")
    bl_lr = trans_lr if baseline_lr is None else baseline_lr
    add_module(getattr(model, "baseline", None), bl_lr, "baseline")

    return groups


def make_warmup_cosine(optimizer, total_steps, warmup_steps=500, final_scale=0.05):
    """Build a linear-warmup + cosine-decay LR scheduler.

    Args:
        optimizer: The ``torch.optim.Optimizer`` to schedule.
        total_steps: Total number of training steps.
        warmup_steps: Number of linear-ramp-up steps from 0 to base LR.
        final_scale: Fraction of base LR at the end of the cosine decay.

    Returns:
        A ``torch.optim.lr_scheduler.LambdaLR`` scheduler.
    """
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def scale_fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / (total_steps - warmup_steps)
        # cosine from 1.0 -> final_scale
        return final_scale + (1.0 - final_scale) * 0.5 * (1.0 + math.cos(math.pi * t))

    # same scaler to all param groups; their base LRs differ already
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[scale_fn] * len(optimizer.param_groups)
    )
