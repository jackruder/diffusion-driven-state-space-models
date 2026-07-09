"""Optimizer construction utilities: per-component AdamW param groups and LR schedules."""

import math
from collections.abc import Iterator

import torch
import torch.nn as nn

# ψ side of the split-loss parameter partition: the score-net family.
# ``embed_layer`` is an attribute of the *transition itself* (not inside
# ``diffmodel``) — it is the score net's input featurization and receives
# gradient only through ``diffmodel``'s forward.
_PSI_TRANSITION_SUBMODULES: tuple[str, ...] = ("diffmodel", "embed_layer")

# Transition top-level submodules that are explicitly known to belong to
# the φθ side. Enumerated from today's transition classes:
#   - DiffusionTransition: baseline (shared μ_p head)
#   - GaussianTransition:  z_hist_proj, context_producer, gaussian_head
# Any transition child that is neither here nor in
# ``_PSI_TRANSITION_SUBMODULES`` hard-errors in the split helpers so a
# future module cannot land silently mis-routed.
_KNOWN_PHITH_TRANSITION_SUBMODULES: frozenset[str] = frozenset({
    "baseline",
    "z_hist_proj",
    "gaussian_head",
    "context_producer",
})


def _validate_transition_routing(model: nn.Module) -> None:
    """Hard-error on transition submodules with no explicit φθ/ψ routing.

    Checks the top-level children of ``model.transition`` against the
    explicit ψ and known-φθ allowlists.

    Args:
        model: The ``DDSSM_base`` model.

    Raises:
        ValueError: Listing the offending submodule names, if any
            transition child is neither explicitly ψ nor known-φθ.
    """
    trans = getattr(model, "transition", None)
    if trans is None:
        return
    unknown = [
        name
        for name, _ in trans.named_children()
        if name not in _KNOWN_PHITH_TRANSITION_SUBMODULES
        and name not in _PSI_TRANSITION_SUBMODULES
    ]
    if unknown:
        raise ValueError(
            f"model.transition has top-level submodule(s) {unknown!r} with no "
            f"explicit phith/psi routing. Add them to "
            f"_PSI_TRANSITION_SUBMODULES or "
            f"_KNOWN_PHITH_TRANSITION_SUBMODULES in "
            f"ddssm.training.train_utils so the split-loss parameter "
            f"partition stays exhaustive."
        )


def _iter_psi_modules(model: nn.Module) -> Iterator[nn.Module]:
    """Yield the score-net-family modules present on ``model.transition``.

    Non-diffusion transitions (no ``diffmodel``/``embed_layer``
    attributes) simply yield nothing.

    Args:
        model: The ``DDSSM_base`` model.

    Yields:
        ``nn.Module``: each ψ-side submodule that exists on the transition.
    """
    transition = getattr(model, "transition", None)
    if transition is None:
        return
    for name in _PSI_TRANSITION_SUBMODULES:
        module = getattr(transition, name, None)
        if module is not None:
            yield module


def split_params_phith_psi(
    model: nn.Module,
    include_frozen: bool = False,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition trainable parameters into the (φθ, ψ) split-loss sides.

    ψ is the score-net family — ``model.transition.diffmodel`` plus
    ``model.transition.embed_layer`` — and φθ is every other
    ``requires_grad=True`` parameter reachable from ``model``
    (encoder, decoder, baseline, aux_posterior, static_embeddings, and
    any non-ψ transition submodule). Modules reachable from multiple
    top-level attributes (notably the shared ``baseline``) are
    deduplicated by parameter identity, so each ``nn.Parameter`` appears
    exactly once across the two lists.

    For non-diffusion transitions (no ``diffmodel``/``embed_layer``)
    the ψ list is empty and φθ covers everything.

    Args:
        model: The ``DDSSM_base`` model.
        include_frozen: When ``True``, partition EVERY parameter onto its
            side regardless of ``requires_grad`` (the exhaustive/disjoint
            fence then runs over all params too). This is what the
            trainer caches at split-topology install so per-stage
            freeze/unfreeze after install can be re-filtered live at each
            backward. Default ``False`` keeps the historical
            requires-grad-only behavior.

    Returns:
        Tuple ``(phith_params, psi_params)`` of parameter lists covering
        every ``requires_grad=True`` parameter exactly once (every
        parameter when ``include_frozen=True``).

    Raises:
        ValueError: If a transition top-level submodule has no explicit
            φθ/ψ routing (see :func:`_validate_transition_routing`).
    """
    _validate_transition_routing(model)

    psi_ids: set[int] = set()
    psi_params: list[torch.nn.Parameter] = []
    for module in _iter_psi_modules(model):
        for p in module.parameters():
            if id(p) in psi_ids:
                continue
            psi_ids.add(id(p))
            if include_frozen or p.requires_grad:
                psi_params.append(p)

    phith_params: list[torch.nn.Parameter] = []
    seen: set[int] = set(psi_ids)
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if include_frozen or p.requires_grad:
            phith_params.append(p)

    # Exhaustive-and-disjoint fence over model.parameters().
    split_ids = {id(p) for p in phith_params} | {id(p) for p in psi_params}
    assert len(phith_params) + len(psi_params) == len(split_ids), (
        "phith/psi parameter lists overlap"
    )
    missing = [
        n
        for n, p in model.named_parameters()
        if (include_frozen or p.requires_grad) and id(p) not in split_ids
    ]
    assert not missing, f"params not covered by the phith/psi split: {missing}"

    return phith_params, psi_params


def _append_module_param_groups(
    groups: list[dict],
    claimed_ids: set[int],
    module: nn.Module | None,
    lr: float | None,
    weight_decay: float,
    betas: tuple[float, float] | None = None,
) -> None:
    """Append the decay/no-decay AdamW groups for one module.

    Norm layers, bias parameters, embeddings, and log-variance raw
    parameters go into a zero-weight-decay group; everything else
    receives the full ``weight_decay``. Params are included regardless
    of ``requires_grad``: the per-stage trainable mask is applied AFTER
    optimizer (re)builds, so filtering here silently dropped modules
    that a stage was about to unfreeze (they would never train), and
    made optimizer state_dicts load-incompatible across stages on
    resume. AdamW skips params whose ``.grad`` is None — no update and
    no weight decay — so ``_set_trainable`` remains the single
    gradient-suppression mechanism and group membership stays stable
    across stage masks.

    Args:
        groups: Output list of parameter-group dicts (mutated in place).
        claimed_ids: Set of ``id(param)`` already assigned to a group
            (mutated in place); the first caller to walk a given
            parameter wins.
        module: The module to walk; ``None`` is a no-op.
        lr: Learning rate for the appended groups; ``None`` is a no-op.
        weight_decay: L2 regularisation coefficient for the decay group.
        betas: Optional per-group Adam betas; when not ``None`` the
            appended group dicts carry a ``"betas"`` key, otherwise no
            such key is emitted.
    """
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
    for n, p in module.named_parameters(recurse=True):
        if id(p) in claimed_ids:
            continue
        claimed_ids.add(id(p))

        n_lower = n.lower()
        is_bias = n_lower.endswith("bias")
        is_norm = ("norm" in n_lower) or n_lower.endswith("weight_g")  # weightnorm g
        is_embed = "embedding" in n_lower
        is_decoder_logvar = "logvar_raw" in n_lower

        if id(p) in no_wd_ids or is_bias or is_norm or is_embed or is_decoder_logvar:
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    if decay_params:
        group = {
            "params": decay_params,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        if betas is not None:
            group["betas"] = betas
        groups.append(group)
    if nodecay_params:
        group = {"params": nodecay_params, "lr": lr, "weight_decay": 0.0}
        if betas is not None:
            group["betas"] = betas
        groups.append(group)


def param_groups_for_adamw(
    model,
    enc_lr: float,
    dec_lr: float,
    trans_lr: float,
    weight_decay: float = 1e-4,
    baseline_lr: float | None = None,
    psi_betas: tuple[float, float] | None = None,
    weight_decay_psi: float | None = None,
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
        psi_betas: Optional Adam betas for the score-net family
            (``transition.diffmodel`` + ``transition.embed_layer``, the
            same assignment as :func:`split_params_phith_psi`). When set,
            exactly those groups carry a ``"betas"`` key (AdamW resolves
            hyperparams per group, so this gives the two-timescale β₂
            inside one optimizer); all other groups carry no ``betas``
            key and inherit the optimizer's constructor default. When
            ``None`` (default) the output is identical to the
            pre-``psi_betas`` behavior: no group carries a ``betas`` key.
        weight_decay_psi: Optional weight decay for the score-net family's
            decay groups (same ψ assignment as ``psi_betas``). ``None``
            (default) applies ``weight_decay`` uniformly.

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

    def add_module(
        module,
        lr: float,
        betas: tuple[float, float] | None = None,
        wd: float | None = None,
    ):
        _append_module_param_groups(
            groups,
            _claimed_ids,
            module,
            lr,
            weight_decay if wd is None else wd,
            betas=betas,
        )

    add_module(getattr(model, "encoder", None), enc_lr)
    add_module(getattr(model, "decoder", None), dec_lr)

    transition = getattr(model, "transition", None)
    if psi_betas is not None or weight_decay_psi is not None:
        # Claim the score-net family first so exactly its groups carry
        # the per-group betas / ψ weight decay; the remaining transition
        # params follow in untagged groups.
        for name in _PSI_TRANSITION_SUBMODULES:
            add_module(
                getattr(transition, name, None) if transition is not None else None,
                trans_lr,
                betas=tuple(psi_betas) if psi_betas is not None else None,
                wd=weight_decay_psi,
            )
    add_module(transition, trans_lr)

    # Capture top-level static embeddings (using encoder's LR)
    add_module(getattr(model, "static_embeddings", None), enc_lr)

    # model-v2 slots: aux_posterior shares the encoder LR (it's part of the
    # variational-inference family); baseline gets its own LR, defaulting to
    # trans_lr if not provided.
    add_module(getattr(model, "aux_posterior", None), enc_lr)
    bl_lr = trans_lr if baseline_lr is None else baseline_lr
    add_module(getattr(model, "baseline", None), bl_lr)

    return groups


def param_groups_phith(
    model: nn.Module,
    enc_lr: float,
    dec_lr: float,
    trans_lr: float,
    weight_decay: float = 1e-4,
    baseline_lr: float | None = None,
) -> list[dict]:
    """Build the φθ-side AdamW parameter groups for split-loss training.

    Mirrors :func:`param_groups_for_adamw` (same walk order, LR
    assignment, and decay/no-decay split) restricted to the φθ side of
    :func:`split_params_phith_psi`: the score-net family
    (``transition.diffmodel`` + ``transition.embed_layer``) is
    pre-claimed and therefore excluded from every group.

    Args:
        model: The ``DDSSM_base`` model.
        enc_lr: Learning rate for encoder parameters.
        dec_lr: Learning rate for decoder parameters.
        trans_lr: Learning rate for the non-ψ transition parameters.
        weight_decay: L2 regularisation coefficient for decay groups.
        baseline_lr: Learning rate for the baseline μ_p head; defaults to
            ``trans_lr`` when ``None``.

    Returns:
        List of parameter-group dicts ready to pass to ``torch.optim.AdamW``.

    Raises:
        ValueError: If a transition top-level submodule has no explicit
            φθ/ψ routing.
    """
    _validate_transition_routing(model)

    groups: list[dict] = []
    claimed_ids: set[int] = set()
    # Pre-claim the ψ side (all its params, frozen included — group
    # membership is mask-independent) so it lands in no φθ group.
    for module in _iter_psi_modules(model):
        claimed_ids.update(id(p) for p in module.parameters())

    def add_module(module: nn.Module | None, lr: float) -> None:
        _append_module_param_groups(groups, claimed_ids, module, lr, weight_decay)

    add_module(getattr(model, "encoder", None), enc_lr)
    add_module(getattr(model, "decoder", None), dec_lr)
    add_module(getattr(model, "transition", None), trans_lr)
    add_module(getattr(model, "static_embeddings", None), enc_lr)
    add_module(getattr(model, "aux_posterior", None), enc_lr)
    bl_lr = trans_lr if baseline_lr is None else baseline_lr
    add_module(getattr(model, "baseline", None), bl_lr)

    return groups


def param_groups_psi(
    model: nn.Module,
    trans_lr: float,
    weight_decay: float = 1e-4,
) -> list[dict]:
    """Build the ψ-side AdamW parameter groups for split-loss training.

    Covers exactly the score-net family — ``transition.diffmodel`` plus
    ``transition.embed_layer`` — with the same decay/no-decay rules as
    :func:`param_groups_for_adamw` (``embed_layer`` is an
    ``nn.Embedding`` and lands in the zero-weight-decay group). For a
    non-diffusion transition the result is an empty list.

    Args:
        model: The ``DDSSM_base`` model.
        trans_lr: Learning rate for the score-net parameters.
        weight_decay: L2 regularisation coefficient for decay groups.

    Returns:
        List of parameter-group dicts ready to pass to ``torch.optim.AdamW``.

    Raises:
        ValueError: If a transition top-level submodule has no explicit
            φθ/ψ routing.
    """
    _validate_transition_routing(model)

    groups: list[dict] = []
    claimed_ids: set[int] = set()
    for module in _iter_psi_modules(model):
        _append_module_param_groups(groups, claimed_ids, module, trans_lr, weight_decay)
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
