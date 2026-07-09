"""Unit tests for the φθ/ψ split-loss parameter partition (train_utils).

Pins the contracts of ``split_params_phith_psi`` and the
``param_groups_phith`` / ``param_groups_psi`` /
``param_groups_for_adamw(psi_betas=...)`` builders: the partition is
exhaustive and disjoint, the score-net family (``transition.diffmodel``
+ ``transition.embed_layer``) is the ψ side, the shared baseline lands
on the φθ side exactly once, unknown transition submodules hard-error,
and the single-mode ``psi_betas`` tagging touches exactly the score-net
groups.
"""

from __future__ import annotations

import os
import copy

import torch
import pytest
import torch.nn as nn

from tests.test_trainer import make_small_model
from ddssm.training.train_utils import (
    param_groups_psi,
    param_groups_phith,
    param_groups_for_adamw,
    split_params_phith_psi,
)
from tests.test_integration.conftest import make_vhp_model


@pytest.fixture(scope="module", autouse=True)
def _eager_models():
    """Build models eagerly (no ``torch.compile``) — fast and deterministic."""
    old = os.environ.get("DDSSM_TORCH_COMPILE")
    os.environ["DDSSM_TORCH_COMPILE"] = "0"
    yield
    if old is None:
        os.environ.pop("DDSSM_TORCH_COMPILE", None)
    else:
        os.environ["DDSSM_TORCH_COMPILE"] = old


@pytest.fixture(scope="module")
def vhp_model(_eager_models):
    """One shared DiffusionTransition model (construction amortized)."""
    torch.manual_seed(0)
    return make_vhp_model()


def _ids(params) -> set[int]:
    return {id(p) for p in params}


def _group_ids(groups) -> set[int]:
    """Union of param ids across groups, asserting no duplicates."""
    out: set[int] = set()
    n = 0
    for g in groups:
        out |= {id(p) for p in g["params"]}
        n += len(g["params"])
    assert n == len(out), "duplicate param across groups"
    return out


def test_split_exhaustive_and_disjoint(vhp_model):
    """The (φθ, ψ) partition covers every requires-grad param exactly once."""
    for model in (vhp_model, make_small_model()):
        phith, psi = split_params_phith_psi(model)
        assert not (_ids(phith) & _ids(psi)), "phith/psi overlap"
        all_rg = {id(p) for p in model.parameters() if p.requires_grad}
        assert _ids(phith) | _ids(psi) == all_rg, "partition not exhaustive"


def test_split_include_frozen_covers_all_params(vhp_model):
    """``include_frozen=True`` partitions EVERY param regardless of the mask."""
    model = vhp_model
    dec = list(model.decoder.parameters())
    emb = list(model.transition.embed_layer.parameters())
    try:
        for p in dec + emb:
            p.requires_grad = False
        phith, psi = split_params_phith_psi(model, include_frozen=True)
        assert not (_ids(phith) & _ids(psi))
        assert _ids(phith) | _ids(psi) == {id(p) for p in model.parameters()}
        assert _ids(emb) <= _ids(psi), "frozen embed_layer must stay on psi"
        assert _ids(dec) <= _ids(phith), "frozen decoder must stay on phith"
        # The default requires-grad-only call excludes the frozen params.
        phith_rg, psi_rg = split_params_phith_psi(model)
        assert not (_ids(dec) & _ids(phith_rg))
        assert not (_ids(emb) & _ids(psi_rg))
    finally:
        for p in dec + emb:
            p.requires_grad = True


def test_score_net_family_lands_psi_and_baseline_phith_once(vhp_model):
    """``embed_layer``/``diffmodel`` → ψ; the shared baseline → φθ, once."""
    model = vhp_model
    phith, psi = split_params_phith_psi(model)
    emb_ids = _ids(model.transition.embed_layer.parameters())
    dm_ids = _ids(model.transition.diffmodel.parameters())
    assert emb_ids <= _ids(psi) and not (emb_ids & _ids(phith))
    assert dm_ids <= _ids(psi) and not (dm_ids & _ids(phith))
    # The baseline is reachable both as model.baseline and via the
    # transition; the dedup must keep each param exactly once, on φθ.
    assert model.baseline is model.transition.baseline, "alias precondition"
    bl_ids = _ids(model.baseline.parameters())
    assert bl_ids <= _ids(phith) and not (bl_ids & _ids(psi))
    count = sum(1 for p in phith if id(p) in bl_ids)
    assert count == len(bl_ids), f"baseline params appear {count}x, want once each"


def test_gaussian_transition_has_empty_psi_side():
    """A non-diffusion transition yields ψ = [] and φθ covering everything."""
    model = make_small_model()
    phith, psi = split_params_phith_psi(model)
    assert psi == []
    assert _ids(phith) == {id(p) for p in model.parameters() if p.requires_grad}


def test_unknown_transition_submodule_raises(vhp_model):
    """A transition child with no explicit routing hard-errors, naming itself."""
    model = copy.deepcopy(vhp_model)
    model.transition.mystery_head = nn.Linear(2, 2)
    try:
        with pytest.raises(ValueError, match="mystery_head"):
            split_params_phith_psi(model)
    finally:
        del model.transition.mystery_head
    split_params_phith_psi(model)  # clean again after removal


def test_param_groups_cover_each_side_exactly(vhp_model):
    """``param_groups_phith``/``param_groups_psi`` mirror the split sides."""
    for model in (vhp_model, make_small_model()):
        phith, psi = split_params_phith_psi(model)
        gp = param_groups_phith(
            model,
            enc_lr=1e-3,
            dec_lr=1e-4,
            trans_lr=5e-4,
            weight_decay=0.01,
            baseline_lr=2e-4,
        )
        gq = param_groups_psi(model, trans_lr=5e-4, weight_decay=0.01)
        assert _group_ids(gp) == _ids(phith)
        assert _group_ids(gq) == _ids(psi)
        assert not (_group_ids(gp) & _group_ids(gq))


def test_embed_layer_in_zero_weight_decay_psi_group(vhp_model):
    """``embed_layer`` (an ``nn.Embedding``) lands in a wd=0.0 ψ group."""
    model = vhp_model
    gq = param_groups_psi(model, trans_lr=5e-4, weight_decay=0.01)
    emb_ids = _ids(model.transition.embed_layer.parameters())
    assert emb_ids <= _group_ids(gq)
    for g in gq:
        if {id(p) for p in g["params"]} & emb_ids:
            assert g["weight_decay"] == 0.0, "embed_layer must be no-decay"


def test_psi_betas_tags_exactly_score_net_groups(vhp_model):
    """``psi_betas`` tags precisely the score-net groups with ``"betas"``."""
    model = vhp_model
    groups = param_groups_for_adamw(
        model,
        enc_lr=1e-3,
        dec_lr=1e-4,
        trans_lr=5e-4,
        weight_decay=0.01,
        psi_betas=(0.9, 0.99),
    )
    tagged = [g for g in groups if "betas" in g]
    untagged = [g for g in groups if "betas" not in g]
    score_ids = _ids(model.transition.diffmodel.parameters()) | _ids(
        model.transition.embed_layer.parameters()
    )
    assert tagged, "psi_betas must produce tagged groups on a diffusion model"
    assert _group_ids(tagged) == score_ids
    assert all(g["betas"] == (0.9, 0.99) for g in tagged)
    assert not (_group_ids(untagged) & score_ids)
    assert _group_ids(groups) == {id(p) for p in model.parameters()}

    # A Gaussian model has no score net — nothing to tag.
    gaussian_groups = param_groups_for_adamw(
        make_small_model(),
        enc_lr=1e-3,
        dec_lr=1e-4,
        trans_lr=5e-4,
        weight_decay=0.01,
        psi_betas=(0.9, 0.99),
    )
    assert all("betas" not in g for g in gaussian_groups)


def test_weight_decay_psi_overrides_exactly_score_net_groups(vhp_model):
    """``weight_decay_psi`` retargets precisely the score-net decay groups."""
    model = vhp_model
    groups = param_groups_for_adamw(
        model,
        enc_lr=1e-3,
        dec_lr=1e-4,
        trans_lr=5e-4,
        weight_decay=1e-4,
        weight_decay_psi=0.0,
    )
    score_ids = _ids(model.transition.diffmodel.parameters()) | _ids(
        model.transition.embed_layer.parameters()
    )
    assert _group_ids(groups) == {id(p) for p in model.parameters()}
    for g in groups:
        gids = {id(p) for p in g["params"]}
        if gids & score_ids:
            assert gids <= score_ids, "ψ groups must not mix in φθ params"
            assert g["weight_decay"] == 0.0
        else:
            assert g["weight_decay"] in (1e-4, 0.0)
    # Without psi_betas, the override alone must not tag betas.
    assert all("betas" not in g for g in groups)
    # φθ decay groups keep the base weight decay.
    phith_decay = [
        g
        for g in groups
        if not ({id(p) for p in g["params"]} & score_ids) and g["weight_decay"] > 0.0
    ]
    assert phith_decay, "φθ side must retain a nonzero-decay group"


def test_weight_decay_psi_none_leaves_groups_untouched(vhp_model):
    """``weight_decay_psi=None`` reproduces the plain group structure."""
    kwargs = dict(enc_lr=1e-3, dec_lr=1e-4, trans_lr=5e-4, weight_decay=1e-4)
    plain = param_groups_for_adamw(vhp_model, **kwargs)
    none_case = param_groups_for_adamw(vhp_model, weight_decay_psi=None, **kwargs)
    assert len(plain) == len(none_case)
    for a, b in zip(plain, none_case):
        assert a.keys() == b.keys()
        assert [id(p) for p in a["params"]] == [id(p) for p in b["params"]]
        assert a["lr"] == b["lr"] and a["weight_decay"] == b["weight_decay"]


def test_psi_betas_none_leaves_groups_untouched(vhp_model):
    """``psi_betas=None`` emits no ``betas`` key and the same group structure."""
    model = vhp_model
    kwargs = dict(enc_lr=1e-3, dec_lr=1e-4, trans_lr=5e-4, weight_decay=0.01)
    plain = param_groups_for_adamw(model, **kwargs)
    none_case = param_groups_for_adamw(model, psi_betas=None, **kwargs)
    assert all("betas" not in g for g in none_case)
    assert len(plain) == len(none_case)
    for a, b in zip(plain, none_case):
        assert a.keys() == b.keys()
        assert [id(p) for p in a["params"]] == [id(p) for p in b["params"]]
        assert a["lr"] == b["lr"] and a["weight_decay"] == b["weight_decay"]


# ---------------------------------------------------------------------------
# Ported from the parallel local implementation (see git stash@{0}).
# ---------------------------------------------------------------------------


def _has_decay_and_nodecay(groups: list[dict], expected_wd: float) -> bool:
    """True iff ``groups`` has ≥1 decay (wd == expected_wd) and ≥1 no-decay (wd == 0)."""
    has_decay = any(g["weight_decay"] == expected_wd for g in groups)
    has_nodecay = any(g["weight_decay"] == 0.0 for g in groups)
    return has_decay and has_nodecay


def test_local_param_groups_decay_split_preserved_per_side(vhp_model):
    """Both φθ and ψ builders emit a decay AND a no-decay group.

    Also pins the embed_layer nn.Embedding to the ψ no-decay bucket — a
    cross-check on the AdamW decay policy that ``test_embed_layer_in_zero_
    weight_decay_psi_group`` above only asserts for the ψ side.
    """
    WD = 0.05

    phith_groups = param_groups_phith(
        vhp_model,
        enc_lr=1e-3,
        dec_lr=1e-4,
        trans_lr=5e-4,
        weight_decay=WD,
    )
    psi_groups = param_groups_psi(vhp_model, trans_lr=5e-4, weight_decay=WD)

    assert phith_groups, "param_groups_phith returned empty list"
    assert psi_groups, "param_groups_psi returned empty list"

    assert _has_decay_and_nodecay(phith_groups, WD), (
        "φθ groups missing decay or no-decay bucket"
    )
    assert _has_decay_and_nodecay(psi_groups, WD), (
        "ψ groups missing decay or no-decay bucket"
    )

    embed_ids = _ids(vhp_model.transition.embed_layer.parameters())
    no_decay_psi_ids = {
        id(p) for g in psi_groups if g["weight_decay"] == 0.0 for p in g["params"]
    }
    assert embed_ids <= no_decay_psi_ids, (
        "embed_layer params should be in ψ's no-decay group (it's nn.Embedding)"
    )


# ---------------------------------------------------------------------------
# Role-tagging tests (new: param-group role="phith" / role="psi" tagging).
# ---------------------------------------------------------------------------


def test_all_groups_have_role_key_no_psi_args(vhp_model):
    """Every group from ``param_groups_for_adamw`` (no ψ args) carries ``role``.

    With the ψ pre-claim gate closed (no ``psi_betas``, no
    ``weight_decay_psi``, no ``claim_psi``), all groups are on the φθ
    side — every group must carry ``role="phith"`` and none may carry
    ``role="psi"``.
    """
    groups = param_groups_for_adamw(
        vhp_model,
        enc_lr=1e-3,
        dec_lr=1e-4,
        trans_lr=5e-4,
        weight_decay=1e-4,
    )
    assert all("role" in g for g in groups), "every group must carry a 'role' key"
    assert all(g["role"] == "phith" for g in groups), (
        "with gate closed, all groups must be role='phith'"
    )


def test_claim_psi_no_psi_args_tags_score_net_psi(vhp_model):
    """``claim_psi=True`` with no other ψ args routes score-net to role="psi".

    Checks:
    - diffmodel + embed_layer params land in role="psi" groups.
    - ψ groups carry weight_decay equal to the global ``weight_decay``
      (fallback; no ``weight_decay_psi`` supplied).
    - ψ groups carry NO ``"betas"`` key (no ``psi_betas`` supplied).
    - All params covered exactly once (exhaustive, disjoint).
    - Non-score-net groups carry role="phith".
    """
    WD = 2e-4
    groups = param_groups_for_adamw(
        vhp_model,
        enc_lr=1e-3,
        dec_lr=1e-4,
        trans_lr=5e-4,
        weight_decay=WD,
        claim_psi=True,
    )
    score_ids = _ids(vhp_model.transition.diffmodel.parameters()) | _ids(
        vhp_model.transition.embed_layer.parameters()
    )
    psi_groups = [g for g in groups if g.get("role") == "psi"]
    phith_groups = [g for g in groups if g.get("role") == "phith"]

    assert all("role" in g for g in groups), "every group must carry a 'role' key"

    # Score-net ids must land exclusively in psi groups
    assert _group_ids(psi_groups) == score_ids, (
        "psi groups must cover exactly the score-net params"
    )

    # Fallback weight_decay applied to ψ decay groups (no override given)
    for g in psi_groups:
        assert g["weight_decay"] in (WD, 0.0), (
            "ψ group weight_decay must be WD or 0 (no betas)"
        )

    # No betas key on ψ groups when psi_betas not given
    assert all("betas" not in g for g in psi_groups), (
        "no 'betas' key expected without psi_betas"
    )

    # φθ groups must not contain any score-net params
    assert not (_group_ids(phith_groups) & score_ids), (
        "φθ groups must not contain score-net params"
    )

    # Exhaustive and disjoint: all model params covered exactly once
    all_model_ids = {id(p) for p in vhp_model.parameters()}
    assert _group_ids(groups) == all_model_ids, "groups must cover all model params"


def test_claim_psi_with_weight_decay_psi_zero_honored(vhp_model):
    """``weight_decay_psi=0.0`` overrides the fallback on ψ decay groups."""
    WD = 1e-4
    groups = param_groups_for_adamw(
        vhp_model,
        enc_lr=1e-3,
        dec_lr=1e-4,
        trans_lr=5e-4,
        weight_decay=WD,
        claim_psi=True,
        weight_decay_psi=0.0,
    )
    score_ids = _ids(vhp_model.transition.diffmodel.parameters()) | _ids(
        vhp_model.transition.embed_layer.parameters()
    )
    for g in groups:
        gids = {id(p) for p in g["params"]}
        if gids & score_ids:
            assert gids <= score_ids, "ψ groups must not mix in φθ params"
            assert g["weight_decay"] == 0.0, (
                "weight_decay_psi=0.0 must force ψ groups to wd=0.0"
            )
            assert g.get("role") == "psi"
        else:
            assert g.get("role") == "phith"


def test_param_groups_phith_all_role_phith(vhp_model):
    """All groups from ``param_groups_phith`` carry ``role="phith"``."""
    for model in (vhp_model, make_small_model()):
        groups = param_groups_phith(
            model,
            enc_lr=1e-3,
            dec_lr=1e-4,
            trans_lr=5e-4,
            weight_decay=0.01,
        )
        assert groups, "param_groups_phith must return non-empty list"
        assert all("role" in g for g in groups), "every group must carry a 'role' key"
        assert all(g["role"] == "phith" for g in groups), (
            "param_groups_phith must tag all groups role='phith'"
        )


def test_param_groups_psi_all_role_psi(vhp_model):
    """All groups from ``param_groups_psi`` carry ``role="psi"``."""
    groups = param_groups_psi(vhp_model, trans_lr=5e-4, weight_decay=0.01)
    assert groups, "param_groups_psi must return non-empty list for diffusion model"
    assert all("role" in g for g in groups), "every group must carry a 'role' key"
    assert all(g["role"] == "psi" for g in groups), (
        "param_groups_psi must tag all groups role='psi'"
    )


def test_none_equivalence_tests_still_pass_with_identical_roles(vhp_model):
    """The None-equivalence contracts still hold: same keys, same structure.

    With both ``weight_decay_psi=None`` and ``psi_betas=None`` the gate
    is closed (no pre-claim), both calls emit identical role keys, and
    structures are identical — mirroring the assertions in
    ``test_weight_decay_psi_none_leaves_groups_untouched`` and
    ``test_psi_betas_none_leaves_groups_untouched``.
    """
    kwargs = dict(enc_lr=1e-3, dec_lr=1e-4, trans_lr=5e-4, weight_decay=1e-4)
    plain = param_groups_for_adamw(vhp_model, **kwargs)
    none_wd = param_groups_for_adamw(vhp_model, weight_decay_psi=None, **kwargs)
    none_betas = param_groups_for_adamw(vhp_model, psi_betas=None, **kwargs)

    for variant in (none_wd, none_betas):
        assert len(plain) == len(variant)
        for a, b in zip(plain, variant):
            assert a.keys() == b.keys(), "key sets must be identical"
            assert a.get("role") == b.get("role"), "role values must match"
            assert [id(p) for p in a["params"]] == [id(p) for p in b["params"]]
            assert a["lr"] == b["lr"] and a["weight_decay"] == b["weight_decay"]
