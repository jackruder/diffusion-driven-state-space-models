"""Tests for the typed SweepSpace builder (import-time path validation)."""

from __future__ import annotations

import dataclasses

import pytest

from experiments._sweep import SweepSpace


@dataclasses.dataclass
class _Target:
    n_pretrain: int = 1
    base_lr: float = 1e-3


def test_valid_field_builds_full_path() -> None:
    s = SweepSpace(target=_Target, prefix="experiment.training.stages")
    s.log_int("n_pretrain", 5, 500).log("base_lr", 1e-5, 1e-3)
    params = s.params()
    assert (
        params["experiment.training.stages.n_pretrain"]
        == "tag(log, int(interval(5, 500)))"
    )
    assert "experiment.training.stages.base_lr" in params


def test_unknown_field_raises_at_build_time() -> None:
    s = SweepSpace(target=_Target, prefix="p")
    with pytest.raises(ValueError, match="not a field of"):
        s.log("nonexistent_field", 1.0, 2.0)


def test_duplicate_field_raises() -> None:
    s = SweepSpace(target=_Target, prefix="p")
    s.log("base_lr", 1e-5, 1e-3)
    with pytest.raises(ValueError, match="duplicate"):
        s.log("base_lr", 1e-4, 1e-2)


def test_moo_direction_objective_mismatch_raises() -> None:
    s = SweepSpace(target=_Target, prefix="p")
    s.log("base_lr", 1e-5, 1e-3)
    objectives = dataclasses.make_dataclass("Obj", [("specs", list)])(specs=[1, 2, 3])
    with pytest.raises(ValueError, match="direction has 2 entries"):
        s.build(
            sweeper="ddssm_optuna_moo",
            direction=["minimize", "minimize"],
            objectives=objectives,
        )


def test_real_init_ablation_space_validates_against_stagesb() -> None:
    """The shipped sweep fields all resolve against the live StagesB config."""
    from experiments.init_centering.hparams import StagesB

    s = SweepSpace(target=StagesB, prefix="experiment.training.stages")
    # All nine swept knobs are real StagesB fields (raises if any isn't).
    for f in (
        "n_pretrain",
        "sigma_pert",
        "anchor_lambda",
        "lambda_sigma_p",
        "base_lr",
        "dec_mult",
        "trans_mult",
        "stage_1_warmup_frac",
        "stage_2_warmup_frac",
    ):
        s.raw(f, "interval(0, 1)")
    assert len(s.params()) == 9
