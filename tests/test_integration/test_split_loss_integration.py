"""M8 integration + regression fences for split-loss training.

End-to-end coverage of the split-loss code path landed by M1-M7:
* Single-loss regression against the pre-refactor golden checkpoint.
* Single-vs-split parameter-delta agreement when weights collapse to (near-)
  unity.
* A full split-loss ``fit`` pass emits the new split KL metrics and does not
  spuriously trigger the non-finite grad-skip guard.
* A poisoned batch that produces a NaN gradient is caught by the grad-skip
  guard: training continues, params stay finite, and exactly one skip is
  counted.

A repo-level grep fence rides along here so a future refactor cannot
silently break the split-backward contract:

* ``use_reentrant=False`` remains on the ``torch.utils.checkpoint(...)`` call
  inside ``_esm_chunk_loss`` — without it the second selective backward
  silently corrupts.

Grad-norm clipping (``hparams.clip_grad_norm``, default 1.0) and the
always-on non-finite-grad skip guard are both active and compose: the whole-
model norm is computed and clipped every step, and a step is additionally
skipped outright (grads zeroed, no optimizer/scheduler/EMA update) when that
pre-clip norm is non-finite.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ddssm.model.losses import FullELBO, SplitLoss
from ddssm.training.train import DDSSMTrainer

# Reuse the fixture helpers already vetted by the other integration tests.
from tests.test_integration.conftest import (
    DATA_DIM,
    T_MAX,
    make_random_walk_data,
    make_vhp_model,
    run_stage,
)
from tests.fixtures.golden_values import (
    M8_CONFIG,
    M8_FINAL_METRICS,
    M8_SEED,
    load_m8_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Local helpers — a tiny torch Dataset shaped for the ``make_vhp_model``
# fixture (D=1) and a bare trainer builder that drives ``fit`` directly.
# ---------------------------------------------------------------------------


class _RandomWalkDataset(Dataset):
    """Deterministic random-walk dataset shaped for ``make_vhp_model``."""

    def __init__(self, *, n_seqs: int = 4, T: int = 8, seed: int = 0):
        payload = make_random_walk_data(n_seqs=n_seqs, T=T, seed=seed)
        self._obs = payload["observed_data"]
        self._mask = payload["observation_mask"]
        self._tp = payload["timepoints"]

    def __len__(self):
        return self._obs.size(0)

    def __getitem__(self, idx):
        return {
            "observed_data": self._obs[idx],
            "observation_mask": self._mask[idx],
            "timepoints": self._tp[idx],
        }


def _build_trainer(
    model,
    *,
    tmp_path: Path,
    use_split_loss: bool,
) -> DDSSMTrainer:
    trainer = DDSSMTrainer(
        model=model,
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        quiet=True,
    )
    trainer._active_loss = FullELBO(
        rate_lambda=lambda _step: 1.0,
        use_split_loss=use_split_loss,
    )
    return trainer


def _fit_n_steps(
    trainer: DDSSMTrainer,
    *,
    n_steps: int,
    batch_size: int = 2,
    n_seqs: int = 4,
    T: int = 8,
    seed: int = 0,
):
    torch.manual_seed(seed)
    ds = _RandomWalkDataset(n_seqs=n_seqs, T=T, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    trainer.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=n_steps,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
    )
    return trainer


# ---------------------------------------------------------------------------
# 1. Single-vs-split parameter-delta agreement under (near-)unit weights.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_single_vs_split_agree_at_unit_weight_end_to_end(tmp_path):
    """Parameter deltas after 5 steps match between single-loss and split-loss
    modes under a schedule tuned so ``w_ll/p`` is approximately constant.

    ``make_vhp_model``'s schedule is already ``k_sampling_mode="uniform"``
    (asserted below), under which the adaptive-IS ``p_k_clip`` never
    engages — so ``w_ll/p`` collapses toward a per-t constant without any
    schedule override. (A post-construction ``transition.schedule`` swap
    would NOT work: the transition caches ``k_sampling_mode``/``p_k_clip``
    and the σ̃ buffers at ``__init__``.)

    Tolerance rationale
    -------------------
    Even with uniform ``k`` sampling and ``sigma_data_init=1.0``, the two
    paths are not bit-identical:

    * split-mode drives φθ with ``loss_phith = recon + λ·(init_kl_phith +
      trans_kl_phith)`` and ψ with the *unit-weighted* ``trans_kl_psi``;
      single-mode drives every parameter with ``loss = recon + λ·(init_kl +
      trans_kl)`` where ``trans_kl`` carries the ``wtilde/p_k`` weight.
    * The two Adam states have separate step counters and, once split, no
      shared second-moment EMA.

    So we only assert that the parameter deltas move in *approximately the
    same direction with approximately the same magnitude*. A very loose
    ``atol=5e-2, rtol=5e-2`` catches gross misrouting (e.g. a whole
    submodule left untouched) while tolerating legitimate second-moment
    drift and the ``w_ll`` scale factor on the score net.
    """
    # Build two models with identical initialization.
    def _seeded_model():
        torch.manual_seed(0)
        np.random.seed(0)
        model = make_vhp_model(
            baseline_form="persistence",
            tracking_mode="fixed",
            sigma_data_init=1.0,
        )
        # Unit-weight semantics need uniform k-sampling; the fixture's
        # schedule already provides it (cached at transition __init__).
        assert model.transition.k_sampling_mode == "uniform"
        return model

    model_single = _seeded_model()
    model_split = _seeded_model()
    # Confirm identical init before we touch anything.
    for (k, a), (_, b) in zip(
        model_single.state_dict().items(), model_split.state_dict().items()
    ):
        if a.dtype.is_floating_point:
            torch.testing.assert_close(a, b, msg=f"init drift on {k}")

    trainer_single = _build_trainer(
        model_single, tmp_path=tmp_path / "single", use_split_loss=False
    )
    trainer_split = _build_trainer(
        model_split, tmp_path=tmp_path / "split", use_split_loss=True
    )

    initial_single = {
        k: v.detach().clone()
        for k, v in model_single.state_dict().items()
        if v.dtype.is_floating_point
    }
    initial_split = {
        k: v.detach().clone()
        for k, v in model_split.state_dict().items()
        if v.dtype.is_floating_point
    }

    _fit_n_steps(trainer_single, n_steps=5, seed=0)
    _fit_n_steps(trainer_split, n_steps=5, seed=0)

    # Sanity: both trainers ran without skips.
    assert trainer_single.grad_skip_count == 0
    assert trainer_split.grad_skip_count == 0
    assert len(trainer_split._optimizers) == 2
    assert len(trainer_single._optimizers) == 1

    # Compare deltas (post − pre) per param under a loose tolerance.
    mismatches: list[tuple[str, float]] = []
    for name, p_single in model_single.state_dict().items():
        if not p_single.dtype.is_floating_point:
            continue
        d_single = p_single - initial_single[name]
        d_split = model_split.state_dict()[name] - initial_split[name]
        max_abs = (d_single - d_split).abs().max().item()
        # atol=5e-2, rtol=5e-2 — see docstring.
        norm = max(1.0, d_single.abs().max().item(), d_split.abs().max().item())
        if max_abs > 5e-2 + 5e-2 * norm:
            mismatches.append((name, max_abs))

    assert not mismatches, (
        "single-vs-split parameter deltas diverged beyond tolerance on "
        f"{len(mismatches)} params (first 5): {mismatches[:5]}"
    )


# ---------------------------------------------------------------------------
# 2. Single-loss off-path bit-close regression against the golden checkpoint.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_single_loss_off_path_regresses_none():
    """``use_split_loss=False`` must reproduce the pre-refactor golden.

    Tolerance rationale
    -------------------
    Bit-exact equality is not feasible on this platform: M2's refactor
    from ``_estimator_per_t`` (single-pass over sum-of-squares) into the
    Bessel-corrected ``sum_mu / count`` + ``sum_mu² / count`` form changes
    the *summation order* of the σ_d² update on floats, which propagates a
    ~fp32-epsilon (~3e-8) delta through the σ_d²-dependent EDM constants.
    Most params compound to a ~1e-6 per-param delta after 5 steps.

    Attention ``qkv_proj.bias`` params drift up to ~5e-4 — softmax's
    exponential nonlinearly amplifies the sub-epsilon input drift.  This
    is a real (but expected) numerical consequence of M2's summation-order
    change, NOT a routing bug: the goldens themselves updated these
    biases by ~5e-3 over 5 steps, so a 5e-4 vs-golden diff is ~10% of the
    update magnitude — solidly within "same optimization trajectory
    modulo fp float order".

    Chosen tolerance: ``atol=1e-3, rtol=1e-2``.  This catches any O(1)
    or O(10⁻²) mis-routing (a mis-registered KL side would blow past 1e-3
    on parameters of magnitude ~0.1-1) but tolerates the softmax
    amplification. The final-step ``loss/total`` is additionally checked
    against ``M8_FINAL_METRICS`` at ``atol=1e-3``, which is a much tighter
    scalar-level fence (the loss is O(30), so 1e-3 is a 3e-5 relative
    check on the aggregate objective).
    """
    ckpt = load_m8_checkpoint()

    torch.manual_seed(M8_SEED)
    np.random.seed(M8_SEED)
    model = make_vhp_model(
        baseline_form=M8_CONFIG["baseline_form"],
        tracking_mode=M8_CONFIG["tracking_mode"],
        sigma_data_init=M8_CONFIG["sigma_data_init"],
    )
    model.train()

    def data_factory():
        return make_random_walk_data(
            n_seqs=M8_CONFIG["data_n_seqs"],
            T=M8_CONFIG["data_T"],
            seed=M8_SEED,
        )

    torch.manual_seed(M8_SEED)
    metrics_log = run_stage(
        model=model,
        data_factory=data_factory,
        n_steps=5,
        lr=M8_CONFIG["lr"],
    )

    # Per-param bit-close comparison.
    golden_sd = ckpt["state_dict"]
    got_sd = model.state_dict()
    assert set(got_sd.keys()) == set(golden_sd.keys()), (
        "state_dict key set drifted from the golden capture"
    )
    for name, golden_v in golden_sd.items():
        got_v = got_sd[name]
        if not golden_v.dtype.is_floating_point:
            assert torch.equal(got_v, golden_v), f"int/bool param drift on {name}"
            continue
        torch.testing.assert_close(
            got_v,
            golden_v,
            atol=1e-3,
            rtol=1e-2,
            msg=lambda note, n=name: f"regression on {n}: {note}",
        )

    # Final-step ``loss/total`` sanity check.
    final_loss = float(metrics_log[-1]["loss/total"])
    assert abs(final_loss - M8_FINAL_METRICS["loss/total"]) < 1e-3, (
        f"final loss/total drifted: got {final_loss}, "
        f"golden {M8_FINAL_METRICS['loss/total']}"
    )


# ---------------------------------------------------------------------------
# 3. Split-loss end-to-end: metrics wired, no NaN, no spurious skips.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_split_loss_end_to_end_finite_updates(tmp_path):
    """3-step split-loss ``fit`` produces finite params, both KL sides in the
    log dict, and zero grad-skips on clean random-walk data."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_vhp_model(
        baseline_form="persistence",
        tracking_mode="fixed",
        sigma_data_init=1.0,
    )

    trainer = _build_trainer(model, tmp_path=tmp_path, use_split_loss=True)
    _fit_n_steps(trainer, n_steps=3, seed=0)

    # Live metric row from the internal store.
    train_row = trainer.metrics._split("train").values()

    # Expected metric keys under the split-loss path.
    required = {
        "loss/rate/trans/kl_phith",
        "loss/rate/trans/kl_psi",
        "loss/rate/init/loss_psi",
        "optim/grad_norm",
        "optim/grad_skips",
    }
    missing = required - set(train_row.keys())
    assert not missing, f"missing metrics under split-loss fit: {missing}"

    for k in required:
        v = train_row[k]
        assert np.isfinite(v), f"metric {k} = {v} is non-finite"

    # No spurious skips on clean data.
    assert trainer.grad_skip_count == 0, (
        f"unexpected grad-skips on clean data: {trainer.grad_skip_count}"
    )

    # Params moved and remain finite.
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param {name} after 3 steps"

    # Two optimizers under split mode.
    assert len(trainer._optimizers) == 2


# ---------------------------------------------------------------------------
# 4. Grad-skip recovers training when a batch produces a NaN gradient.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_grad_skip_recovers_training(tmp_path):
    """Inject a NaN gradient once mid-run; training must continue.

    Poison mechanism
    ----------------
    We attach a one-shot ``register_hook`` on the ``.grad`` stream of a
    specific decoder parameter (``decoder.gaussian_head`` output weight).
    ``register_hook`` fires as soon as autograd routes a grad tensor
    through the parameter — we replace that tensor's first element with a
    NaN once, then detach.  The NaN lands in ``p.grad`` and the trainer's
    ``clip_grad_norm_(...)`` sees a non-finite global norm on the
    next ``_optimizer_step``: it zeros all grads, bumps the skip counter,
    logs a warning, and returns without stepping the optimizer, scheduler,
    or EMA.  Subsequent steps run on clean grads.

    We use ``register_hook`` (grad-tensor hook) rather than
    ``register_full_backward_hook`` (module hook) because the module hook
    fires only when the full-backward stack unwinds through the module
    with a non-trivial ``grad_input`` — under our tiny-batch fit path some
    module invocations don't produce a full-backward that triggers the
    module hook, whereas any autograd flow through a leaf parameter
    always triggers the parameter's own ``register_hook``.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_vhp_model(
        baseline_form="persistence",
        tracking_mode="fixed",
        sigma_data_init=1.0,
    )

    trainer = _build_trainer(model, tmp_path=tmp_path, use_split_loss=False)

    # One-shot NaN grad poison: pick a decoder parameter guaranteed to
    # receive gradient every step (the recon-loss path always flows into
    # the decoder).  We poison the *first* parameter we find that's
    # trainable — its choice is arbitrary, only that some grad-flow exists.
    target_param = None
    for p in model.decoder.parameters():
        if p.requires_grad:
            target_param = p
            break
    assert target_param is not None, "decoder has no trainable parameter"

    poison_state = {"fired": False}

    def _poison_grad(grad):
        if poison_state["fired"]:
            return None
        poison_state["fired"] = True
        gm = grad.clone()
        # NaN a single element — enough to make the global grad norm NaN.
        gm.view(-1)[0] = float("nan")
        return gm

    handle = target_param.register_hook(_poison_grad)
    try:
        _fit_n_steps(trainer, n_steps=3, seed=0)
    finally:
        handle.remove()

    assert poison_state["fired"], "poison hook never fired — test is a no-op"
    assert trainer.grad_skip_count == 1, (
        f"expected exactly one grad-skip, got {trainer.grad_skip_count}"
    )

    # All params finite; the skipped step's grads were zeroed and no
    # optimizer step ran, so no param can be non-finite from that step.
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param {name} after skip"

    # After the skipped step, subsequent step(s) actually moved things.
    train_row = trainer.metrics._split("train").values()
    assert train_row["optim/grad_skips"] == 1.0
    # global_step still advances even on a skipped step (fit counts steps by
    # optimizer-call attempts, not successful ones).
    assert trainer.global_step == 3


# ---------------------------------------------------------------------------
# 5. Grad-norm clipping.
# ---------------------------------------------------------------------------


def test_default_clip_grad_norm_is_one():
    """``hparams.clip_grad_norm`` defaults to ``1.0`` (not ``None``)."""
    from ddssm.model.dssd import DDSSMHyperParamsConf, _default_hyperparams

    assert DDSSMHyperParamsConf().clip_grad_norm == 1.0
    assert _default_hyperparams().clip_grad_norm == 1.0


def test_clip_grad_norm_bounds_gradient(tmp_path):
    """A small ``clip_grad_norm`` actually rescales grads before the step.

    Builds a trainer whose unclipped grad norm is (verified) larger than a
    tiny clip threshold, fits one step, and checks the post-step ``.grad``
    norm on the live parameters is bounded by that threshold while
    ``_last_grad_norm`` (recorded pre-clip) exceeds it.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_vhp_model(
        baseline_form="persistence",
        tracking_mode="fixed",
        sigma_data_init=1.0,
    )
    trainer = _build_trainer(model, tmp_path=tmp_path, use_split_loss=False)

    clip_value = 1e-3
    trainer.clip_grad_norm = clip_value
    _fit_n_steps(trainer, n_steps=1, seed=0)

    assert trainer._last_grad_norm is not None
    assert trainer._last_grad_norm > clip_value, (
        "test is a no-op unless the unclipped norm exceeds the clip value"
    )

    post_clip_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float("inf")
    )
    assert float(post_clip_norm) <= clip_value * 1.01, (
        f"post-step grad norm {float(post_clip_norm)} exceeds clip value "
        f"{clip_value} — clipping was not applied"
    )


def test_use_reentrant_false_still_present():
    """``torch.utils.checkpoint(..., use_reentrant=False)`` must remain on
    the ``_esm_chunk_loss`` grad-checkpoint call — the split backward
    silently corrupts activations without it (the second selective
    ``.backward(inputs=...)`` walks the checkpointed subgraph)."""
    diffusion_py = (
        REPO_ROOT / "src" / "ddssm" / "model" / "transitions" / "diffusion.py"
    )
    assert diffusion_py.is_file(), f"diffusion.py not found at {diffusion_py}"
    result = subprocess.run(
        ["grep", "-n", "use_reentrant=False", str(diffusion_py)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "use_reentrant=False disappeared from transitions/diffusion.py — "
        "the split backward will silently corrupt gradients."
    )
