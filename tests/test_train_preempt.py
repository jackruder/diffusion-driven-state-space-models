"""Tests for preempt-aware signal handling in :class:`DDSSMTrainer`.

Covers ADR-0009 Phase 2c: ``PreemptError``, the SIGUSR1/SIGTERM
flag-setting signal handler, and the ``fit()`` loop's flag check that
saves a checkpoint and raises ``PreemptError(resume_from=...)``.
"""

from __future__ import annotations

import os
import signal
from functools import partial
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ddssm.aggregators import IdentityAggregator
from ddssm.aux_posterior import AuxPosterior
from ddssm.combiners import CompoundCombiner
from ddssm.decoder import GaussianDecoder
from ddssm.diffnets import ContextProducer, FeatureMixerConfig, ResidualBlockConfig
from ddssm.dist_heads import GaussianDistHead
from ddssm.dssd import DDSSM_base
from ddssm.encoder import GaussianEncoder
from ddssm.fusions import ConcatLinearFusion
from ddssm.futsum import GRUFutureSummary
from ddssm.gaussians import GaussianHead
from ddssm.train import DDSSMTrainer, PreemptError
from ddssm.transitions.transitions import GaussianTransition


# ---------------------------------------------------------------------------
# Minimal-model fixtures (mirror tests/test_trainer.py).
# ---------------------------------------------------------------------------

J = 1
DATA_DIM = 3
LATENT_DIM = 2
EMB_TIME = 8
CHANNELS = 8
NHEADS = 4

_CTX = partial(
    ContextProducer,
    channels=CHANNELS,
    num_layers=1,
    residual_block=ResidualBlockConfig(
        feature=FeatureMixerConfig(nheads=NHEADS, n_layers=1)
    ),
)
_GH = GaussianHead
_FS = partial(GRUFutureSummary, summary_dim=CHANNELS, num_layers=1)


def _make_small_model() -> DDSSM_base:
    combiner = partial(
        CompoundCombiner,
        aggregator=partial(IdentityAggregator),
        fusion=partial(ConcatLinearFusion),
    )
    enc = GaussianEncoder(
        data_dim=DATA_DIM, latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        use_mask=True, hidden_dim=CHANNELS,
        combiner=combiner,
        dist_head=partial(GaussianDistHead),
        fut_summary=_FS,
    )
    dec = GaussianDecoder(
        latent_dim=LATENT_DIM, data_dim=DATA_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH,
    )
    trans = GaussianTransition(
        latent_dim=LATENT_DIM, j=J, emb_time_dim=EMB_TIME,
        hidden_dim=CHANNELS,
        context=_CTX, gaussian_head=_GH,
    )
    aux = AuxPosterior(latent_dim=LATENT_DIM, j=J, hidden_dim=CHANNELS, n_layers=1)
    return DDSSM_base(
        encoder=enc, decoder=dec, transition=trans, aux_posterior=aux,
        j=J, data_dim=DATA_DIM, latent_dim=LATENT_DIM, emb_time_dim=EMB_TIME,
    )


class _SyntheticBatchDataset(Dataset):
    def __init__(self, B: int = 2, T: int = 4):
        self.B = B
        self.T = T

    def __len__(self) -> int:
        return self.B

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "observed_data": torch.randn(DATA_DIM, self.T),
            "observation_mask": torch.ones(DATA_DIM, self.T),
            "timepoints": torch.arange(self.T, dtype=torch.float32),
        }


def _make_trainer(tmp_path, **kwargs: Any) -> DDSSMTrainer:
    return DDSSMTrainer(
        model=_make_small_model(),
        device=torch.device("cpu"),
        tensorboard_dir=str(tmp_path / "tb"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        quiet=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_preempt_error_carries_resume_from() -> None:
    err = PreemptError("/tmp/x.pth")
    assert err.resume_from == "/tmp/x.pth"
    assert isinstance(err, RuntimeError)


def test_signal_handler_sets_flag_without_raising(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)
    assert trainer._preempt_pending is False
    # Direct invocation — no exception.
    trainer._handle_preempt_signal(signal.SIGUSR1, None)
    assert trainer._preempt_pending is True


def test_fit_saves_ckpt_and_raises_preempt_error_after_signal(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)

    # Set the preempt flag after the first call to _log_train_step (i.e.
    # immediately after step 1 completes). This mirrors the runtime path
    # where the OS signal handler sets the flag asynchronously between
    # the per-step instructions of the fit loop.
    real_log = trainer._log_train_step
    call_count = {"n": 0}

    def _spy_log(*args, **kwargs):
        real_log(*args, **kwargs)
        call_count["n"] += 1
        if call_count["n"] == 1:
            trainer._preempt_pending = True

    trainer._log_train_step = _spy_log  # type: ignore[assignment]

    with pytest.raises(PreemptError) as excinfo:
        trainer.fit(
            train_loader=loader,
            val_loader=None,
            total_steps=20,
            validate_every=0,
            log_every=1,
            checkpoint_every=5,
            checkpoint_prefix="preempt_test",
            amp=False,
        )

    err = excinfo.value
    assert err.resume_from, "PreemptError must carry a non-empty resume_from path"
    assert os.path.isfile(err.resume_from), (
        f"resume_from path {err.resume_from!r} must point to a real file on disk"
    )
    # File should be loadable as a torch checkpoint.
    loaded = torch.load(err.resume_from, map_location="cpu", weights_only=False)
    assert isinstance(loaded, dict), "checkpoint payload must be a dict"
    # ADR-0009 multi-stage resume: the payload carries the stage prefix the
    # caller passed into fit(checkpoint_prefix=...) so the orchestrator can
    # resume into the right stage on retry.
    assert loaded.get("stage_prefix") == "preempt_test", (
        "checkpoint payload must carry stage_prefix matching fit(checkpoint_prefix=...)"
    )


def test_resume_from_preempt_ckpt_continues_global_step(tmp_path) -> None:
    # First trainer: run, preempt after one step, capture the ckpt.
    trainer1 = _make_trainer(tmp_path)
    loader = DataLoader(_SyntheticBatchDataset(B=2, T=4), batch_size=2)

    real_log = trainer1._log_train_step
    call_count = {"n": 0}

    def _spy_log(*args, **kwargs):
        real_log(*args, **kwargs)
        call_count["n"] += 1
        if call_count["n"] == 1:
            trainer1._preempt_pending = True

    trainer1._log_train_step = _spy_log  # type: ignore[assignment]

    with pytest.raises(PreemptError) as excinfo:
        trainer1.fit(
            train_loader=loader,
            val_loader=None,
            total_steps=20,
            validate_every=0,
            log_every=1,
            checkpoint_every=5,
            checkpoint_prefix="preempt_test",
            amp=False,
        )

    saved_path = excinfo.value.resume_from
    saved_step = trainer1.global_step
    assert saved_step >= 1, "trainer should have taken at least one step before preempt"

    # Second trainer: resume from the saved ckpt and train a few more steps.
    trainer2 = _make_trainer(tmp_path)
    assert trainer2.global_step == 0  # fresh start
    trainer2.fit(
        train_loader=loader,
        val_loader=None,
        total_steps=saved_step + 5,
        validate_every=0,
        log_every=1,
        checkpoint_every=None,
        amp=False,
        resume_from=saved_path,
    )
    assert trainer2.global_step > saved_step, (
        f"resumed trainer should advance past saved step "
        f"({trainer2.global_step} <= {saved_step})"
    )


def _is_bound_method_of(handler, instance, method_name: str) -> bool:
    """Robust comparison: bound methods aren't ``is``-equal across rebinds."""
    return (
        callable(handler)
        and getattr(handler, "__self__", None) is instance
        and getattr(handler, "__func__", None)
        is getattr(type(instance), method_name, None)
    )


def test_sigint_handler_only_installed_under_DDSSM_PREEMPTIVE_env(
    tmp_path, monkeypatch
) -> None:
    # Snapshot the original SIGINT handler so we can restore it.
    original_sigint = signal.getsignal(signal.SIGINT)
    try:
        # Without the env var: SIGINT must remain Python's default.
        monkeypatch.delenv("DDSSM_PREEMPTIVE", raising=False)
        # Reset SIGINT to default before constructing trainer.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        trainer_no_env = _make_trainer(tmp_path)
        # Trainer must NOT have installed its handler on SIGINT.
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, (
            "SIGINT must remain Python's default when DDSSM_PREEMPTIVE is unset"
        )
        assert not _is_bound_method_of(
            signal.getsignal(signal.SIGINT),
            trainer_no_env,
            "_handle_preempt_signal",
        )

        # With the env var: SIGINT routes through the trainer handler.
        monkeypatch.setenv("DDSSM_PREEMPTIVE", "1")
        trainer_env = _make_trainer(tmp_path)
        assert _is_bound_method_of(
            signal.getsignal(signal.SIGINT),
            trainer_env,
            "_handle_preempt_signal",
        ), (
            "SIGINT must be routed through the trainer handler under "
            "DDSSM_PREEMPTIVE=1"
        )
    finally:
        # Restore original SIGINT to avoid cross-test contamination.
        try:
            signal.signal(signal.SIGINT, original_sigint)
        except (TypeError, ValueError):
            signal.signal(signal.SIGINT, signal.default_int_handler)


def test_periodic_checkpoint_pair_is_atomic_on_fault(tmp_path, monkeypatch) -> None:
    """SIGKILL between the step-N write and the latest write must not strand
    a stale ``ckpt_latest`` pointing to N-K while step-N is on disk.

    Simulates the fault by patching ``os.replace`` to raise the second time
    it would advance ``ckpt_latest`` (i.e. after step-N is already durable on
    disk). ``ckpt_latest`` must still load as a valid checkpoint — either the
    prior snapshot or step-N itself — and must never reference a half-written
    or missing file.
    """
    trainer = _make_trainer(tmp_path)
    # Establish a first valid pair so ``ckpt_latest`` exists on disk and is
    # loadable. This is the "N-K" snapshot the resume path would fall back to.
    first_latest = trainer._save_periodic_checkpoint(step=5, checkpoint_prefix="atom_test")
    assert os.path.isfile(first_latest)
    prior_payload = torch.load(first_latest, map_location="cpu", weights_only=False)
    assert prior_payload.get("global_step") == 0
    latest_path = first_latest
    step_n_path = os.path.join(
        trainer.checkpoint_dir, "ckpt_atom_test_step10.pth",
    )

    # Bump trainer state so the would-be step-N ckpt is distinguishable from
    # the prior snapshot. We don't actually train — just advance the counter.
    trainer.global_step = 10

    # Inject a fault into the latest-pointer transition only. The step-N write
    # goes through ``Checkpoint.save`` -> ``_atomic_save`` (its own
    # ``os.replace``) and must complete normally. Patching by call ordinal
    # is fragile; instead, patch by the target path so only the
    # ``ckpt_*_latest.pth`` transition raises.
    import os as _os
    real_replace = _os.replace

    def _faulty_replace(src, dst, *args, **kwargs):
        if str(dst).endswith("ckpt_atom_test_latest.pth"):
            raise OSError("simulated SIGKILL between step-N and latest writes")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr("ddssm.train.os.replace", _faulty_replace)

    with pytest.raises(OSError, match="simulated SIGKILL"):
        trainer._save_periodic_checkpoint(step=10, checkpoint_prefix="atom_test")

    # Invariant: ``ckpt_latest`` still points to a fully-written, loadable
    # checkpoint. It may be either the prior snapshot (preferred — that's
    # what the fix guarantees) or step-N, but never a half-written tmp file.
    assert os.path.isfile(latest_path), (
        "ckpt_latest must still exist after the fault"
    )
    payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict) and "model_state" in payload, (
        "ckpt_latest must remain a fully-written, loadable checkpoint"
    )
    # Step-N's own file should have been written durably before the fault —
    # the fix writes it first via ``_atomic_save``, so the resume path can
    # always recover at least up to step N from this file.
    assert os.path.isfile(step_n_path), (
        "step-N checkpoint must be durably on disk before the latest pointer "
        "is advanced"
    )
    step_n_payload = torch.load(step_n_path, map_location="cpu", weights_only=False)
    assert step_n_payload.get("global_step") == 10

    # No leftover tmp file in the checkpoint dir — the fault path must clean
    # up its same-dir tmp so a sweep-of-sweeps doesn't leak disk on Lustre.
    leftovers = [
        n for n in os.listdir(trainer.checkpoint_dir)
        if n.startswith("tmp_latest_") or n.startswith("tmp_save_")
    ]
    assert not leftovers, f"leftover tmp files after fault: {leftovers}"
