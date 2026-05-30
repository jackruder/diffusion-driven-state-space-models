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
